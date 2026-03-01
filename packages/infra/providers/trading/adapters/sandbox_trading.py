"""Internal sandbox trading adapter backed by Alpaca market-data feeds."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from uuid import uuid4

from packages.infra.db import session as db_session_module
from packages.infra.db.models.sandbox_ledger_entry import SandboxLedgerEntry
from packages.infra.observability.logger import logger
from packages.infra.providers.market_data.alpaca_rest import AlpacaRestProvider
from packages.infra.providers.trading.adapters.base import (
    AccountState,
    BrokerAdapter,
    FillRecord,
    MarketDataEvent,
    OhlcvBar,
    OrderIntent,
    OrderState,
    PositionRecord,
    QuoteSnapshot,
)
from packages.infra.redis.client import get_sync_redis_client

_CRYPTO_QUOTES = ("USDT", "USDC", "USD", "BTC", "ETH", "EUR")
_ASSET_CLASSES = ("us_equity", "crypto", "forex", "futures")
_FX_CURRENCIES = {
    "USD",
    "EUR",
    "JPY",
    "GBP",
    "CHF",
    "CAD",
    "AUD",
    "NZD",
    "CNY",
    "HKD",
}
_MARKET_TO_ASSET_CLASS = {
    "stocks": "us_equity",
    "equity": "us_equity",
    "us_equity": "us_equity",
    "crypto": "crypto",
    "forex": "forex",
    "fx": "forex",
    "futures": "futures",
    "future": "futures",
}


def _to_decimal(value: Any, *, default: str = "0") -> Decimal:
    if value is None:
        return Decimal(default)
    try:
        return Decimal(str(value))
    except Exception:  # noqa: BLE001
        return Decimal(default)


def _normalize_symbol(symbol: str) -> str:
    text = str(symbol or "").strip().upper()
    if not text:
        return "BTC/USD"
    if "/" in text:
        base, quote = text.split("/", 1)
        if base and quote:
            return f"{base}/{quote}"
    compact = text.replace("-", "").replace("/", "")
    for quote in _CRYPTO_QUOTES:
        if compact.endswith(quote) and len(compact) > len(quote):
            base = compact[: -len(quote)]
            return f"{base}/{quote}"
    return compact


def _infer_market(symbol: str) -> str:
    normalized = _normalize_symbol(symbol)
    if "/" in normalized:
        return "crypto"
    compact = normalized.replace("/", "")
    for quote in _CRYPTO_QUOTES:
        if compact.endswith(quote) and len(compact) > len(quote):
            return "crypto"
    return "stocks"


def _normalize_asset_class(value: Any) -> str | None:
    normalized = str(value or "").strip().lower()
    if normalized in _ASSET_CLASSES:
        return normalized
    mapped = _MARKET_TO_ASSET_CLASS.get(normalized)
    if mapped in _ASSET_CLASSES:
        return mapped
    return None


def _infer_asset_class(symbol: str, *, market_hint: str | None = None) -> str:
    hinted = _normalize_asset_class(market_hint)
    if hinted is not None:
        return hinted

    normalized = _normalize_symbol(symbol)
    if "/" in normalized:
        base, quote = normalized.split("/", 1)
        if base in _FX_CURRENCIES and quote in _FX_CURRENCIES:
            return "forex"
        return "crypto"
    if normalized.endswith("=F"):
        return "futures"
    return "us_equity"


def _build_bps_profile(
    *,
    default_bps: Decimal,
    overrides: dict[str, Any] | None,
) -> dict[str, Decimal]:
    profile = {asset_class: default_bps for asset_class in _ASSET_CLASSES}
    raw = overrides if isinstance(overrides, dict) else {}
    for key, value in raw.items():
        normalized_key = _normalize_asset_class(key)
        if normalized_key is None:
            continue
        profile[normalized_key] = max(_to_decimal(value, default=str(default_bps)), Decimal("0"))
    return profile


def _serialize_position(
    *,
    qty: Decimal,
    avg_entry_price: Decimal,
    mark_price: Decimal,
    realized_pnl: Decimal,
) -> dict[str, str]:
    return {
        "qty": str(qty),
        "avg_entry_price": str(avg_entry_price),
        "mark_price": str(mark_price),
        "realized_pnl": str(realized_pnl),
    }


class SandboxTradingAdapter(BrokerAdapter):
    """Internal paper-execution adapter with Alpaca market data as source-of-truth."""

    provider = "sandbox"

    def __init__(
        self,
        *,
        account_uid: str,
        starting_cash: Decimal | float | str = Decimal("100000"),
        slippage_bps: Decimal | float | str = Decimal("0"),
        fee_bps: Decimal | float | str = Decimal("0"),
        slippage_bps_by_asset_class: dict[str, Any] | None = None,
        fee_bps_by_asset_class: dict[str, Any] | None = None,
        market_data_provider: AlpacaRestProvider | None = None,
    ) -> None:
        normalized_uid = str(account_uid or "").strip()
        if not normalized_uid:
            normalized_uid = f"sandbox-{uuid4().hex[:16]}"
        self._account_uid = normalized_uid
        self._starting_cash = _to_decimal(starting_cash, default="100000")
        self._slippage_bps = max(_to_decimal(slippage_bps, default="0"), Decimal("0"))
        self._fee_bps = max(_to_decimal(fee_bps, default="0"), Decimal("0"))
        self._slippage_bps_profile = _build_bps_profile(
            default_bps=self._slippage_bps,
            overrides=slippage_bps_by_asset_class,
        )
        self._fee_bps_profile = _build_bps_profile(
            default_bps=self._fee_bps,
            overrides=fee_bps_by_asset_class,
        )
        self._market_data_provider = market_data_provider or AlpacaRestProvider()
        self._owns_market_data_provider = market_data_provider is None
        self._state_key = f"sandbox:account:{self._account_uid}:state"
        self._orders_key = f"sandbox:account:{self._account_uid}:orders"
        self._fills_key = f"sandbox:account:{self._account_uid}:fills"

    def _resolve_slippage_bps(self, asset_class: str) -> Decimal:
        return self._slippage_bps_profile.get(asset_class, self._slippage_bps)

    def _resolve_fee_bps(self, asset_class: str) -> Decimal:
        return self._fee_bps_profile.get(asset_class, self._fee_bps)

    async def _load_state(self) -> dict[str, Any]:
        redis = get_sync_redis_client()
        raw = await asyncio.to_thread(redis.get, self._state_key)
        if isinstance(raw, str) and raw.strip():
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
        return {
            "account_uid": self._account_uid,
            "cash": str(self._starting_cash),
            "positions": {},
            "updated_at": datetime.now(UTC).isoformat(),
        }

    async def _save_state(self, state: dict[str, Any]) -> None:
        redis = get_sync_redis_client()
        await asyncio.to_thread(redis.set, self._state_key, json.dumps(state, separators=(",", ":")))

    async def _save_order(self, payload: dict[str, Any]) -> None:
        redis = get_sync_redis_client()
        await asyncio.to_thread(
            redis.hset,
            self._orders_key,
            str(payload["provider_order_id"]),
            json.dumps(payload, separators=(",", ":")),
        )

    async def _append_fill(self, payload: dict[str, Any]) -> None:
        redis = get_sync_redis_client()
        encoded = json.dumps(payload, separators=(",", ":"))
        await asyncio.to_thread(redis.rpush, self._fills_key, encoded)
        await asyncio.to_thread(redis.ltrim, self._fills_key, -2000, -1)

    async def _append_ledger_entry(self, payload: dict[str, Any]) -> None:
        session_factory = db_session_module.AsyncSessionLocal
        if session_factory is None:
            return
        happened_at_raw = payload.get("happened_at")
        if isinstance(happened_at_raw, datetime):
            happened_at = happened_at_raw.astimezone(UTC)
        elif isinstance(happened_at_raw, str):
            try:
                happened_at = datetime.fromisoformat(happened_at_raw).astimezone(UTC)
            except ValueError:
                happened_at = datetime.now(UTC)
        else:
            happened_at = datetime.now(UTC)
        metadata = payload.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        try:
            async with session_factory() as db:
                db.add(
                    SandboxLedgerEntry(
                        account_uid=self._account_uid,
                        provider_order_id=(
                            str(payload.get("provider_order_id")).strip()
                            if payload.get("provider_order_id") is not None
                            else None
                        ),
                        client_order_id=(
                            str(payload.get("client_order_id")).strip()
                            if payload.get("client_order_id") is not None
                            else None
                        ),
                        event_type="order_fill",
                        symbol=str(payload.get("symbol") or ""),
                        asset_class=str(payload.get("asset_class") or "unknown"),
                        side=str(payload.get("side") or "buy"),
                        qty=_to_decimal(payload.get("qty"), default="0"),
                        fill_price=_to_decimal(payload.get("fill_price"), default="0"),
                        notional=_to_decimal(payload.get("notional"), default="0"),
                        fee=_to_decimal(payload.get("fee"), default="0"),
                        fee_bps=_to_decimal(payload.get("fee_bps"), default="0"),
                        slippage_bps=_to_decimal(payload.get("slippage_bps"), default="0"),
                        cash_before=_to_decimal(payload.get("cash_before"), default="0"),
                        cash_after=_to_decimal(payload.get("cash_after"), default="0"),
                        position_qty_before=_to_decimal(payload.get("position_qty_before"), default="0"),
                        position_qty_after=_to_decimal(payload.get("position_qty_after"), default="0"),
                        avg_entry_before=_to_decimal(payload.get("avg_entry_before"), default="0"),
                        avg_entry_after=_to_decimal(payload.get("avg_entry_after"), default="0"),
                        realized_pnl_before=_to_decimal(payload.get("realized_pnl_before"), default="0"),
                        realized_pnl_after=_to_decimal(payload.get("realized_pnl_after"), default="0"),
                        metadata_=metadata,
                        happened_at=happened_at,
                    )
                )
                await db.commit()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[sandbox-ledger] failed to append entry account_uid=%s error=%s",
                self._account_uid,
                type(exc).__name__,
            )

    async def _resolve_fill_price(
        self,
        *,
        symbol: str,
        side: str,
        slippage_bps: Decimal | None = None,
    ) -> tuple[Decimal, datetime]:
        quote = await self.fetch_latest_quote(symbol)
        if quote is not None:
            last = _to_decimal(quote.last, default="0")
            bid = _to_decimal(quote.bid, default="0")
            ask = _to_decimal(quote.ask, default="0")
            if side == "buy":
                base = ask if ask > 0 else last if last > 0 else bid
            else:
                base = bid if bid > 0 else last if last > 0 else ask
            if base > 0:
                effective_slippage_bps = slippage_bps if slippage_bps is not None else self._slippage_bps
                slippage = effective_slippage_bps / Decimal("10000")
                adjusted = base * (Decimal("1") + slippage if side == "buy" else Decimal("1") - slippage)
                return adjusted, quote.timestamp
        latest_bar = await self.fetch_latest_1m_bar(symbol)
        if latest_bar is not None:
            return _to_decimal(latest_bar.close, default="0"), latest_bar.timestamp
        return Decimal("0"), datetime.now(UTC)

    async def fetch_account_state(self) -> AccountState:
        state = await self._load_state()
        cash = _to_decimal(state.get("cash"), default=str(self._starting_cash))
        positions_map = state.get("positions")
        positions_map = positions_map if isinstance(positions_map, dict) else {}

        net_position_value = Decimal("0")
        unrealized = Decimal("0")
        for raw_symbol, raw_position in positions_map.items():
            if not isinstance(raw_position, dict):
                continue
            symbol = _normalize_symbol(str(raw_symbol))
            qty_signed = _to_decimal(raw_position.get("qty"), default="0")
            if qty_signed == 0:
                continue
            avg_entry = _to_decimal(raw_position.get("avg_entry_price"), default="0")
            mark = _to_decimal(raw_position.get("mark_price"), default="0")
            quote = await self.fetch_latest_quote(symbol)
            if quote is not None and quote.last is not None:
                mark = _to_decimal(quote.last, default=str(mark))
            if mark <= 0:
                mark = avg_entry
            raw_position["mark_price"] = str(mark)
            net_position_value += qty_signed * mark
            if qty_signed > 0:
                unrealized += (mark - avg_entry) * qty_signed
            else:
                unrealized += (avg_entry - mark) * abs(qty_signed)

        equity = cash + net_position_value
        buying_power = max(cash, Decimal("0"))
        state["updated_at"] = datetime.now(UTC).isoformat()
        await self._save_state(state)
        return AccountState(
            cash=cash,
            equity=equity,
            buying_power=buying_power,
            margin_used=Decimal("0"),
            raw={"account_uid": self._account_uid, "unrealized_pnl": str(unrealized)},
        )

    async def fetch_positions(self) -> list[PositionRecord]:
        state = await self._load_state()
        positions_map = state.get("positions")
        positions_map = positions_map if isinstance(positions_map, dict) else {}
        rows: list[PositionRecord] = []
        dirty = False
        for raw_symbol, raw_position in positions_map.items():
            if not isinstance(raw_position, dict):
                continue
            symbol = _normalize_symbol(str(raw_symbol))
            qty_signed = _to_decimal(raw_position.get("qty"), default="0")
            if qty_signed == 0:
                continue
            avg_entry = _to_decimal(raw_position.get("avg_entry_price"), default="0")
            mark_price = _to_decimal(raw_position.get("mark_price"), default="0")
            quote = await self.fetch_latest_quote(symbol)
            if quote is not None and quote.last is not None:
                resolved_mark = _to_decimal(quote.last, default=str(mark_price))
                if resolved_mark > 0 and resolved_mark != mark_price:
                    mark_price = resolved_mark
                    raw_position["mark_price"] = str(mark_price)
                    dirty = True
            if mark_price <= 0:
                mark_price = avg_entry
            side = "long" if qty_signed > 0 else "short"
            qty = abs(qty_signed)
            realized = _to_decimal(raw_position.get("realized_pnl"), default="0")
            if side == "long":
                unrealized = (mark_price - avg_entry) * qty
            else:
                unrealized = (avg_entry - mark_price) * qty
            rows.append(
                PositionRecord(
                    symbol=symbol,
                    side=side,
                    qty=qty,
                    avg_entry_price=avg_entry,
                    mark_price=mark_price,
                    unrealized_pnl=unrealized,
                    realized_pnl=realized,
                    raw=dict(raw_position),
                )
            )
        if dirty:
            state["updated_at"] = datetime.now(UTC).isoformat()
            await self._save_state(state)
        return rows

    async def submit_order(self, intent: OrderIntent) -> OrderState:
        symbol = _normalize_symbol(intent.symbol)
        side = str(intent.side).strip().lower()
        qty = _to_decimal(intent.qty, default="0")
        if qty <= 0:
            raise RuntimeError("Sandbox order qty must be > 0.")
        metadata = intent.metadata if isinstance(intent.metadata, dict) else {}
        market_hint = metadata.get("market")
        market_hint_text = str(market_hint).strip().lower() if market_hint is not None else ""
        asset_class = _infer_asset_class(symbol, market_hint=market_hint_text)
        slippage_bps = self._resolve_slippage_bps(asset_class)
        fee_bps = self._resolve_fee_bps(asset_class)
        fill_price, submitted_at = await self._resolve_fill_price(
            symbol=symbol,
            side=side,
            slippage_bps=slippage_bps,
        )
        if fill_price <= 0:
            raise RuntimeError("Sandbox fill price unavailable from market data.")

        state = await self._load_state()
        cash = _to_decimal(state.get("cash"), default=str(self._starting_cash))
        positions_map = state.get("positions")
        positions_map = positions_map if isinstance(positions_map, dict) else {}
        position = positions_map.get(symbol)
        position = dict(position) if isinstance(position, dict) else {}
        current_qty = _to_decimal(position.get("qty"), default="0")
        avg_entry = _to_decimal(position.get("avg_entry_price"), default="0")
        mark_price = fill_price
        realized = _to_decimal(position.get("realized_pnl"), default="0")
        cash_before = cash
        position_qty_before = current_qty
        avg_entry_before = avg_entry
        realized_before = realized
        notional = qty * fill_price
        fee = (notional * fee_bps) / Decimal("10000")

        if side == "buy":
            cash -= notional + fee
            if current_qty >= 0:
                new_qty = current_qty + qty
                if new_qty > 0:
                    avg_entry = (
                        (current_qty * avg_entry) + (qty * fill_price)
                    ) / new_qty if current_qty > 0 else fill_price
                current_qty = new_qty
            else:
                short_qty = abs(current_qty)
                close_qty = min(short_qty, qty)
                if close_qty > 0:
                    realized += (avg_entry - fill_price) * close_qty
                remainder = qty - close_qty
                current_qty = current_qty + qty
                if current_qty > 0 and remainder > 0:
                    avg_entry = fill_price
                elif current_qty == 0:
                    avg_entry = Decimal("0")
        else:
            cash += notional - fee
            if current_qty <= 0:
                short_qty = abs(current_qty)
                new_short_qty = short_qty + qty
                if new_short_qty > 0:
                    avg_entry = (
                        (short_qty * avg_entry) + (qty * fill_price)
                    ) / new_short_qty if short_qty > 0 else fill_price
                current_qty = Decimal("0") - new_short_qty
            else:
                close_qty = min(current_qty, qty)
                if close_qty > 0:
                    realized += (fill_price - avg_entry) * close_qty
                remainder = qty - close_qty
                current_qty = current_qty - qty
                if current_qty < 0 and remainder > 0:
                    avg_entry = fill_price
                elif current_qty == 0:
                    avg_entry = Decimal("0")

        realized -= fee
        positions_map[symbol] = _serialize_position(
            qty=current_qty,
            avg_entry_price=avg_entry,
            mark_price=mark_price,
            realized_pnl=realized,
        )
        state["cash"] = str(cash)
        state["positions"] = positions_map
        state["updated_at"] = datetime.now(UTC).isoformat()
        await self._save_state(state)

        provider_order_id = f"sandbox-{uuid4().hex[:24]}"
        order_payload = {
            "provider_order_id": provider_order_id,
            "client_order_id": intent.client_order_id,
            "symbol": symbol,
            "side": side,
            "order_type": intent.order_type,
            "qty": str(qty),
            "filled_qty": str(qty),
            "status": "filled",
            "submitted_at": submitted_at.isoformat(),
            "avg_fill_price": str(fill_price),
            "provider_updated_at": submitted_at.isoformat(),
            "raw": {
                "provider": "sandbox",
                "account_uid": self._account_uid,
                "simulated": True,
                "asset_class": asset_class,
                "slippage_bps": str(slippage_bps),
                "fee_bps": str(fee_bps),
                "fee": str(fee),
            },
        }
        await self._save_order(order_payload)
        fill_payload = {
            "provider_fill_id": f"fill-{uuid4().hex[:24]}",
            "provider_order_id": provider_order_id,
            "symbol": symbol,
            "side": side,
            "qty": str(qty),
            "price": str(fill_price),
            "fee": str(fee),
            "filled_at": submitted_at.isoformat(),
        }
        await self._append_fill(fill_payload)
        await self._append_ledger_entry(
            {
                "provider_order_id": provider_order_id,
                "client_order_id": intent.client_order_id,
                "symbol": symbol,
                "asset_class": asset_class,
                "side": side,
                "qty": str(qty),
                "fill_price": str(fill_price),
                "notional": str(notional),
                "fee": str(fee),
                "fee_bps": str(fee_bps),
                "slippage_bps": str(slippage_bps),
                "cash_before": str(cash_before),
                "cash_after": str(cash),
                "position_qty_before": str(position_qty_before),
                "position_qty_after": str(current_qty),
                "avg_entry_before": str(avg_entry_before),
                "avg_entry_after": str(avg_entry),
                "realized_pnl_before": str(realized_before),
                "realized_pnl_after": str(realized),
                "happened_at": submitted_at,
                "metadata": {
                    "provider": "sandbox",
                    "account_uid": self._account_uid,
                    "intent_metadata": metadata,
                },
            }
        )
        return OrderState(
            provider_order_id=provider_order_id,
            client_order_id=intent.client_order_id,
            symbol=symbol,
            side=side,
            order_type=intent.order_type,
            qty=qty,
            filled_qty=qty,
            status="filled",
            submitted_at=submitted_at,
            avg_fill_price=fill_price,
            reject_reason=None,
            provider_updated_at=submitted_at,
            raw={
                "provider": "sandbox",
                "account_uid": self._account_uid,
                "asset_class": asset_class,
                "slippage_bps": str(slippage_bps),
                "fee_bps": str(fee_bps),
                "fee": str(fee),
            },
        )

    async def cancel_order(self, order_id: str) -> bool:
        order = await self.fetch_order(order_id)
        if order is None:
            return False
        if str(order.status).strip().lower() == "filled":
            return False
        redis = get_sync_redis_client()
        payload = {
            "provider_order_id": order.provider_order_id,
            "client_order_id": order.client_order_id,
            "symbol": order.symbol,
            "side": order.side,
            "order_type": order.order_type,
            "qty": str(order.qty),
            "filled_qty": str(order.filled_qty),
            "status": "canceled",
            "submitted_at": order.submitted_at.isoformat() if order.submitted_at else None,
            "avg_fill_price": str(order.avg_fill_price) if order.avg_fill_price is not None else None,
            "provider_updated_at": datetime.now(UTC).isoformat(),
            "raw": dict(order.raw),
        }
        await asyncio.to_thread(redis.hset, self._orders_key, order_id, json.dumps(payload, separators=(",", ":")))
        return True

    async def fetch_order(self, order_id: str) -> OrderState | None:
        redis = get_sync_redis_client()
        raw = await asyncio.to_thread(redis.hget, self._orders_key, order_id)
        if not isinstance(raw, str) or not raw.strip():
            return None
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        submitted_at = payload.get("submitted_at")
        provider_updated_at = payload.get("provider_updated_at")
        avg_fill_raw = payload.get("avg_fill_price")
        avg_fill_price = _to_decimal(avg_fill_raw, default="0") if avg_fill_raw is not None else None
        return OrderState(
            provider_order_id=str(payload.get("provider_order_id", order_id)),
            client_order_id=str(payload.get("client_order_id", "")),
            symbol=str(payload.get("symbol", "")),
            side=str(payload.get("side", "buy")),
            order_type=str(payload.get("order_type", "market")),
            qty=_to_decimal(payload.get("qty"), default="0"),
            filled_qty=_to_decimal(payload.get("filled_qty"), default="0"),
            status=str(payload.get("status", "filled")),
            submitted_at=(
                datetime.fromisoformat(submitted_at).astimezone(UTC)
                if isinstance(submitted_at, str)
                else None
            ),
            avg_fill_price=avg_fill_price,
            reject_reason=None,
            provider_updated_at=(
                datetime.fromisoformat(provider_updated_at).astimezone(UTC)
                if isinstance(provider_updated_at, str)
                else None
            ),
            raw=payload.get("raw") if isinstance(payload.get("raw"), dict) else {},
        )

    async def fetch_recent_fills(self, since: datetime | None = None) -> list[FillRecord]:
        redis = get_sync_redis_client()
        rows = await asyncio.to_thread(redis.lrange, self._fills_key, 0, -1)
        records: list[FillRecord] = []
        since_utc = since.astimezone(UTC) if since is not None else None
        for row in rows:
            if not isinstance(row, str):
                continue
            try:
                payload = json.loads(row)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            filled_at_text = payload.get("filled_at")
            if not isinstance(filled_at_text, str):
                continue
            try:
                filled_at = datetime.fromisoformat(filled_at_text).astimezone(UTC)
            except ValueError:
                continue
            if since_utc is not None and filled_at < since_utc:
                continue
            records.append(
                FillRecord(
                    provider_fill_id=str(payload.get("provider_fill_id", "")),
                    provider_order_id=str(payload.get("provider_order_id", "")),
                    symbol=str(payload.get("symbol", "")),
                    side=str(payload.get("side", "buy")),
                    qty=_to_decimal(payload.get("qty"), default="0"),
                    price=_to_decimal(payload.get("price"), default="0"),
                    fee=_to_decimal(payload.get("fee"), default="0"),
                    filled_at=filled_at,
                    raw=payload,
                )
            )
        return records

    async def fetch_ohlcv_1m(
        self,
        symbol: str,
        *,
        since: datetime | None = None,
        limit: int = 500,
    ) -> list[OhlcvBar]:
        market = _infer_market(symbol)
        return await self._market_data_provider.fetch_recent_1m_bars(
            symbol=symbol,
            market=market,
            since=since,
            limit=limit,
        )

    async def fetch_latest_1m_bar(self, symbol: str) -> OhlcvBar | None:
        market = _infer_market(symbol)
        return await self._market_data_provider.fetch_latest_1m_bar(
            symbol=symbol,
            market=market,
        )

    async def fetch_latest_quote(self, symbol: str) -> QuoteSnapshot | None:
        market = _infer_market(symbol)
        return await self._market_data_provider.fetch_quote(
            symbol=symbol,
            market=market,
        )

    async def stream_market_data(self, symbols: list[str]) -> AsyncIterator[MarketDataEvent]:
        _ = symbols
        if False:  # pragma: no cover
            yield MarketDataEvent(
                channel="trade",
                symbol="",
                timestamp=datetime.now(UTC),
                payload={},
            )

    async def aclose(self) -> None:
        if self._owns_market_data_provider:
            await self._market_data_provider.aclose()
