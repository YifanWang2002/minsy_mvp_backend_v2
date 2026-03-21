"""CCXT trading adapter implementation (crypto spot focused)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from packages.infra.providers.trading.adapters.base import (
    AccountState,
    AdapterError,
    BrokerAdapter,
    FillRecord,
    MarketDataEvent,
    OhlcvBar,
    OrderIntent,
    OrderState,
    PositionRecord,
    QuoteSnapshot,
)


def _to_decimal(value: Any, *, default: str = "0") -> Decimal:
    if value is None:
        return Decimal(default)
    return Decimal(str(value))


def _positive_decimal_or_none(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        parsed = Decimal(str(value))
    except Exception:  # noqa: BLE001
        return None
    return parsed if parsed > 0 else None


def _parse_timestamp_ms(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        return datetime.fromtimestamp(float(value) / 1000.0, tz=UTC)
    except Exception:  # noqa: BLE001
        return None


def _normalize_symbol(symbol: str) -> str:
    text = str(symbol).strip().upper()
    if not text:
        return "BTC/USDT"
    if "/" in text:
        base, quote = text.split("/", 1)
        if base and quote:
            return f"{base}/{quote}"
    compact = text.replace("-", "").replace("/", "")
    for quote in ("USDT", "USDC", "USD", "BTC", "ETH"):
        if compact.endswith(quote) and len(compact) > len(quote):
            base = compact[: -len(quote)]
            normalized_quote = "USDT" if quote == "USD" else quote
            return f"{base}/{normalized_quote}"
    return f"{compact}/USDT"


def _extract_okx_info_account(raw: dict[str, Any]) -> dict[str, Any] | None:
    info = raw.get("info")
    if not isinstance(info, dict):
        return None
    data = info.get("data")
    if not isinstance(data, list) or not data:
        return None
    account = data[0]
    return account if isinstance(account, dict) else None


def _extract_okx_equity(raw: dict[str, Any]) -> Decimal | None:
    account = _extract_okx_info_account(raw)
    if not isinstance(account, dict):
        return None
    for key in ("totalEq", "adjEq", "eqUsd"):
        parsed = _positive_decimal_or_none(account.get(key))
        if parsed is not None:
            return parsed
    return None


def _extract_okx_cash(raw: dict[str, Any]) -> Decimal | None:
    account = _extract_okx_info_account(raw)
    if not isinstance(account, dict):
        return None

    for key in ("availEq", "cashBal"):
        parsed = _positive_decimal_or_none(account.get(key))
        if parsed is not None:
            return parsed

    details = account.get("details")
    if not isinstance(details, list):
        return None

    stable_total = Decimal("0")
    for item in details:
        if not isinstance(item, dict):
            continue
        currency = str(item.get("ccy") or "").strip().upper()
        if currency not in {"USD", "USDT", "USDC"}:
            continue
        for key in ("availEq", "eqUsd", "availBal", "cashBal"):
            parsed = _positive_decimal_or_none(item.get(key))
            if parsed is not None:
                stable_total += parsed
                break
    return stable_total if stable_total > 0 else None


class CcxtTradingAdapter(BrokerAdapter):
    """Trading adapter over ccxt async client."""

    provider = "ccxt"

    def __init__(
        self,
        *,
        exchange_id: str,
        api_key: str,
        api_secret: str,
        password: str | None = None,
        uid: str | None = None,
        sandbox: bool = False,
        timeout_seconds: float = 10.0,
    ) -> None:
        try:
            import ccxt.async_support as ccxt  # type: ignore[import-not-found]
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("ccxt package is not installed") from exc

        resolved_exchange_id = str(exchange_id).strip().lower()
        if not resolved_exchange_id:
            raise AdapterError("CCXT exchange_id is required.")
        if not hasattr(ccxt, resolved_exchange_id):
            raise AdapterError(f"Unsupported CCXT exchange: {resolved_exchange_id}")

        exchange_cls = getattr(ccxt, resolved_exchange_id)
        params: dict[str, Any] = {
            "enableRateLimit": True,
            "timeout": int(max(float(timeout_seconds), 1.0) * 1000),
            "apiKey": str(api_key).strip(),
            "secret": str(api_secret).strip(),
        }
        if password:
            params["password"] = str(password).strip()
        if uid:
            params["uid"] = str(uid).strip()

        self._exchange = exchange_cls(params)
        self._exchange_id = resolved_exchange_id
        self._sandbox_enabled = bool(sandbox)
        self._symbol_cache: dict[str, str] = {}

        if (
            sandbox
            and self._exchange_id != "okx"
            and hasattr(self._exchange, "set_sandbox_mode")
        ):
            with_exception = False
            try:
                self._exchange.set_sandbox_mode(True)
            except Exception:  # noqa: BLE001
                with_exception = True
            if with_exception:
                raise AdapterError(
                    "Failed to enable CCXT sandbox mode for selected exchange."
                )

        if self._exchange_id == "okx":
            self._configure_okx_exchange()

    def _configure_okx_exchange(self) -> None:
        urls = getattr(self._exchange, "urls", None)
        if isinstance(urls, dict):
            api_block = urls.get("api")
            if isinstance(api_block, dict):
                api_block["rest"] = (
                    "https://us.okx.com"
                    if self._sandbox_enabled
                    else "https://www.okx.com"
                )

        if self._sandbox_enabled:
            current_headers = getattr(self._exchange, "headers", None)
            headers: dict[str, Any] = (
                dict(current_headers) if isinstance(current_headers, dict) else {}
            )
            headers["x-simulated-trading"] = "1"
            self._exchange.headers = headers

            has_map = getattr(self._exchange, "has", None)
            if isinstance(has_map, dict):
                has_map["fetchCurrencies"] = False

            async def _noop_fetch_currencies(
                params: dict[str, Any] | None = None,
            ) -> dict[str, Any]:
                _ = params
                return {}

            self._exchange.fetch_currencies = _noop_fetch_currencies

    async def _load_markets(self) -> None:
        if not self._symbol_cache:
            await self._exchange.load_markets()
            markets = getattr(self._exchange, "markets", {}) or {}
            if isinstance(markets, dict):
                self._symbol_cache = {key.upper(): key for key in markets}

    async def _resolve_symbol(self, symbol: str) -> str:
        candidate = _normalize_symbol(symbol)
        await self._load_markets()
        direct = self._symbol_cache.get(candidate.upper())
        if direct:
            return direct

        # Futures exchanges use settle-coin suffixes (e.g. BTC/USD:USD).
        # Try common derivatives formats when the normalized spot symbol misses.
        base_quote = candidate.upper()
        if ":" not in base_quote and "/" in base_quote:
            quote = base_quote.split("/", 1)[1]
            for settle in (quote, quote.replace("USDT", "USD")):
                futures_candidate = f"{base_quote.split('/')[0]}/USD:{settle}"
                hit = self._symbol_cache.get(futures_candidate.upper())
                if hit:
                    return hit

        return candidate

    async def fetch_account_state(self) -> AccountState:
        raw = await self._exchange.fetch_balance()
        free = raw.get("free") if isinstance(raw, dict) else {}
        used = raw.get("used") if isinstance(raw, dict) else {}
        total = raw.get("total") if isinstance(raw, dict) else {}
        free = free if isinstance(free, dict) else {}
        used = used if isinstance(used, dict) else {}
        total = total if isinstance(total, dict) else {}
        quote = "USDT"
        equity = _to_decimal(total.get("USD"), default="0")
        if equity <= 0:
            equity = _to_decimal(total.get("USDT"), default="0")
        if equity <= 0:
            equity = _to_decimal(free.get("USDT"), default="0") + _to_decimal(
                used.get("USDT"), default="0"
            )
        cash = _to_decimal(free.get(quote), default="0")
        if cash <= 0:
            cash = _to_decimal(free.get("USD"), default="0")
        if self._exchange_id == "okx" and isinstance(raw, dict):
            normalized_equity = _extract_okx_equity(raw)
            if normalized_equity is not None:
                equity = normalized_equity
            normalized_cash = _extract_okx_cash(raw)
            if normalized_cash is not None:
                cash = normalized_cash
            if cash > 0 and equity > 0 and cash > (equity * Decimal("5")):
                cash = equity
        buying_power = cash if cash > 0 else equity
        return AccountState(
            cash=cash,
            equity=equity if equity > 0 else buying_power,
            buying_power=buying_power,
            margin_used=Decimal("0"),
            raw=raw if isinstance(raw, dict) else {},
        )

    async def fetch_positions(self) -> list[PositionRecord]:
        raw = await self._exchange.fetch_balance()
        total = raw.get("total") if isinstance(raw, dict) else {}
        total = total if isinstance(total, dict) else {}
        positions: list[PositionRecord] = []
        for asset, amount in total.items():
            qty = _to_decimal(amount, default="0")
            if qty <= 0:
                continue
            symbol = str(asset).strip().upper()
            if symbol in {"USD", "USDT", "USDC"}:
                continue
            positions.append(
                PositionRecord(
                    symbol=symbol,
                    side="long",
                    qty=qty,
                    avg_entry_price=Decimal("0"),
                    mark_price=Decimal("0"),
                    unrealized_pnl=Decimal("0"),
                    realized_pnl=Decimal("0"),
                    raw={"asset": symbol, "qty": str(qty)},
                )
            )
        return positions

    async def submit_order(self, intent: OrderIntent) -> OrderState:
        symbol = await self._resolve_symbol(intent.symbol)
        params: dict[str, Any] = {}
        if self._exchange_id == "okx":
            # OKX spot trade endpoints require explicit trading mode even in demo.
            params.setdefault("tdMode", "cash")
            compact_client_id = (
                str(intent.client_order_id or "").strip().replace("-", "")
            )
            if compact_client_id:
                params["clOrdId"] = compact_client_id[:32]
        else:
            params["clientOrderId"] = intent.client_order_id
        price = float(intent.limit_price) if intent.limit_price is not None else None
        raw = await self._exchange.create_order(
            symbol=symbol,
            type=intent.order_type,
            side=intent.side,
            amount=float(intent.qty),
            price=price,
            params=params,
        )
        return self._map_order(raw, client_order_id=intent.client_order_id)

    async def cancel_order(self, order_id: str) -> bool:
        await self._exchange.cancel_order(order_id)
        return True

    async def fetch_order(
        self,
        order_id: str,
        *,
        symbol: str | None = None,
    ) -> OrderState | None:
        try:
            resolved_symbol = await self._resolve_symbol(symbol) if symbol else None
            if resolved_symbol:
                raw = await self._exchange.fetch_order(order_id, symbol=resolved_symbol)
            else:
                raw = await self._exchange.fetch_order(order_id)
        except Exception as exc:  # noqa: BLE001
            text = str(exc).lower()
            if "not found" in text or "does not exist" in text:
                return None
            raise AdapterError(
                f"CCXT fetch_order failed: {type(exc).__name__}"
            ) from exc
        return self._map_order(raw)

    async def fetch_recent_fills(
        self, since: datetime | None = None
    ) -> list[FillRecord]:
        since_ms = (
            int(since.astimezone(UTC).timestamp() * 1000) if since is not None else None
        )
        try:
            rows = await self._exchange.fetch_my_trades(
                symbol=None, since=since_ms, limit=200
            )
        except Exception:
            return []
        if not isinstance(rows, list):
            return []
        fills: list[FillRecord] = []
        for item in rows:
            if not isinstance(item, dict):
                continue
            filled_at = _parse_timestamp_ms(item.get("timestamp")) or datetime.now(UTC)
            fills.append(
                FillRecord(
                    provider_fill_id=str(item.get("id", "")),
                    provider_order_id=str(item.get("order", "")),
                    symbol=str(item.get("symbol", "")),
                    side=str(item.get("side", "buy")),
                    qty=_to_decimal(item.get("amount"), default="0"),
                    price=_to_decimal(item.get("price"), default="0"),
                    fee=_to_decimal((item.get("fee") or {}).get("cost"), default="0")
                    if isinstance(item.get("fee"), dict)
                    else Decimal("0"),
                    filled_at=filled_at,
                    raw=item,
                )
            )
        return fills

    async def fetch_ohlcv_1m(
        self,
        symbol: str,
        *,
        since: datetime | None = None,
        limit: int = 500,
    ) -> list[OhlcvBar]:
        ccxt_symbol = await self._resolve_symbol(symbol)
        since_ms = (
            int(since.astimezone(UTC).timestamp() * 1000) if since is not None else None
        )
        rows = await self._exchange.fetch_ohlcv(
            symbol=ccxt_symbol,
            timeframe="1m",
            since=since_ms,
            limit=max(1, int(limit)),
        )
        return self._map_ohlcv_rows(rows)

    async def fetch_latest_1m_bar(self, symbol: str) -> OhlcvBar | None:
        bars = await self.fetch_ohlcv_1m(symbol, limit=1)
        if not bars:
            return None
        return bars[-1]

    async def fetch_latest_quote(self, symbol: str) -> QuoteSnapshot | None:
        ccxt_symbol = await self._resolve_symbol(symbol)
        try:
            ticker = await self._exchange.fetch_ticker(ccxt_symbol)
        except Exception as exc:  # noqa: BLE001
            raise AdapterError(
                f"CCXT fetch_ticker failed: {type(exc).__name__}"
            ) from exc
        if not isinstance(ticker, dict):
            return None
        timestamp = _parse_timestamp_ms(ticker.get("timestamp")) or datetime.now(UTC)
        return QuoteSnapshot(
            symbol=ccxt_symbol,
            bid=_to_decimal(ticker.get("bid"), default="0"),
            ask=_to_decimal(ticker.get("ask"), default="0"),
            last=_to_decimal(ticker.get("last"), default="0"),
            timestamp=timestamp,
            raw=ticker,
        )

    async def stream_market_data(
        self, symbols: list[str]
    ) -> AsyncIterator[MarketDataEvent]:
        _ = symbols
        if False:  # pragma: no cover
            yield MarketDataEvent(
                channel="trade",
                symbol="",
                timestamp=datetime.now(UTC),
                payload={},
            )

    async def aclose(self) -> None:
        await self._exchange.close()

    def _map_order(self, raw: Any, *, client_order_id: str | None = None) -> OrderState:
        if not isinstance(raw, dict):
            raise AdapterError("Invalid CCXT order payload.")
        submitted_at = _parse_timestamp_ms(raw.get("timestamp"))
        provider_updated_at = _parse_timestamp_ms(raw.get("lastTradeTimestamp"))
        order_status = str(raw.get("status", "open")).strip().lower() or "open"
        return OrderState(
            provider_order_id=str(raw.get("id", "")),
            client_order_id=str(
                raw.get("clientOrderId") or raw.get("clOrdId") or client_order_id or ""
            ),
            symbol=str(raw.get("symbol", "")),
            side=str(raw.get("side", "buy")),
            order_type=str(raw.get("type", "market")),
            qty=_to_decimal(raw.get("amount"), default="0"),
            filled_qty=_to_decimal(raw.get("filled"), default="0"),
            status=order_status,
            submitted_at=submitted_at,
            avg_fill_price=_to_decimal(raw.get("average"), default="0"),
            reject_reason=None,
            provider_updated_at=provider_updated_at or submitted_at,
            raw=raw,
        )

    def _map_ohlcv_rows(self, rows: Any) -> list[OhlcvBar]:
        if not isinstance(rows, list):
            return []
        bars: list[OhlcvBar] = []
        for row in rows:
            if not isinstance(row, list) or len(row) < 6:
                continue
            timestamp = _parse_timestamp_ms(row[0])
            if timestamp is None:
                continue
            bars.append(
                OhlcvBar(
                    timestamp=timestamp,
                    open=_to_decimal(row[1], default="0"),
                    high=_to_decimal(row[2], default="0"),
                    low=_to_decimal(row[3], default="0"),
                    close=_to_decimal(row[4], default="0"),
                    volume=_to_decimal(row[5], default="0"),
                )
            )
        return bars
