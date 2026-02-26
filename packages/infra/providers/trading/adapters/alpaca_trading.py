"""Alpaca trading adapter implementation."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import httpx

from packages.shared_settings.schema.settings import settings
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
from packages.infra.providers.market_data.alpaca_client import AlpacaMarketDataClient


def _to_decimal(value: Any, *, default: str = "0") -> Decimal:
    if value is None:
        return Decimal(default)
    return Decimal(str(value))


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    return datetime.fromisoformat(normalized).astimezone(UTC)


class AlpacaTradingAdapter(BrokerAdapter):
    """Paper/live trading adapter for Alpaca Trading API."""

    provider = "alpaca"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_secret: str | None = None,
        trading_base_url: str | None = None,
        market_data_client: AlpacaMarketDataClient | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_key = (api_key or settings.alpaca_api_key).strip()
        self.api_secret = (api_secret or settings.alpaca_api_secret).strip()
        self.trading_base_url = (trading_base_url or settings.alpaca_trading_base_url).rstrip("/")
        self._client = client or httpx.AsyncClient(timeout=10.0)
        self._owns_client = client is None
        self._market_data_client = market_data_client or AlpacaMarketDataClient(
            api_key=self.api_key,
            api_secret=self.api_secret,
        )
        self._owns_market_data_client = market_data_client is None

    def _headers(self) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
        }

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> Any:
        try:
            response = await self._client.request(
                method,
                f"{self.trading_base_url}{path}",
                headers=self._headers(),
                params=params,
                json=json_body,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            body = ""
            if exc.response is not None:
                body = exc.response.text.strip()
            suffix = f" response={body[:500]}" if body else ""
            status = exc.response.status_code if exc.response is not None else "unknown"
            raise AdapterError(
                f"Alpaca request failed [{status}] {method} {path}.{suffix}",
            ) from exc
        except httpx.HTTPError as exc:
            raise AdapterError(f"Alpaca request failed: {exc}") from exc

        if response.status_code == 204:
            return None
        return response.json()

    async def fetch_account_state(self) -> AccountState:
        payload = await self._request("GET", "/v2/account")
        if not isinstance(payload, dict):
            raise AdapterError("Invalid account payload from Alpaca.")
        equity = _to_decimal(payload.get("equity"))
        cash = _to_decimal(payload.get("cash"))
        buying_power = _to_decimal(payload.get("buying_power"), default="0")
        margin_used = _to_decimal(payload.get("initial_margin"), default="0")
        return AccountState(
            cash=cash,
            equity=equity,
            buying_power=buying_power,
            margin_used=margin_used,
            raw=payload,
        )

    async def fetch_positions(self) -> list[PositionRecord]:
        payload = await self._request("GET", "/v2/positions")
        if not isinstance(payload, list):
            return []
        positions: list[PositionRecord] = []
        for row in payload:
            if not isinstance(row, dict):
                continue
            qty = _to_decimal(row.get("qty"), default="0")
            side = "long" if qty >= 0 else "short"
            positions.append(
                PositionRecord(
                    symbol=str(row.get("symbol", "")),
                    side=side,
                    qty=abs(qty),
                    avg_entry_price=_to_decimal(row.get("avg_entry_price"), default="0"),
                    mark_price=_to_decimal(row.get("current_price"), default="0"),
                    unrealized_pnl=_to_decimal(row.get("unrealized_pl"), default="0"),
                    realized_pnl=_to_decimal(row.get("realized_pl"), default="0"),
                    raw=row,
                )
            )
        return positions

    async def submit_order(self, intent: OrderIntent) -> OrderState:
        payload: dict[str, Any] = {
            "symbol": intent.symbol,
            "qty": str(intent.qty),
            "side": intent.side,
            "type": intent.order_type,
            "time_in_force": intent.time_in_force,
            "client_order_id": intent.client_order_id,
        }
        if intent.limit_price is not None:
            payload["limit_price"] = str(intent.limit_price)
        if intent.stop_price is not None:
            payload["stop_price"] = str(intent.stop_price)
        raw = await self._request("POST", "/v2/orders", json_body=payload)
        return self._map_order(raw)

    async def cancel_order(self, order_id: str) -> bool:
        await self._request("DELETE", f"/v2/orders/{order_id}")
        return True

    async def fetch_order(self, order_id: str) -> OrderState | None:
        try:
            raw = await self._request("GET", f"/v2/orders/{order_id}")
        except AdapterError as exc:
            if "[404]" in str(exc):
                return None
            raise
        if not isinstance(raw, dict):
            return None
        return self._map_order(raw)

    async def fetch_recent_fills(self, since: datetime | None = None) -> list[FillRecord]:
        params: dict[str, Any] = {}
        if since is not None:
            params["after"] = since.astimezone(UTC).isoformat()
        raw = await self._request("GET", "/v2/account/activities/FILL", params=params)
        if not isinstance(raw, list):
            return []

        fills: list[FillRecord] = []
        for row in raw:
            if not isinstance(row, dict):
                continue
            filled_at = _parse_timestamp(str(row.get("transaction_time")))
            if filled_at is None:
                filled_at = datetime.now(UTC)
            fills.append(
                FillRecord(
                    provider_fill_id=str(row.get("id", "")),
                    provider_order_id=str(row.get("order_id", "")),
                    symbol=str(row.get("symbol", "")),
                    side=str(row.get("side", "buy")),
                    qty=_to_decimal(row.get("qty"), default="0"),
                    price=_to_decimal(row.get("price"), default="0"),
                    fee=_to_decimal(row.get("fee"), default="0"),
                    filled_at=filled_at,
                    raw=row,
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
        return await self._market_data_client.fetch_ohlcv(
            symbol,
            since=since,
            timeframe="1Min",
            limit=limit,
        )

    async def fetch_latest_1m_bar(self, symbol: str) -> OhlcvBar | None:
        return await self._market_data_client.fetch_latest_bar(symbol)

    async def fetch_latest_quote(self, symbol: str) -> QuoteSnapshot | None:
        return await self._market_data_client.fetch_latest_quote(symbol)

    async def stream_market_data(self, symbols: list[str]) -> AsyncIterator[MarketDataEvent]:
        async for event in self._market_data_client.stream_market_data(symbols):
            yield event

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
        if self._owns_market_data_client:
            await self._market_data_client.aclose()

    def _map_order(self, raw: Any) -> OrderState:
        if not isinstance(raw, dict):
            raise AdapterError("Invalid order payload from Alpaca.")
        reject_reason_raw = raw.get("reject_reason") or raw.get("rejected_reason")
        reject_reason = str(reject_reason_raw).strip() if reject_reason_raw is not None else None
        if reject_reason == "":
            reject_reason = None
        provider_updated = (
            _parse_timestamp(str(raw.get("updated_at")))
            if raw.get("updated_at")
            else _parse_timestamp(str(raw.get("filled_at")))
            if raw.get("filled_at")
            else _parse_timestamp(str(raw.get("canceled_at")))
            if raw.get("canceled_at")
            else _parse_timestamp(str(raw.get("expired_at")))
            if raw.get("expired_at")
            else _parse_timestamp(str(raw.get("submitted_at")))
            if raw.get("submitted_at")
            else None
        )
        return OrderState(
            provider_order_id=str(raw.get("id", "")),
            client_order_id=str(raw.get("client_order_id", "")),
            symbol=str(raw.get("symbol", "")),
            side=str(raw.get("side", "buy")),
            order_type=str(raw.get("type", "market")),
            qty=_to_decimal(raw.get("qty"), default="0"),
            filled_qty=_to_decimal(raw.get("filled_qty"), default="0"),
            status=str(raw.get("status", "new")),
            submitted_at=_parse_timestamp(str(raw.get("submitted_at"))),
            avg_fill_price=(
                _to_decimal(raw.get("filled_avg_price")) if raw.get("filled_avg_price") else None
            ),
            reject_reason=reject_reason,
            provider_updated_at=provider_updated,
            raw=raw,
        )
