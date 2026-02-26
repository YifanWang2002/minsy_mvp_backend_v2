"""Alpaca market-data REST client."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import httpx

from packages.shared_settings.schema.settings import settings
from packages.infra.providers.trading.adapters.base import MarketDataEvent, OhlcvBar, QuoteSnapshot


def _to_decimal(value: Any, *, default: str = "0") -> Decimal:
    if value is None:
        return Decimal(default)
    return Decimal(str(value))


def _parse_timestamp(value: str | None) -> datetime:
    if not value:
        return datetime.now(UTC)
    normalized = value.replace("Z", "+00:00")
    return datetime.fromisoformat(normalized).astimezone(UTC)


def _normalize_symbol(symbol: str) -> str:
    return symbol.strip().upper().replace("/", "")


def _internal_symbol(symbol: str) -> str:
    return symbol.strip().upper().replace("/", "").replace("-", "")


def _is_probable_crypto_symbol(symbol: str) -> bool:
    normalized = _internal_symbol(symbol)
    if not normalized:
        return False
    quotes = ("USDT", "USDC", "USD", "BTC", "ETH", "EUR")
    return any(normalized.endswith(quote) and len(normalized) > len(quote) + 1 for quote in quotes)


def _to_alpaca_symbol(symbol: str, *, market: str) -> str:
    cleaned = symbol.strip().upper().replace("-", "/")
    if market != "crypto":
        return cleaned.replace("/", "")

    if "/" in cleaned:
        base, quote = cleaned.split("/", 1)
        return f"{base}/{quote}"

    quotes = ("USDT", "USDC", "USD", "BTC", "ETH", "EUR")
    for quote in quotes:
        if cleaned.endswith(quote) and len(cleaned) > len(quote):
            base = cleaned[: -len(quote)]
            return f"{base}/{quote}"
    return cleaned


class AlpacaMarketDataClient:
    """Market-data client for Alpaca stocks + crypto endpoints."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_secret: str | None = None,
        base_url: str | None = None,
        stocks_feed: str | None = None,
        crypto_feed: str | None = None,
        request_timeout_seconds: float = 10.0,
        max_retries: int = 2,
        retry_backoff_seconds: float = 0.2,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_key = (api_key or settings.alpaca_api_key).strip()
        self.api_secret = (api_secret or settings.alpaca_api_secret).strip()
        self.base_url = (base_url or settings.alpaca_market_data_base_url).rstrip("/")
        self.stocks_feed = (stocks_feed or settings.alpaca_stocks_feed).strip() or "iex"
        self.crypto_feed = (crypto_feed or settings.alpaca_crypto_feed).strip() or "us"
        self.request_timeout_seconds = max(float(request_timeout_seconds), 0.1)
        self.max_retries = max(int(max_retries), 0)
        self.retry_backoff_seconds = max(float(retry_backoff_seconds), 0.05)
        self._client = client or httpx.AsyncClient(timeout=self.request_timeout_seconds)
        self._owns_client = client is None

    def _headers(self) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
        }

    def _infer_market(self, symbol: str) -> str:
        normalized = symbol.strip().upper()
        if "/" in normalized or "-" in normalized:
            return "crypto"
        if _is_probable_crypto_symbol(normalized):
            return "crypto"
        return "stocks"

    async def _get(self, path: str, *, params: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        for attempt in range(self.max_retries + 1):
            try:
                response = await self._client.get(
                    url,
                    headers=self._headers(),
                    params=params,
                )
                if response.status_code in {429, 500, 502, 503, 504} and attempt < self.max_retries:
                    await asyncio.sleep(self.retry_backoff_seconds * (2**attempt))
                    continue
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise ValueError("Unexpected Alpaca market-data payload.")
                return payload
            except httpx.TimeoutException:
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_backoff_seconds * (2**attempt))
                    continue
                raise
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code if exc.response is not None else None
                if status in {429, 500, 502, 503, 504} and attempt < self.max_retries:
                    await asyncio.sleep(self.retry_backoff_seconds * (2**attempt))
                    continue
                raise
            except httpx.HTTPError:
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_backoff_seconds * (2**attempt))
                    continue
                raise
        raise RuntimeError("unreachable")

    async def fetch_ohlcv(
        self,
        symbol: str,
        *,
        since: datetime | None = None,
        timeframe: str = "1Min",
        limit: int = 500,
        market: str | None = None,
    ) -> list[OhlcvBar]:
        """Fetch bars from Alpaca stocks/crypto REST endpoints."""
        market_type = market or self._infer_market(symbol)
        request_symbol = _to_alpaca_symbol(symbol, market=market_type)
        params: dict[str, Any] = {
            "symbols": request_symbol,
            "timeframe": timeframe,
            "limit": limit,
        }
        if since is not None:
            params["start"] = since.astimezone(UTC).isoformat()

        if market_type == "crypto":
            path = f"/v1beta3/crypto/{self.crypto_feed}/bars"
        else:
            path = "/v2/stocks/bars"
            params["feed"] = self.stocks_feed

        payload = await self._get(path, params=params)
        bars_root = payload.get("bars")
        if not isinstance(bars_root, dict):
            return []
        rows = bars_root.get(request_symbol)
        if not isinstance(rows, list):
            fallback_symbol = _normalize_symbol(symbol)
            rows = bars_root.get(fallback_symbol)
        if not isinstance(rows, list):
            return []

        bars: list[OhlcvBar] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            bars.append(
                OhlcvBar(
                    timestamp=_parse_timestamp(str(row.get("t"))),
                    open=_to_decimal(row.get("o")),
                    high=_to_decimal(row.get("h")),
                    low=_to_decimal(row.get("l")),
                    close=_to_decimal(row.get("c")),
                    volume=_to_decimal(row.get("v")),
                )
            )
        return bars

    async def fetch_latest_quote(
        self,
        symbol: str,
        *,
        market: str | None = None,
    ) -> QuoteSnapshot | None:
        """Fetch latest quote for one symbol."""
        market_type = market or self._infer_market(symbol)
        request_symbol = _to_alpaca_symbol(symbol, market=market_type)
        params: dict[str, Any] = {"symbols": request_symbol}
        if market_type == "crypto":
            paths = (
                f"/v1beta3/crypto/{self.crypto_feed}/latest/quotes",
                f"/v1beta3/crypto/{self.crypto_feed}/quotes/latest",
            )
        else:
            params["feed"] = self.stocks_feed
            paths = ("/v2/stocks/quotes/latest",)

        payload: dict[str, Any] | None = None
        for path in paths:
            try:
                payload = await self._get(path, params=params)
                break
            except httpx.HTTPStatusError as exc:
                if exc.response is not None and exc.response.status_code == 404:
                    continue
                raise
        if payload is None:
            return None

        quotes_root = payload.get("quotes")
        if not isinstance(quotes_root, dict):
            return None
        quote = quotes_root.get(request_symbol)
        if not isinstance(quote, dict):
            fallback_symbol = _normalize_symbol(symbol)
            quote = quotes_root.get(fallback_symbol)
        if not isinstance(quote, dict):
            return None

        bid = quote.get("bp")
        ask = quote.get("ap")
        last = quote.get("p")
        bid_decimal = _to_decimal(bid) if bid is not None else None
        ask_decimal = _to_decimal(ask) if ask is not None else None
        last_decimal = _to_decimal(last) if last is not None else None
        if last_decimal is None and bid_decimal is not None and ask_decimal is not None:
            last_decimal = (bid_decimal + ask_decimal) / Decimal("2")
        timestamp = _parse_timestamp(str(quote.get("t")))
        return QuoteSnapshot(
            symbol=_normalize_symbol(symbol),
            bid=bid_decimal,
            ask=ask_decimal,
            last=last_decimal,
            timestamp=timestamp,
            raw=quote,
        )

    async def fetch_latest_bar(
        self,
        symbol: str,
        *,
        market: str | None = None,
    ) -> OhlcvBar | None:
        """Fetch latest single bar via latest-bars endpoints."""
        market_type = market or self._infer_market(symbol)
        request_symbol = _to_alpaca_symbol(symbol, market=market_type)
        params: dict[str, Any] = {"symbols": request_symbol}

        if market_type == "crypto":
            paths = (
                f"/v1beta3/crypto/{self.crypto_feed}/bars/latest",
                f"/v1beta3/crypto/{self.crypto_feed}/latest/bars",
            )
        else:
            paths = ("/v2/stocks/bars/latest",)
            params["feed"] = self.stocks_feed

        payload: dict[str, Any] | None = None
        for path in paths:
            try:
                payload = await self._get(path, params=params)
                break
            except httpx.HTTPStatusError as exc:
                if exc.response is not None and exc.response.status_code == 404:
                    continue
                raise
        if payload is None:
            return None
        bars_root = payload.get("bars")
        if not isinstance(bars_root, dict):
            return None
        bar = bars_root.get(request_symbol)
        if not isinstance(bar, dict):
            fallback_symbol = _normalize_symbol(symbol)
            bar = bars_root.get(fallback_symbol)
        if not isinstance(bar, dict):
            return None
        return OhlcvBar(
            timestamp=_parse_timestamp(str(bar.get("t"))),
            open=_to_decimal(bar.get("o")),
            high=_to_decimal(bar.get("h")),
            low=_to_decimal(bar.get("l")),
            close=_to_decimal(bar.get("c")),
            volume=_to_decimal(bar.get("v")),
        )

    async def stream_market_data(self, _symbols: list[str]) -> AsyncIterator[MarketDataEvent]:
        """Stream market data (REST placeholder until websocket step)."""
        if False:
            yield MarketDataEvent(
                channel="noop",
                symbol="",
                timestamp=datetime.now(UTC),
                payload={},
            )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
