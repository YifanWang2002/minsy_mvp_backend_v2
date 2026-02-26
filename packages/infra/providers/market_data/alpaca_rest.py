"""Alpaca REST provider wrapper for market-data polling/backfill."""

from __future__ import annotations

from datetime import datetime

from packages.infra.providers.market_data.alpaca_client import AlpacaMarketDataClient
from packages.infra.providers.trading.adapters.base import OhlcvBar, QuoteSnapshot


class AlpacaRestProvider:
    """REST wrapper around Alpaca market-data client."""

    def __init__(self, client: AlpacaMarketDataClient | None = None) -> None:
        self._client = client or AlpacaMarketDataClient()
        self._owns_client = client is None

    async def fetch_quote(self, *, symbol: str, market: str) -> QuoteSnapshot | None:
        quote = await self._client.fetch_latest_quote(symbol, market=market)
        if quote is not None:
            return quote
        # Quote endpoints occasionally return 404/empty for crypto symbols;
        # fall back to latest 1m close so downstream mark-price logic keeps working.
        latest_bar = await self._client.fetch_latest_bar(symbol, market=market)
        if latest_bar is None:
            return None
        return QuoteSnapshot(
            symbol=symbol.strip().upper(),
            bid=None,
            ask=None,
            last=latest_bar.close,
            timestamp=latest_bar.timestamp,
            raw={"source": "latest_bar_fallback"},
        )

    async def fetch_recent_1m_bars(
        self,
        *,
        symbol: str,
        market: str,
        since: datetime | None = None,
        limit: int = 500,
    ) -> list[OhlcvBar]:
        return await self._client.fetch_ohlcv(
            symbol=symbol,
            market=market,
            timeframe="1Min",
            since=since,
            limit=limit,
        )

    async def fetch_latest_1m_bar(self, *, symbol: str, market: str) -> OhlcvBar | None:
        return await self._client.fetch_latest_bar(symbol, market=market)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
