"""Alpaca REST provider wrapper for market-data polling/backfill."""

from __future__ import annotations

from datetime import datetime

from packages.infra.providers.market_data.alpaca_client import AlpacaMarketDataClient
from packages.infra.providers.trading.adapters.base import OhlcvBar, QuoteSnapshot

_TIMEFRAME_TO_ALPACA: dict[str, str] = {
    "1m": "1Min",
    "5m": "5Min",
    "15m": "15Min",
    "60m": "1Hour",
    "1h": "1Hour",
    "4h": "4Hour",
    "1d": "1Day",
}


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
        until: datetime | None = None,
        limit: int = 500,
    ) -> list[OhlcvBar]:
        return await self.fetch_recent_bars(
            symbol=symbol,
            market=market,
            timeframe="1m",
            since=since,
            until=until,
            limit=limit,
        )

    async def fetch_recent_bars(
        self,
        *,
        symbol: str,
        market: str,
        timeframe: str,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 500,
    ) -> list[OhlcvBar]:
        normalized_tf = timeframe.strip().lower()
        provider_tf = _TIMEFRAME_TO_ALPACA.get(normalized_tf)
        if provider_tf is None:
            return []
        return await self._client.fetch_ohlcv(
            symbol=symbol,
            market=market,
            timeframe=provider_tf,
            since=since,
            until=until,
            limit=limit,
        )

    async def fetch_latest_1m_bar(self, *, symbol: str, market: str) -> OhlcvBar | None:
        return await self._client.fetch_latest_bar(symbol, market=market)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
