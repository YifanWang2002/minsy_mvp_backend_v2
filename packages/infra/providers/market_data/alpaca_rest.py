"""Alpaca REST provider wrapper for market-data polling/backfill."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from decimal import Decimal

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

_ONE = Decimal("1")
_CRYPTO_ZERO_VOLUME_WICK_MAX_RATIO = Decimal("0.03")
_CRYPTO_STICKY_WICK_MAX_RATIO = Decimal("0.01")
_CRYPTO_STICKY_REPEAT_MIN_COUNT = 3


def _normalize_market(value: str) -> str:
    return value.strip().lower()


def _sticky_decimal_key(value: Decimal) -> str:
    return f"{value:.6f}"


def _sanitize_bar_bounds(bar: OhlcvBar) -> OhlcvBar:
    high = max(bar.high, bar.open, bar.low, bar.close)
    low = min(bar.low, bar.open, bar.high, bar.close)
    return OhlcvBar(
        timestamp=bar.timestamp,
        open=bar.open,
        high=high,
        low=low,
        close=bar.close,
        volume=bar.volume,
    )


def _sanitize_zero_volume_wicks(bar: OhlcvBar) -> OhlcvBar:
    if bar.volume > 0:
        return _sanitize_bar_bounds(bar)
    upper = max(bar.open, bar.close)
    lower = min(bar.open, bar.close)
    high = bar.high
    low = bar.low
    if upper > 0 and high > upper * (_ONE + _CRYPTO_ZERO_VOLUME_WICK_MAX_RATIO):
        high = upper
    if lower > 0 and low < lower * (_ONE - _CRYPTO_ZERO_VOLUME_WICK_MAX_RATIO):
        low = lower
    return _sanitize_bar_bounds(
        OhlcvBar(
            timestamp=bar.timestamp,
            open=bar.open,
            high=high,
            low=low,
            close=bar.close,
            volume=bar.volume,
        )
    )


def _sanitize_crypto_history_bars(rows: list[OhlcvBar]) -> list[OhlcvBar]:
    if not rows:
        return rows
    normalized_rows = [_sanitize_zero_volume_wicks(row) for row in rows]
    high_counts = Counter(_sticky_decimal_key(row.high) for row in normalized_rows)
    low_counts = Counter(_sticky_decimal_key(row.low) for row in normalized_rows)
    sticky_highs = {
        key for key, count in high_counts.items() if count >= _CRYPTO_STICKY_REPEAT_MIN_COUNT
    }
    sticky_lows = {
        key for key, count in low_counts.items() if count >= _CRYPTO_STICKY_REPEAT_MIN_COUNT
    }
    if not sticky_highs and not sticky_lows:
        return normalized_rows

    sanitized: list[OhlcvBar] = []
    for bar in normalized_rows:
        upper = max(bar.open, bar.close)
        lower = min(bar.open, bar.close)
        high = bar.high
        low = bar.low
        if (
            upper > 0
            and _sticky_decimal_key(high) in sticky_highs
            and high > upper * (_ONE + _CRYPTO_STICKY_WICK_MAX_RATIO)
        ):
            high = upper
        if (
            lower > 0
            and _sticky_decimal_key(low) in sticky_lows
            and low < lower * (_ONE - _CRYPTO_STICKY_WICK_MAX_RATIO)
        ):
            low = lower
        sanitized.append(
            _sanitize_bar_bounds(
                OhlcvBar(
                    timestamp=bar.timestamp,
                    open=bar.open,
                    high=high,
                    low=low,
                    close=bar.close,
                    volume=bar.volume,
                )
            )
        )
    return sanitized


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
        rows = await self._client.fetch_ohlcv(
            symbol=symbol,
            market=market,
            timeframe=provider_tf,
            since=since,
            until=until,
            limit=limit,
        )
        if _normalize_market(market) == "crypto":
            return _sanitize_crypto_history_bars(rows)
        return rows

    async def fetch_latest_1m_bar(self, *, symbol: str, market: str) -> OhlcvBar | None:
        bar = await self._client.fetch_latest_bar(symbol, market=market)
        if bar is None:
            return None
        if _normalize_market(market) == "crypto":
            return _sanitize_zero_volume_wicks(bar)
        return bar

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
