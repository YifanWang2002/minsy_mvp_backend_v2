from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from packages.infra.providers.market_data.alpaca_rest import AlpacaRestProvider
from packages.infra.providers.trading.adapters.base import OhlcvBar


class _StubAlpacaClient:
    def __init__(
        self,
        *,
        bars: list[OhlcvBar] | None = None,
        latest_bar: OhlcvBar | None = None,
    ) -> None:
        self._bars = list(bars or [])
        self._latest_bar = latest_bar

    async def fetch_ohlcv(
        self,
        symbol: str,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        timeframe: str = "1Min",
        limit: int = 500,
        market: str | None = None,
    ) -> list[OhlcvBar]:
        _ = (symbol, since, until, timeframe, limit, market)
        return list(self._bars)

    async def fetch_latest_bar(
        self,
        symbol: str,
        *,
        market: str | None = None,
    ) -> OhlcvBar | None:
        _ = (symbol, market)
        return self._latest_bar

    async def fetch_latest_quote(self, symbol: str, *, market: str | None = None):
        _ = (symbol, market)
        return None

    async def aclose(self) -> None:
        return None


def _bar(
    ts: datetime,
    *,
    open_: str,
    high: str,
    low: str,
    close: str,
    volume: str,
) -> OhlcvBar:
    return OhlcvBar(
        timestamp=ts,
        open=Decimal(open_),
        high=Decimal(high),
        low=Decimal(low),
        close=Decimal(close),
        volume=Decimal(volume),
    )


@pytest.mark.asyncio
async def test_fetch_recent_crypto_bars_clamps_sticky_repeated_high_outliers() -> None:
    start = datetime(2026, 3, 9, 0, 0, tzinfo=UTC)
    sticky_high = "68802.662075"
    bars = [
        _bar(
            start + timedelta(minutes=index),
            open_=str(Decimal("66000") + index),
            high=sticky_high,
            low=str(Decimal("65990") + index),
            close=str(Decimal("66020") + index),
            volume="0",
        )
        for index in range(3)
    ]
    bars.append(
        _bar(
            start + timedelta(minutes=3),
            open_="66100",
            high="66200",
            low="66090",
            close="66180",
            volume="0.5",
        )
    )
    provider = AlpacaRestProvider(client=_StubAlpacaClient(bars=bars))

    rows = await provider.fetch_recent_bars(
        symbol="BTCUSD",
        market="crypto",
        timeframe="1m",
        limit=10,
    )

    assert len(rows) == 4
    for index in range(3):
        row = rows[index]
        assert row.high == max(row.open, row.close)
    assert rows[3].high == Decimal("66200")


@pytest.mark.asyncio
async def test_fetch_latest_crypto_bar_clamps_zero_volume_extreme_wicks() -> None:
    latest = _bar(
        datetime(2026, 3, 9, 1, 0, tzinfo=UTC),
        open_="66000",
        high="68850",
        low="64000",
        close="66010",
        volume="0",
    )
    provider = AlpacaRestProvider(client=_StubAlpacaClient(latest_bar=latest))

    result = await provider.fetch_latest_1m_bar(symbol="BTCUSD", market="crypto")

    assert result is not None
    assert result.high == Decimal("66010")
    assert result.low == Decimal("66000")
    assert result.open == Decimal("66000")
    assert result.close == Decimal("66010")


@pytest.mark.asyncio
async def test_fetch_recent_bars_keeps_stocks_data_unchanged() -> None:
    bars = [
        _bar(
            datetime(2026, 3, 9, 2, 0, tzinfo=UTC),
            open_="100",
            high="120",
            low="99",
            close="101",
            volume="10",
        ),
        _bar(
            datetime(2026, 3, 9, 2, 1, tzinfo=UTC),
            open_="101",
            high="120",
            low="100",
            close="102",
            volume="11",
        ),
        _bar(
            datetime(2026, 3, 9, 2, 2, tzinfo=UTC),
            open_="102",
            high="120",
            low="101",
            close="103",
            volume="12",
        ),
    ]
    provider = AlpacaRestProvider(client=_StubAlpacaClient(bars=bars))

    rows = await provider.fetch_recent_bars(
        symbol="AAPL",
        market="stocks",
        timeframe="1m",
        limit=10,
    )

    assert [row.high for row in rows] == [Decimal("120"), Decimal("120"), Decimal("120")]
