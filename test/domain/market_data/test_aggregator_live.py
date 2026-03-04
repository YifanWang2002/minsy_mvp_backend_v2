from __future__ import annotations

from packages.domain.market_data.aggregator import BarAggregator


def test_000_accessibility_aggregator_closes_bucket_on_boundary() -> None:
    agg = BarAggregator(timeframes=("5m",), timezone="UTC")

    first = agg.ingest_1m_bar(
        market="stocks",
        symbol="SPY",
        ts_ms=1704067200000,
        open_=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        volume=10.0,
    )
    assert first == {}

    second = agg.ingest_1m_bar(
        market="stocks",
        symbol="SPY",
        ts_ms=1704067500000,
        open_=100.5,
        high=102.0,
        low=100.0,
        close=101.5,
        volume=20.0,
    )
    assert "5m" in second
    closed = second["5m"]
    assert closed.open == 100.0
    assert closed.close == 100.5


def test_010_aggregator_checkpoint_and_flush() -> None:
    agg = BarAggregator(timeframes=("5m", "15m"), timezone="UTC")
    _ = agg.ingest_1m_bar(
        market="stocks",
        symbol="SPY",
        ts_ms=1704067200000,
        open_=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        volume=10.0,
    )

    checkpoint = agg.checkpoint()
    assert checkpoint

    restored = BarAggregator(timeframes=("5m", "15m"), timezone="UTC")
    restored.restore_checkpoint(checkpoint)
    flushed = restored.flush(market="stocks", symbol="SPY")
    assert flushed == {}


def test_020_aggregator_duplicate_latest_minute_replaces_instead_of_double_counting() -> None:
    agg = BarAggregator(timeframes=("5m",), timezone="UTC")

    _ = agg.ingest_1m_bar(
        market="crypto",
        symbol="BTCUSD",
        ts_ms=1704067200000,
        open_=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        volume=10.0,
    )
    _ = agg.ingest_1m_bar(
        market="crypto",
        symbol="BTCUSD",
        ts_ms=1704067260000,
        open_=100.5,
        high=102.0,
        low=100.0,
        close=101.0,
        volume=20.0,
    )
    _ = agg.ingest_1m_bar(
        market="crypto",
        symbol="BTCUSD",
        ts_ms=1704067260000,
        open_=100.5,
        high=103.0,
        low=98.5,
        close=102.5,
        volume=25.0,
    )

    closed = agg.ingest_1m_bar(
        market="crypto",
        symbol="BTCUSD",
        ts_ms=1704067500000,
        open_=102.5,
        high=104.0,
        low=102.0,
        close=103.0,
        volume=15.0,
    )

    assert "5m" in closed
    bar = closed["5m"]
    assert bar.open == 100.0
    assert bar.high == 103.0
    assert bar.low == 98.5
    assert bar.close == 102.5
    assert bar.volume == 35.0
