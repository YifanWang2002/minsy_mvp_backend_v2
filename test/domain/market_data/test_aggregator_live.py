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
    assert set(flushed.keys()) == {"5m", "15m"}
