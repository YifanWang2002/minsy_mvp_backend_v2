from __future__ import annotations

from datetime import UTC, datetime

from src.engine.market_data.aggregator import BarAggregator


def _ms(value: str) -> int:
    return int(datetime.fromisoformat(value).astimezone(UTC).timestamp() * 1000)


def test_aggregator_closes_5m_bucket_on_boundary() -> None:
    aggregator = BarAggregator(timeframes=("5m",), timezone="UTC")

    for minute in range(5):
        ts = _ms(f"2026-01-05T10:0{minute}:00+00:00")
        closed = aggregator.ingest_1m_bar(
            market="stocks",
            symbol="AAPL",
            ts_ms=ts,
            open_=100 + minute,
            high=101 + minute,
            low=99 + minute,
            close=100.5 + minute,
            volume=10,
        )
        assert closed == {}

    boundary_ts = _ms("2026-01-05T10:05:00+00:00")
    closed = aggregator.ingest_1m_bar(
        market="stocks",
        symbol="AAPL",
        ts_ms=boundary_ts,
        open_=200,
        high=201,
        low=199,
        close=200.5,
        volume=20,
    )

    assert "5m" in closed
    bar = closed["5m"]
    assert bar.ts_ms == _ms("2026-01-05T10:00:00+00:00")
    assert bar.open == 100.0
    assert bar.high == 105.0
    assert bar.low == 99.0
    assert bar.close == 104.5
    assert bar.volume == 50.0


def test_aggregator_closes_hour_bucket_on_hour_switch() -> None:
    aggregator = BarAggregator(timeframes=("1h",), timezone="UTC")

    aggregator.ingest_1m_bar(
        market="stocks",
        symbol="AAPL",
        ts_ms=_ms("2026-01-05T12:59:00+00:00"),
        open_=100,
        high=101,
        low=99,
        close=100.5,
        volume=10,
    )
    closed = aggregator.ingest_1m_bar(
        market="stocks",
        symbol="AAPL",
        ts_ms=_ms("2026-01-05T13:00:00+00:00"),
        open_=102,
        high=103,
        low=101,
        close=102.5,
        volume=10,
    )
    assert "1h" in closed
    assert closed["1h"].ts_ms == _ms("2026-01-05T12:00:00+00:00")


def test_aggregator_respects_timezone_for_daily_boundary() -> None:
    aggregator = BarAggregator(timeframes=("1d",), timezone="America/New_York")

    aggregator.ingest_1m_bar(
        market="stocks",
        symbol="AAPL",
        ts_ms=_ms("2026-01-05T04:59:00+00:00"),  # 23:59 previous NY day
        open_=100,
        high=101,
        low=99,
        close=100.5,
        volume=10,
    )
    closed = aggregator.ingest_1m_bar(
        market="stocks",
        symbol="AAPL",
        ts_ms=_ms("2026-01-05T05:00:00+00:00"),  # 00:00 next NY day
        open_=102,
        high=103,
        low=101,
        close=102.5,
        volume=10,
    )

    assert "1d" in closed
    assert closed["1d"].ts_ms == _ms("2026-01-04T05:00:00+00:00")
