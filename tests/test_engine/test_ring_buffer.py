from __future__ import annotations

from src.engine.market_data.ring_buffer import OhlcvRing


def test_ring_buffer_append_and_latest_window() -> None:
    ring = OhlcvRing.create(5)
    for idx in range(3):
        ring.append(
            ts_ms=1_700_000_000_000 + idx * 60_000,
            open_=100 + idx,
            high=101 + idx,
            low=99 + idx,
            close=100.5 + idx,
            volume=10 + idx,
        )

    latest = ring.latest(2)
    assert latest["ts"].tolist() == [1_700_000_060_000, 1_700_000_120_000]
    assert latest["open"].tolist() == [101.0, 102.0]
    assert ring.latest_timestamp() == 1_700_000_120_000


def test_ring_buffer_overwrite_when_capacity_exceeded() -> None:
    ring = OhlcvRing.create(3)
    for idx in range(5):
        ring.append(
            ts_ms=1_700_000_000_000 + idx * 1_000,
            open_=float(idx),
            high=float(idx) + 1,
            low=float(idx) - 1,
            close=float(idx) + 0.5,
            volume=float(idx) + 10,
        )

    latest = ring.latest(10)
    assert ring.size == 3
    assert latest["ts"].tolist() == [1_700_000_002_000, 1_700_000_003_000, 1_700_000_004_000]
    assert latest["close"].tolist() == [2.5, 3.5, 4.5]


def test_ring_buffer_zero_or_negative_latest() -> None:
    ring = OhlcvRing.create(4)
    empty = ring.latest(0)
    assert empty["ts"].size == 0

    ring.append(
        ts_ms=1,
        open_=1,
        high=1,
        low=1,
        close=1,
        volume=1,
    )
    negative = ring.latest(-5)
    assert negative["ts"].size == 0
