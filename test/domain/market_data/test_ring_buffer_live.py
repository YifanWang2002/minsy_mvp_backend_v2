from __future__ import annotations

from packages.domain.market_data.ring_buffer import OhlcvRing


def test_000_accessibility_ring_buffer_append_and_latest() -> None:
    ring = OhlcvRing.create(3)
    ring.append(ts_ms=1, open_=1.0, high=2.0, low=0.5, close=1.5, volume=10.0)
    ring.append(ts_ms=2, open_=1.5, high=2.5, low=1.0, close=2.0, volume=20.0)

    latest = ring.latest(2)
    assert latest["ts"].tolist() == [1, 2]
    assert latest["close"].tolist() == [1.5, 2.0]
    assert ring.latest_timestamp() == 2


def test_010_ring_buffer_overwrite_when_capacity_exceeded() -> None:
    ring = OhlcvRing.create(2)
    ring.append(ts_ms=1, open_=1.0, high=1.0, low=1.0, close=1.0, volume=1.0)
    ring.append(ts_ms=2, open_=2.0, high=2.0, low=2.0, close=2.0, volume=2.0)
    ring.append(ts_ms=3, open_=3.0, high=3.0, low=3.0, close=3.0, volume=3.0)

    latest = ring.latest(2)
    assert latest["ts"].tolist() == [2, 3]
    assert latest["open"].tolist() == [2.0, 3.0]


def test_020_ring_buffer_non_positive_latest_returns_empty() -> None:
    ring = OhlcvRing.create(2)
    latest = ring.latest(0)
    assert latest["ts"].tolist() == []


def test_030_ring_buffer_append_or_replace_is_idempotent_for_latest_bar() -> None:
    ring = OhlcvRing.create(3)
    ring.append(ts_ms=1, open_=1.0, high=2.0, low=0.5, close=1.5, volume=10.0)

    replaced = ring.append_or_replace(
        ts_ms=1,
        open_=1.0,
        high=3.0,
        low=0.25,
        close=2.5,
        volume=12.0,
    )
    ignored = ring.append_or_replace(
        ts_ms=0,
        open_=9.0,
        high=9.0,
        low=9.0,
        close=9.0,
        volume=9.0,
    )

    latest = ring.latest(3)
    assert replaced == "replaced"
    assert ignored == "ignored"
    assert latest["ts"].tolist() == [1]
    assert latest["high"].tolist() == [3.0]
    assert latest["low"].tolist() == [0.25]
    assert latest["close"].tolist() == [2.5]
