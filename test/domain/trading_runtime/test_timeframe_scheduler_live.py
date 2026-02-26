from __future__ import annotations

from datetime import UTC, datetime, timedelta

from packages.domain.trading.runtime.timeframe_scheduler import (
    compute_time_bucket,
    should_trigger_cycle,
    timeframe_to_seconds,
)


def test_000_accessibility_timeframe_to_seconds() -> None:
    assert timeframe_to_seconds("1m") == 60
    assert timeframe_to_seconds("15m") == 900
    assert timeframe_to_seconds("2h") == 7200


def test_010_timeframe_to_seconds_fallback_for_invalid_input() -> None:
    assert timeframe_to_seconds("bad", default_seconds=75) == 75
    assert timeframe_to_seconds("0x", default_seconds=42) == 42


def test_020_should_trigger_cycle_bucket_progression() -> None:
    now = datetime(2026, 1, 1, 12, 0, 10, tzinfo=UTC)
    interval = 60
    bucket = compute_time_bucket(now=now, interval_seconds=interval)

    should_first, first_bucket = should_trigger_cycle(
        now=now,
        interval_seconds=interval,
        last_trigger_bucket=None,
    )
    assert should_first is True
    assert first_bucket == bucket

    should_same, same_bucket = should_trigger_cycle(
        now=now + timedelta(seconds=20),
        interval_seconds=interval,
        last_trigger_bucket=first_bucket,
    )
    assert should_same is False
    assert same_bucket == first_bucket

    should_next, next_bucket = should_trigger_cycle(
        now=now + timedelta(seconds=70),
        interval_seconds=interval,
        last_trigger_bucket=first_bucket,
    )
    assert should_next is True
    assert next_bucket > first_bucket
