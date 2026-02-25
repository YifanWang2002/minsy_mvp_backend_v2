from __future__ import annotations

from datetime import UTC, datetime

from src.engine.execution.timeframe_scheduler import (
    compute_time_bucket,
    should_trigger_cycle,
    timeframe_to_seconds,
)


def test_timeframe_to_seconds_parses_supported_units() -> None:
    assert timeframe_to_seconds("1m") == 60
    assert timeframe_to_seconds("2m") == 120
    assert timeframe_to_seconds("5m") == 300
    assert timeframe_to_seconds("5min") == 300
    assert timeframe_to_seconds("2 minutes") == 120
    assert timeframe_to_seconds("1h") == 3600
    assert timeframe_to_seconds("1hour") == 3600
    assert timeframe_to_seconds("1d") == 86400
    assert timeframe_to_seconds("2days") == 172800
    assert timeframe_to_seconds("30s") == 30


def test_timeframe_to_seconds_falls_back_for_invalid_values() -> None:
    assert timeframe_to_seconds("abc", default_seconds=45) == 45
    assert timeframe_to_seconds("", default_seconds=45) == 45


def test_should_trigger_cycle_advances_by_bucket() -> None:
    now = datetime(2026, 2, 21, 12, 34, 56, tzinfo=UTC)
    bucket = compute_time_bucket(now=now, interval_seconds=60)

    due_first, first_bucket = should_trigger_cycle(
        now=now,
        interval_seconds=60,
        last_trigger_bucket=None,
    )
    assert due_first is True
    assert first_bucket == bucket

    due_same_bucket, _ = should_trigger_cycle(
        now=now,
        interval_seconds=60,
        last_trigger_bucket=bucket,
    )
    assert due_same_bucket is False

    next_minute = datetime(2026, 2, 21, 12, 35, 1, tzinfo=UTC)
    due_next_bucket, next_bucket = should_trigger_cycle(
        now=next_minute,
        interval_seconds=60,
        last_trigger_bucket=bucket,
    )
    assert due_next_bucket is True
    assert next_bucket == bucket + 1
