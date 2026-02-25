"""Helpers for timeframe-based deployment scheduling."""

from __future__ import annotations

import re
from datetime import UTC, datetime

_TIMEFRAME_RE = re.compile(r"^\s*(\d+)\s*([a-zA-Z]+)\s*$")
_UNIT_SECONDS = {
    "s": 1,
    "sec": 1,
    "secs": 1,
    "second": 1,
    "seconds": 1,
    "m": 60,
    "min": 60,
    "mins": 60,
    "minute": 60,
    "minutes": 60,
    "h": 60 * 60,
    "hr": 60 * 60,
    "hrs": 60 * 60,
    "hour": 60 * 60,
    "hours": 60 * 60,
    "d": 60 * 60 * 24,
    "day": 60 * 60 * 24,
    "days": 60 * 60 * 24,
}


def timeframe_to_seconds(timeframe: str, *, default_seconds: int = 60) -> int:
    """Parse timeframe text (for example: 1m/2m/5m/1h) into seconds."""
    if not isinstance(timeframe, str):
        return max(1, int(default_seconds))
    match = _TIMEFRAME_RE.match(timeframe)
    if match is None:
        return max(1, int(default_seconds))
    value = int(match.group(1))
    unit = match.group(2).lower()
    if unit.endswith("s") and unit not in _UNIT_SECONDS:
        unit = unit[:-1]
    factor = _UNIT_SECONDS.get(unit)
    if factor is None:
        return max(1, int(default_seconds))
    return max(1, value * factor)


def compute_time_bucket(*, now: datetime, interval_seconds: int) -> int:
    """Map wall-clock time into a monotonically increasing timeframe bucket."""
    safe_interval = max(1, int(interval_seconds))
    epoch_seconds = int(now.astimezone(UTC).timestamp())
    return epoch_seconds // safe_interval


def should_trigger_cycle(
    *,
    now: datetime,
    interval_seconds: int,
    last_trigger_bucket: int | None,
) -> tuple[bool, int]:
    """Return whether a deployment should run now and the current bucket id."""
    bucket = compute_time_bucket(now=now, interval_seconds=interval_seconds)
    if last_trigger_bucket is None:
        return True, bucket
    return bucket > last_trigger_bucket, bucket
