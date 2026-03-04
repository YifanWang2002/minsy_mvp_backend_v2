"""Helpers for timeframe-based deployment scheduling."""

from __future__ import annotations

from datetime import UTC, datetime

SUPPORTED_RUNTIME_TIMEFRAMES = (
    "1m",
    "2m",
    "5m",
    "15m",
    "30m",
    "1h",
    "2h",
    "4h",
    "1d",
)
_TIMEFRAME_SECONDS = {
    "1m": 60,
    "2m": 120,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "1d": 86400,
}


def is_supported_runtime_timeframe(timeframe: str) -> bool:
    """Return whether timeframe is supported by both DSL and paper runtime."""
    if not isinstance(timeframe, str):
        return False
    return str(timeframe).strip().lower() in _TIMEFRAME_SECONDS


def normalize_runtime_timeframe(timeframe: str, *, default_timeframe: str = "1m") -> str:
    """Return canonical runtime timeframe, falling back to a supported default."""
    if is_supported_runtime_timeframe(timeframe):
        return str(timeframe).strip().lower()
    fallback = str(default_timeframe).strip().lower()
    if fallback in _TIMEFRAME_SECONDS:
        return fallback
    return "1m"


def timeframe_to_seconds(timeframe: str, *, default_seconds: int = 60) -> int:
    """Parse one DSL-supported timeframe (for example: 1m/2m/5m/1h) into seconds."""
    if not isinstance(timeframe, str):
        return max(1, int(default_seconds))
    seconds = _TIMEFRAME_SECONDS.get(str(timeframe).strip().lower())
    if seconds is None:
        return max(1, int(default_seconds))
    return seconds


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
