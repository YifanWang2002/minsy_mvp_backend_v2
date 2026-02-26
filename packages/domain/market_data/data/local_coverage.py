"""Detect local market-data coverage gaps for one symbol/timeframe."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd

from packages.domain.market_data.data import DataLoader


@dataclass(frozen=True, slots=True)
class MissingRange:
    """A contiguous missing range in local data."""

    start: datetime
    end: datetime
    bars: int


@dataclass(frozen=True, slots=True)
class LocalCoverageReport:
    """Coverage result for one symbol and time window."""

    market: str
    symbol: str
    timeframe: str
    start: datetime
    end: datetime
    expected_bars: int
    present_bars: int
    missing_bars: int
    local_coverage_pct: float
    missing_ranges: tuple[MissingRange, ...]


class LocalCoverageInputError(ValueError):
    """Raised when local coverage inputs are invalid."""


def detect_missing_ranges(
    *,
    loader: DataLoader,
    market: str,
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
) -> LocalCoverageReport:
    """Compare expected timestamps against local rows and return missing ranges."""

    market_key = loader.normalize_market(market)
    symbol_key = symbol.strip().upper()
    if not symbol_key:
        raise LocalCoverageInputError("symbol cannot be empty")

    timeframe_key = str(timeframe).strip().lower()
    minutes = loader.TIMEFRAME_MINUTES.get(timeframe_key)
    if minutes is None:
        raise LocalCoverageInputError(f"Unsupported timeframe: {timeframe}")

    start_utc = _align_timestamp(_ensure_utc(start), minutes=minutes)
    end_utc = _align_timestamp(_ensure_utc(end), minutes=minutes)
    if end_utc < start_utc:
        raise LocalCoverageInputError("end_date must be greater than or equal to start_date")

    expected_index = _expected_index(start=start_utc, end=end_utc, minutes=minutes)
    expected_bars = len(expected_index)
    if expected_bars == 0:
        return LocalCoverageReport(
            market=market_key,
            symbol=symbol_key,
            timeframe=timeframe_key,
            start=start_utc,
            end=end_utc,
            expected_bars=0,
            present_bars=0,
            missing_bars=0,
            local_coverage_pct=100.0,
            missing_ranges=(),
        )

    try:
        frame = loader.load(
            market=market_key,
            symbol=symbol_key,
            timeframe=timeframe_key,
            start_date=start_utc,
            end_date=end_utc,
        )
    except Exception:  # noqa: BLE001
        frame = pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"))

    actual_index = _normalize_index(frame.index)
    actual_index = actual_index.intersection(expected_index)
    missing_index = expected_index.difference(actual_index)

    missing_ranges = _compress_missing_ranges(missing_index=missing_index, minutes=minutes)
    missing_bars = len(missing_index)
    present_bars = max(0, expected_bars - missing_bars)
    coverage_pct = (present_bars / expected_bars) * 100.0 if expected_bars > 0 else 100.0

    return LocalCoverageReport(
        market=market_key,
        symbol=symbol_key,
        timeframe=timeframe_key,
        start=start_utc,
        end=end_utc,
        expected_bars=expected_bars,
        present_bars=present_bars,
        missing_bars=missing_bars,
        local_coverage_pct=round(coverage_pct, 4),
        missing_ranges=tuple(missing_ranges),
    )


def serialize_missing_ranges(ranges: list[MissingRange] | tuple[MissingRange, ...]) -> list[dict[str, Any]]:
    """Serialize ranges for JSON payloads/DB rows."""

    output: list[dict[str, Any]] = []
    for item in ranges:
        output.append(
            {
                "start": item.start.isoformat(),
                "end": item.end.isoformat(),
                "bars": int(item.bars),
            }
        )
    return output


def deserialize_missing_ranges(items: list[dict[str, Any]] | None) -> list[MissingRange]:
    """Deserialize JSON-safe missing ranges into typed objects."""

    output: list[MissingRange] = []
    for raw in items or []:
        if not isinstance(raw, dict):
            continue
        try:
            start = _ensure_utc(datetime.fromisoformat(str(raw.get("start")).replace("Z", "+00:00")))
            end = _ensure_utc(datetime.fromisoformat(str(raw.get("end")).replace("Z", "+00:00")))
            bars = max(1, int(raw.get("bars", 1)))
        except Exception:  # noqa: BLE001
            continue
        output.append(MissingRange(start=start, end=end, bars=bars))
    return output


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _align_timestamp(value: datetime, *, minutes: int) -> datetime:
    aligned = value.astimezone(UTC).replace(second=0, microsecond=0)
    if minutes >= 1440:
        return aligned.replace(hour=0, minute=0)

    minute_bucket = aligned.minute - (aligned.minute % minutes)
    return aligned.replace(minute=minute_bucket)


def _expected_index(*, start: datetime, end: datetime, minutes: int) -> pd.DatetimeIndex:
    if end < start:
        return pd.DatetimeIndex([], tz="UTC")
    if minutes >= 1440:
        return pd.date_range(start=start, end=end, freq="1D", tz="UTC")
    return pd.date_range(start=start, end=end, freq=f"{minutes}min", tz="UTC")


def _normalize_index(index: Any) -> pd.DatetimeIndex:
    if not isinstance(index, pd.DatetimeIndex):
        return pd.DatetimeIndex([], tz="UTC")

    normalized = index
    if normalized.tz is None:
        normalized = normalized.tz_localize("UTC")
    else:
        normalized = normalized.tz_convert("UTC")
    return normalized.sort_values().unique()


def _compress_missing_ranges(*, missing_index: pd.DatetimeIndex, minutes: int) -> list[MissingRange]:
    if missing_index.empty:
        return []

    step = timedelta(days=1) if minutes >= 1440 else timedelta(minutes=minutes)
    ranges: list[MissingRange] = []

    range_start = missing_index[0]
    prev = missing_index[0]
    bars = 1
    for current in missing_index[1:]:
        if (current - prev).to_pytimedelta() == step:
            bars += 1
            prev = current
            continue

        ranges.append(MissingRange(start=range_start.to_pydatetime(), end=prev.to_pydatetime(), bars=bars))
        range_start = current
        prev = current
        bars = 1

    ranges.append(MissingRange(start=range_start.to_pydatetime(), end=prev.to_pydatetime(), bars=bars))
    return ranges
