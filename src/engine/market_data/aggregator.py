"""1m to higher-timeframe bar aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from zoneinfo import ZoneInfo


def _parse_timeframe(timeframe: str) -> tuple[str, int]:
    normalized = timeframe.strip().lower()
    if normalized.endswith("m"):
        return "minute", int(normalized[:-1] or "0")
    if normalized.endswith("h"):
        return "hour", int(normalized[:-1] or "0")
    if normalized.endswith("d"):
        return "day", int(normalized[:-1] or "0")
    raise ValueError(f"Unsupported timeframe: {timeframe}")


def _bucket_start_ts_ms(ts_ms: int, timeframe: str, tz: ZoneInfo) -> int:
    unit, step = _parse_timeframe(timeframe)
    if step <= 0:
        raise ValueError(f"Invalid timeframe step: {timeframe}")

    utc_dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=UTC)
    local_dt = utc_dt.astimezone(tz)

    if unit == "minute":
        floored_minute = (local_dt.minute // step) * step
        bucket_local = local_dt.replace(second=0, microsecond=0, minute=floored_minute)
    elif unit == "hour":
        floored_hour = (local_dt.hour // step) * step
        bucket_local = local_dt.replace(
            second=0,
            microsecond=0,
            minute=0,
            hour=floored_hour,
        )
    else:
        day_start = local_dt.replace(second=0, microsecond=0, minute=0, hour=0)
        ordinal = day_start.toordinal()
        floored_ordinal = ordinal - ((ordinal - 1) % step)
        bucket_local = datetime.fromordinal(floored_ordinal).replace(tzinfo=tz)

    return int(bucket_local.astimezone(UTC).timestamp() * 1000)


@dataclass(frozen=True, slots=True)
class AggregatedBar:
    """Normalized aggregated OHLCV bar."""

    ts_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(slots=True)
class _BucketState:
    ts_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float

    def to_bar(self) -> AggregatedBar:
        return AggregatedBar(
            ts_ms=self.ts_ms,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
        )


class BarAggregator:
    """Aggregates base bars into multiple configured timeframes."""

    def __init__(
        self,
        *,
        timeframes: tuple[str, ...] = ("5m", "15m", "1h", "1d"),
        timezone: str = "UTC",
    ) -> None:
        self.timeframes = tuple(timeframes)
        self.timezone = ZoneInfo(timezone)
        self._states: dict[tuple[str, str, str], _BucketState] = {}

    def ingest_1m_bar(
        self,
        *,
        market: str,
        symbol: str,
        ts_ms: int,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: float,
    ) -> dict[str, AggregatedBar]:
        """Ingest one 1m bar and return closed buckets keyed by timeframe."""
        closed: dict[str, AggregatedBar] = {}
        for timeframe in self.timeframes:
            key = (market, symbol, timeframe)
            bucket_start = _bucket_start_ts_ms(ts_ms, timeframe, self.timezone)
            state = self._states.get(key)

            if state is None:
                self._states[key] = _BucketState(
                    ts_ms=bucket_start,
                    open=float(open_),
                    high=float(high),
                    low=float(low),
                    close=float(close),
                    volume=float(volume),
                )
                continue

            if state.ts_ms != bucket_start:
                closed[timeframe] = state.to_bar()
                self._states[key] = _BucketState(
                    ts_ms=bucket_start,
                    open=float(open_),
                    high=float(high),
                    low=float(low),
                    close=float(close),
                    volume=float(volume),
                )
                continue

            state.high = max(state.high, float(high))
            state.low = min(state.low, float(low))
            state.close = float(close)
            state.volume += float(volume)
        return closed

    def checkpoint(self) -> dict[str, int]:
        """Return lightweight bucket checkpoint keyed by serialized key."""
        result: dict[str, int] = {}
        for (market, symbol, timeframe), state in self._states.items():
            result[f"{market}:{symbol}:{timeframe}"] = state.ts_ms
        return result

    def restore_checkpoint(self, checkpoint: dict[str, int]) -> None:
        """Restore bucket starts only (OHLCV values are cold-started on next bar)."""
        for key, ts_ms in checkpoint.items():
            parts = key.split(":")
            if len(parts) != 3:
                continue
            market, symbol, timeframe = parts
            self._states[(market, symbol, timeframe)] = _BucketState(
                ts_ms=int(ts_ms),
                open=0.0,
                high=0.0,
                low=0.0,
                close=0.0,
                volume=0.0,
            )

    def flush(
        self,
        *,
        market: str,
        symbol: str,
    ) -> dict[str, AggregatedBar]:
        """Force-flush open buckets for one symbol."""
        closed: dict[str, AggregatedBar] = {}
        for timeframe in self.timeframes:
            key = (market, symbol, timeframe)
            state = self._states.pop(key, None)
            if state is None:
                continue
            closed[timeframe] = state.to_bar()
        return closed
