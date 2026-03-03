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
class _ComponentBar:
    ts_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(slots=True)
class _BucketState:
    ts_ms: int
    components: dict[int, _ComponentBar]

    def has_components(self) -> bool:
        return bool(self.components)

    def upsert_component(
        self,
        *,
        ts_ms: int,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: float,
    ) -> None:
        self.components[int(ts_ms)] = _ComponentBar(
            ts_ms=int(ts_ms),
            open=float(open_),
            high=float(high),
            low=float(low),
            close=float(close),
            volume=float(volume),
        )

    def to_bar(self) -> AggregatedBar:
        ordered = [self.components[key] for key in sorted(self.components)]
        if not ordered:
            return AggregatedBar(
                ts_ms=self.ts_ms,
                open=0.0,
                high=0.0,
                low=0.0,
                close=0.0,
                volume=0.0,
            )
        return AggregatedBar(
            ts_ms=self.ts_ms,
            open=ordered[0].open,
            high=max(item.high for item in ordered),
            low=min(item.low for item in ordered),
            close=ordered[-1].close,
            volume=sum(item.volume for item in ordered),
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
                    components={},
                )
                self._states[key].upsert_component(
                    ts_ms=ts_ms,
                    open_=open_,
                    high=high,
                    low=low,
                    close=close,
                    volume=volume,
                )
                continue

            if state.ts_ms != bucket_start:
                if state.has_components():
                    closed[timeframe] = state.to_bar()
                self._states[key] = _BucketState(
                    ts_ms=bucket_start,
                    components={},
                )
            self._states[key].upsert_component(
                ts_ms=ts_ms,
                open_=open_,
                high=high,
                low=low,
                close=close,
                volume=volume,
            )
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
                components={},
            )

    def clear_symbol(self, *, market: str, symbol: str) -> None:
        stale_keys = [
            key
            for key in self._states
            if key[0] == market and key[1] == symbol
        ]
        for key in stale_keys:
            self._states.pop(key, None)

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
            if state is None or not state.has_components():
                continue
            closed[timeframe] = state.to_bar()
        return closed
