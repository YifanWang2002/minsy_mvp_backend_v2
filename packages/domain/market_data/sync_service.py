"""Missing-range market-data sync job orchestration service."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from packages.domain.market_data.catalog_service import (
    upsert_catalog_entry_from_parquet,
)
from packages.domain.market_data.data import DataLoader
from packages.domain.market_data.data.local_coverage import (
    LocalCoverageInputError,
    MissingRange,
    deserialize_missing_ranges,
    detect_missing_ranges,
    serialize_missing_ranges,
)
from packages.domain.market_data.data.parquet_writer import append_ohlcv_rows
from packages.domain.ports.queue_ports import get_job_queue_ports
from packages.infra.db import session as db_module
from packages.infra.db.models.market_data_sync_chunk import MarketDataSyncChunk
from packages.infra.db.models.market_data_sync_job import MarketDataSyncJob
from packages.infra.observability.logger import logger
from packages.infra.providers.market_data.alpaca_client import AlpacaMarketDataClient
from packages.infra.providers.market_data.ccxt_rest import CcxtRestProvider
from packages.infra.providers.trading.adapters.base import OhlcvBar
from packages.infra.redis.client import get_redis_client, init_redis
from packages.shared_settings.schema.settings import settings

_INTERNAL_TO_EXTERNAL_STATUS: dict[str, str] = {
    "queued": "pending",
    "running": "running",
    "completed": "done",
    "failed": "failed",
    "cancelled": "failed",
}

_SUPPORTED_PROVIDERS: frozenset[str] = frozenset({"alpaca", "ccxt"})
_RANGE_SPLIT_SPAN_BY_TIMEFRAME: dict[str, timedelta] = {
    "1m": timedelta(days=90),
    "5m": timedelta(days=180),
}
_FETCH_CONCURRENCY_BY_TIMEFRAME: dict[str, int] = {
    "1m": 12,
    "5m": 8,
}


@dataclass(frozen=True, slots=True)
class MarketDataSyncJobReceipt:
    """Create-job response payload."""

    job_id: UUID
    status: str
    progress: int
    deduplicated: bool = False


@dataclass(frozen=True, slots=True)
class MarketDataSyncJobView:
    """Read model for MCP/API payloads."""

    job_id: UUID
    user_id: UUID | None
    provider: str
    market: str
    symbol: str
    timeframe: str
    requested_start: datetime
    requested_end: datetime
    missing_ranges: tuple[MissingRange, ...]
    status: str
    progress: int
    current_step: str | None
    rows_written: int
    range_filled: int
    total_ranges: int
    errors: tuple[dict[str, Any], ...]
    submitted_at: datetime
    completed_at: datetime | None


class _RequestStartLimiter:
    """Spread request start times so high concurrency does not burst the provider."""

    def __init__(self, *, min_interval_seconds: float) -> None:
        self._min_interval_seconds = max(float(min_interval_seconds), 0.0)
        self._lock = asyncio.Lock()
        self._next_ready_at = 0.0

    async def wait_turn(self) -> None:
        if self._min_interval_seconds <= 0:
            return
        loop = asyncio.get_running_loop()
        async with self._lock:
            now = loop.time()
            if now < self._next_ready_at:
                await asyncio.sleep(self._next_ready_at - now)
                now = loop.time()
            self._next_ready_at = now + self._min_interval_seconds


class MarketDataSyncJobNotFoundError(LookupError):
    """Raised when a sync job does not exist."""


class MarketDataSyncInputError(ValueError):
    """Raised for invalid sync job input."""


class MarketDataProviderUnavailableError(RuntimeError):
    """Raised when provider is not configured or cannot be initialized."""


class MarketDataNoMissingDataError(ValueError):
    """Raised when no missing ranges are detected."""


async def create_market_data_sync_job(
    db: AsyncSession,
    *,
    provider: str,
    market: str,
    symbol: str,
    timeframe: str,
    requested_start: datetime,
    requested_end: datetime,
    missing_ranges: list[dict[str, Any]] | None = None,
    user_id: UUID | None = None,
    auto_commit: bool = True,
) -> MarketDataSyncJobReceipt:
    """Create one queued market-data sync job."""

    provider_key = _normalize_provider(provider)
    loader = DataLoader()

    market_key = loader.normalize_market(market)
    symbol_key = _normalize_symbol(symbol)
    timeframe_key = _normalize_timeframe(timeframe)
    start_utc = _ensure_utc(requested_start)
    end_utc = _ensure_utc(requested_end)
    if end_utc < start_utc:
        raise MarketDataSyncInputError("requested_end must be greater than requested_start")

    existing_job = await db.scalar(
        select(MarketDataSyncJob)
        .where(
            MarketDataSyncJob.provider == provider_key,
            MarketDataSyncJob.market == market_key,
            MarketDataSyncJob.symbol == symbol_key,
            MarketDataSyncJob.timeframe == timeframe_key,
            MarketDataSyncJob.status.in_(("queued", "running")),
        )
        .order_by(MarketDataSyncJob.submitted_at.desc())
    )
    if existing_job is not None:
        return MarketDataSyncJobReceipt(
            job_id=existing_job.id,
            status=_to_external_status(existing_job.status),
            progress=int(existing_job.progress),
            deduplicated=True,
        )

    ranges = _resolve_missing_ranges(
        loader=loader,
        market=market_key,
        symbol=symbol_key,
        timeframe=timeframe_key,
        requested_start=start_utc,
        requested_end=end_utc,
        missing_ranges=missing_ranges,
    )
    if not ranges:
        raise MarketDataNoMissingDataError("No missing data ranges detected in requested window")

    if len(ranges) > settings.market_data_sync_max_ranges:
        raise MarketDataSyncInputError(
            f"Too many missing ranges ({len(ranges)}). "
            f"Limit is {settings.market_data_sync_max_ranges}."
        )

    job = MarketDataSyncJob(
        user_id=user_id,
        provider=provider_key,
        market=market_key,
        symbol=symbol_key,
        timeframe=timeframe_key,
        requested_start=start_utc,
        requested_end=end_utc,
        missing_ranges=serialize_missing_ranges(ranges),
        status="queued",
        progress=0,
        current_step="queued",
        rows_written=0,
        error_message=None,
    )
    db.add(job)
    await db.flush()

    if auto_commit:
        await db.commit()
        await db.refresh(job)

    return MarketDataSyncJobReceipt(
        job_id=job.id,
        status=_to_external_status(job.status),
        progress=int(job.progress),
        deduplicated=False,
    )


async def execute_market_data_sync_job_with_fresh_session(job_id: UUID) -> MarketDataSyncJobView:
    """Execute one sync job with a dedicated DB session."""

    if db_module.AsyncSessionLocal is None:
        await db_module.init_postgres(ensure_schema=False)
    assert db_module.AsyncSessionLocal is not None

    async with db_module.AsyncSessionLocal() as session:
        return await execute_market_data_sync_job(session, job_id=job_id, auto_commit=True)


async def schedule_market_data_sync_job(job_id: UUID) -> str:
    """Enqueue one sync job and return Celery task id."""

    task_id = get_job_queue_ports().enqueue_market_data_sync_job(job_id)
    logger.info("[market-data-sync] enqueued job_id=%s celery_task_id=%s", job_id, task_id)
    return task_id


async def execute_market_data_sync_job(
    db: AsyncSession,
    *,
    job_id: UUID,
    auto_commit: bool = True,
) -> MarketDataSyncJobView:
    """Run one sync job from queued -> running -> completed/failed."""

    job = await db.scalar(
        select(MarketDataSyncJob)
        .options(selectinload(MarketDataSyncJob.chunks))
        .where(MarketDataSyncJob.id == job_id)
    )
    if job is None:
        raise MarketDataSyncJobNotFoundError(f"market_data_sync_job not found: {job_id}")

    if job.status == "completed":
        return _to_view(job)

    lock_key = _market_data_sync_lock_key(job)
    lock_value = str(job.id)
    lock_ttl = int(settings.market_data_sync_lock_ttl_seconds)
    lock_acquired = False
    redis_client = None
    try:
        await init_redis()
        redis_client = get_redis_client()
        lock_acquired = bool(
            await redis_client.set(lock_key, lock_value, nx=True, ex=max(lock_ttl, 1))
        )
    except Exception as exc:  # noqa: BLE001
        job.status = "failed"
        job.progress = 100
        job.current_step = "failed"
        job.error_message = f"Redis lock unavailable: {type(exc).__name__}: {exc}"
        job.completed_at = datetime.now(UTC)
        await _commit_if_requested(db, auto_commit=auto_commit)
        return _to_view(job)

    if not lock_acquired:
        job.status = "cancelled"
        job.progress = 100
        job.current_step = "cancelled_duplicate"
        job.error_message = "Another worker is syncing this market/symbol/timeframe"
        job.completed_at = datetime.now(UTC)
        await _commit_if_requested(db, auto_commit=auto_commit)
        return _to_view(job)

    provider_client = None
    close_provider = None
    try:
        job.status = "running"
        job.progress = 5
        job.current_step = "fetching"
        job.error_message = None
        await _commit_if_requested(db, auto_commit=auto_commit)

        provider_name = _normalize_provider(job.provider)
        timeframe_key = _normalize_timeframe(job.timeframe)
        ranges = deserialize_missing_ranges(list(job.missing_ranges or []))
        if not ranges:
            job.status = "failed"
            job.progress = 100
            job.current_step = "failed"
            job.error_message = "No missing ranges to process"
            job.completed_at = datetime.now(UTC)
            await _commit_if_requested(db, auto_commit=auto_commit)
            return _to_view(job)

        rows_written_total = int(job.rows_written or 0)
        errors: list[str] = []
        catalog_years_touched: set[int] = set()

        provider_client, close_provider = await _build_provider_client(provider_name)
        loader = DataLoader()

        total = len(ranges)
        fetch_concurrency = max(1, _FETCH_CONCURRENCY_BY_TIMEFRAME.get(timeframe_key, 1))
        request_semaphore = asyncio.Semaphore(fetch_concurrency)
        request_start_limiter = _RequestStartLimiter(
            min_interval_seconds=settings.market_data_sync_request_start_interval_seconds
        )
        completed = 0
        for batch_start in range(0, total, fetch_concurrency):
            batch_ranges = ranges[batch_start : batch_start + fetch_concurrency]
            batch_chunks: list[MarketDataSyncChunk] = []
            for offset, target_range in enumerate(batch_ranges):
                chunk = MarketDataSyncChunk(
                    job_id=job.id,
                    chunk_index=batch_start + offset,
                    chunk_start=target_range.start,
                    chunk_end=target_range.end,
                    fetched_rows=0,
                    written_rows=0,
                    status="pending",
                    metadata_={},
                    error_message=None,
                )
                db.add(chunk)
                batch_chunks.append(chunk)
            await db.flush()

            job.current_step = (
                f"fetching_batch_{(batch_start // fetch_concurrency) + 1}_of_"
                f"{((total - 1) // fetch_concurrency) + 1}"
            )
            await _commit_if_requested(db, auto_commit=auto_commit)

            batch_results = await asyncio.gather(
                *[
                    _fetch_provider_range(
                        provider_name=provider_name,
                        provider_client=provider_client,
                        market=job.market,
                        symbol=job.symbol,
                        timeframe=timeframe_key,
                        start=target_range.start,
                        end=target_range.end,
                        request_semaphore=request_semaphore,
                        request_start_limiter=request_start_limiter,
                    )
                    for target_range in batch_ranges
                ],
                return_exceptions=True,
            )

            for target_range, chunk, batch_result in zip(
                batch_ranges,
                batch_chunks,
                batch_results,
                strict=False,
            ):
                if isinstance(batch_result, Exception):
                    message = f"{type(batch_result).__name__}: {batch_result}"
                    errors.append(message)
                    chunk.status = "failed"
                    chunk.error_message = message
                else:
                    frame = _bars_to_frame(bars=batch_result)
                    try:
                        write_result = append_ohlcv_rows(
                            loader=loader,
                            market=job.market,
                            symbol=job.symbol,
                            timeframe=timeframe_key,
                            rows=frame,
                        )
                        rows_written_total += write_result.rows_written
                        catalog_years_touched.update(_extract_frame_years(frame))

                        chunk.fetched_rows = len(frame)
                        chunk.written_rows = write_result.rows_written
                        chunk.status = "completed"
                        chunk.metadata_ = {
                            "rows_input": write_result.rows_input,
                            "files_touched": write_result.files_touched,
                            "range_start": target_range.start.isoformat(),
                            "range_end": target_range.end.isoformat(),
                        }
                    except Exception as exc:  # noqa: BLE001
                        message = f"{type(exc).__name__}: {exc}"
                        errors.append(message)
                        chunk.status = "failed"
                        chunk.error_message = message

                completed += 1
                job.rows_written = rows_written_total
                job.progress = min(95, int((completed / total) * 90) + 5)
                job.current_step = f"processing_range_{completed}_of_{total}"
                await _commit_if_requested(db, auto_commit=auto_commit)

        if rows_written_total > 0 and catalog_years_touched:
            try:
                await _sync_catalog_for_job_years(
                    db=db,
                    loader=loader,
                    market=job.market,
                    symbol=job.symbol,
                    timeframe=timeframe_key,
                    years=sorted(catalog_years_touched),
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"CatalogSyncError: {type(exc).__name__}: {exc}")

        partial_failure = bool(errors and rows_written_total > 0)
        if errors:
            job.status = "failed"
            job.error_message = "; ".join(errors[:3])
        else:
            job.status = "completed"
            job.error_message = None

        job.progress = 100
        if job.status == "completed":
            job.current_step = "completed"
        elif partial_failure:
            job.current_step = "failed_partial"
        else:
            job.current_step = "failed"
        job.completed_at = datetime.now(UTC)
        await _commit_if_requested(db, auto_commit=auto_commit)
        refreshed = await db.scalar(
            select(MarketDataSyncJob)
            .options(selectinload(MarketDataSyncJob.chunks))
            .where(MarketDataSyncJob.id == job.id)
        )
        return _to_view(refreshed or job)
    finally:
        if close_provider is not None:
            with suppress(Exception):
                await close_provider()
        if redis_client is not None and lock_acquired:
            await _release_market_data_sync_lock(
                redis_client=redis_client,
                key=lock_key,
                lock_value=lock_value,
            )


async def get_market_data_sync_job_view(
    db: AsyncSession,
    *,
    job_id: UUID,
    user_id: UUID | None = None,
) -> MarketDataSyncJobView:
    """Load one sync job view with optional user scoping."""

    job = await db.scalar(
        select(MarketDataSyncJob)
        .options(selectinload(MarketDataSyncJob.chunks))
        .where(MarketDataSyncJob.id == job_id)
    )
    if job is None:
        raise MarketDataSyncJobNotFoundError(f"market_data_sync_job not found: {job_id}")
    if user_id is not None and job.user_id != user_id:
        raise MarketDataSyncJobNotFoundError(f"market_data_sync_job not found: {job_id}")

    return _to_view(job)


def _resolve_missing_ranges(
    *,
    loader: DataLoader,
    market: str,
    symbol: str,
    timeframe: str,
    requested_start: datetime,
    requested_end: datetime,
    missing_ranges: list[dict[str, Any]] | None,
) -> list[MissingRange]:
    if isinstance(missing_ranges, list) and missing_ranges:
        parsed = deserialize_missing_ranges(missing_ranges)
        ranges = [
            item
            for item in parsed
            if item.end >= requested_start and item.start <= requested_end
        ]
        return _split_missing_ranges_for_sync(ranges=ranges, timeframe=timeframe)

    try:
        coverage = detect_missing_ranges(
            loader=loader,
            market=market,
            symbol=symbol,
            timeframe=timeframe,
            start=requested_start,
            end=requested_end,
        )
    except LocalCoverageInputError as exc:
        raise MarketDataSyncInputError(str(exc)) from exc
    return _split_missing_ranges_for_sync(
        ranges=list(coverage.missing_ranges),
        timeframe=timeframe,
    )


def _to_view(job: MarketDataSyncJob) -> MarketDataSyncJobView:
    chunks = list(job.chunks or [])
    total_ranges = len(deserialize_missing_ranges(list(job.missing_ranges or [])))
    completed_ranges = sum(1 for item in chunks if item.status == "completed")
    errors = tuple(
        {
            "chunk_index": int(item.chunk_index),
            "message": str(item.error_message),
        }
        for item in chunks
        if item.status == "failed" and item.error_message
    )
    if not errors and isinstance(job.error_message, str) and job.error_message.strip():
        errors = ({"chunk_index": -1, "message": job.error_message.strip()},)
    if completed_ranges == 0 and job.status == "completed" and total_ranges > 0:
        completed_ranges = max(0, total_ranges - len(errors))
    return MarketDataSyncJobView(
        job_id=job.id,
        user_id=job.user_id,
        provider=job.provider,
        market=job.market,
        symbol=job.symbol,
        timeframe=job.timeframe,
        requested_start=_ensure_utc(job.requested_start),
        requested_end=_ensure_utc(job.requested_end),
        missing_ranges=tuple(deserialize_missing_ranges(list(job.missing_ranges or []))),
        status=_to_external_status(job.status),
        progress=int(job.progress),
        current_step=job.current_step,
        rows_written=int(job.rows_written or 0),
        range_filled=completed_ranges,
        total_ranges=total_ranges,
        errors=errors,
        submitted_at=_ensure_utc(job.submitted_at),
        completed_at=_ensure_utc(job.completed_at) if job.completed_at else None,
    )


async def _build_provider_client(provider_name: str) -> tuple[Any, Any]:
    if provider_name == "alpaca":
        client = AlpacaMarketDataClient()

        async def _close() -> None:
            await client.aclose()

        return client, _close

    if provider_name == "ccxt":
        if not settings.ccxt_market_data_enabled:
            raise MarketDataProviderUnavailableError("ccxt provider is disabled")
        try:
            client = CcxtRestProvider()
        except Exception as exc:  # noqa: BLE001
            raise MarketDataProviderUnavailableError(str(exc)) from exc

        async def _close() -> None:
            await client.aclose()

        return client, _close

    raise MarketDataProviderUnavailableError(f"Unsupported provider: {provider_name}")


async def _fetch_provider_range(
    *,
    provider_name: str,
    provider_client: Any,
    market: str,
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
    request_semaphore: asyncio.Semaphore | None = None,
    request_start_limiter: _RequestStartLimiter | None = None,
) -> list[OhlcvBar]:
    request_windows = _split_range_into_request_windows(
        start=start,
        end=end,
        timeframe=timeframe,
        limit=max(1, int(settings.market_data_sync_batch_limit)),
    )
    if not request_windows:
        return []

    timeframe_value = _to_alpaca_timeframe(timeframe) if provider_name == "alpaca" else timeframe
    window_results = await asyncio.gather(
        *[
            _fetch_provider_window(
                provider_client=provider_client,
                market=market,
                symbol=symbol,
                timeframe=timeframe_value,
                start=window.start,
                end=window.end,
                limit=max(1, min(int(settings.market_data_sync_batch_limit), int(window.bars))),
                request_semaphore=request_semaphore,
                request_start_limiter=request_start_limiter,
            )
            for window in request_windows
        ],
        return_exceptions=True,
    )

    for item in window_results:
        if isinstance(item, Exception):
            raise item

    bars: dict[datetime, OhlcvBar] = {}
    for fetched in window_results:
        assert not isinstance(fetched, Exception)
        normalized = sorted((_normalize_bar(item) for item in fetched), key=lambda item: item.timestamp)
        for item in normalized:
            if item.timestamp < start or item.timestamp > end:
                continue
            bars[item.timestamp] = item

    return [bars[key] for key in sorted(bars)]


async def _fetch_provider_window(
    *,
    provider_client: Any,
    market: str,
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
    limit: int,
    request_semaphore: asyncio.Semaphore | None,
    request_start_limiter: _RequestStartLimiter | None,
) -> list[OhlcvBar]:
    for attempt in range(2):
        try:
            if request_semaphore is None:
                if request_start_limiter is not None:
                    await request_start_limiter.wait_turn()
                return await provider_client.fetch_ohlcv(
                    symbol=symbol,
                    market=market,
                    timeframe=timeframe,
                    since=_ensure_utc(start),
                    until=_ensure_utc(end),
                    limit=max(1, int(limit)),
                )

            async with request_semaphore:
                if request_start_limiter is not None:
                    await request_start_limiter.wait_turn()
                return await provider_client.fetch_ohlcv(
                    symbol=symbol,
                    market=market,
                    timeframe=timeframe,
                    since=_ensure_utc(start),
                    until=_ensure_utc(end),
                    limit=max(1, int(limit)),
                )
        except Exception as exc:  # noqa: BLE001
            if not _is_rate_limit_error(exc) or attempt >= 1:
                raise
            await asyncio.sleep(float(attempt + 1))

    raise RuntimeError("unreachable")


def _split_range_into_request_windows(
    *,
    start: datetime,
    end: datetime,
    timeframe: str,
    limit: int,
) -> list[MissingRange]:
    safe_limit = max(1, int(limit))
    step = _timeframe_delta(timeframe)
    max_span = step * max(safe_limit - 1, 0)
    cursor = _ensure_utc(start)
    boundary = _ensure_utc(end)
    if boundary < cursor:
        return []

    windows: list[MissingRange] = []
    while cursor <= boundary:
        window_end = min(boundary, cursor + max_span)
        bars = max(int((window_end - cursor) / step) + 1, 1)
        windows.append(MissingRange(start=cursor, end=window_end, bars=bars))
        cursor = window_end + step
    return windows


def _bars_to_frame(*, bars: list[OhlcvBar]) -> pd.DataFrame:
    if not bars:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    rows: list[dict[str, Any]] = []
    for bar in bars:
        rows.append(
            {
                "timestamp": _ensure_utc(bar.timestamp),
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
            }
        )
    return pd.DataFrame(rows)


def _normalize_bar(bar: OhlcvBar) -> OhlcvBar:
    timestamp = _ensure_utc(bar.timestamp).replace(second=0, microsecond=0)
    return OhlcvBar(
        timestamp=timestamp,
        open=bar.open,
        high=bar.high,
        low=bar.low,
        close=bar.close,
        volume=bar.volume,
    )


def _is_rate_limit_error(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    return int(status_code or 0) == 429


def _normalize_provider(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in _SUPPORTED_PROVIDERS:
        raise MarketDataProviderUnavailableError(f"Unsupported provider: {value}")
    return normalized


def _normalize_symbol(value: str) -> str:
    normalized = str(value).strip().upper()
    if not normalized:
        raise MarketDataSyncInputError("symbol cannot be empty")
    return normalized


def _normalize_timeframe(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in DataLoader.TIMEFRAME_MINUTES:
        raise MarketDataSyncInputError(f"Unsupported timeframe: {value}")
    if normalized not in DataLoader.FILE_TIMEFRAME_MAP:
        raise MarketDataSyncInputError(
            f"Only local file timeframes are currently writable: {sorted(DataLoader.FILE_TIMEFRAME_MAP)}"
        )
    return normalized


def _to_alpaca_timeframe(timeframe: str) -> str:
    mapping = {
        "1m": "1Min",
        "5m": "5Min",
    }
    if timeframe not in mapping:
        raise MarketDataSyncInputError(f"Unsupported timeframe for alpaca provider: {timeframe}")
    return mapping[timeframe]


def _timeframe_delta(timeframe: str) -> timedelta:
    minutes = DataLoader.TIMEFRAME_MINUTES.get(timeframe)
    if minutes is None:
        raise MarketDataSyncInputError(f"Unsupported timeframe: {timeframe}")
    if minutes >= 1440:
        return timedelta(days=1)
    return timedelta(minutes=minutes)


def _split_missing_ranges_for_sync(
    *,
    ranges: list[MissingRange],
    timeframe: str,
) -> list[MissingRange]:
    max_span = _RANGE_SPLIT_SPAN_BY_TIMEFRAME.get(timeframe)
    if max_span is None or not ranges:
        return list(ranges)

    step = _timeframe_delta(timeframe)
    split_ranges: list[MissingRange] = []
    for item in ranges:
        start = _ensure_utc(item.start)
        end = _ensure_utc(item.end)
        if end <= start:
            split_ranges.append(MissingRange(start=start, end=end, bars=max(int(item.bars), 1)))
            continue

        cursor = start
        while cursor <= end:
            chunk_end = min(end, cursor + max_span - step)
            bars = max(int((chunk_end - cursor) / step) + 1, 1)
            split_ranges.append(
                MissingRange(
                    start=cursor,
                    end=chunk_end,
                    bars=bars,
                )
            )
            cursor = chunk_end + step
    return split_ranges


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _to_external_status(status: str) -> str:
    return _INTERNAL_TO_EXTERNAL_STATUS.get(status, status)


def _extract_frame_years(frame: pd.DataFrame) -> set[int]:
    if frame.empty or "timestamp" not in frame.columns:
        return set()
    timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce").dropna()
    if timestamps.empty:
        return set()
    return {int(item.year) for item in timestamps}


async def _sync_catalog_for_job_years(
    *,
    db: AsyncSession,
    loader: DataLoader,
    market: str,
    symbol: str,
    timeframe: str,
    years: list[int],
) -> None:
    file_timeframe = loader.FILE_TIMEFRAME_MAP[timeframe]
    session = loader.MARKET_SESSION_MAP.get(market, "eth")
    market_dir = loader.data_dir / market
    for year in years:
        file_path = market_dir / f"{symbol}_{file_timeframe}_{session}_{int(year)}.parquet"
        if not file_path.exists():
            continue
        await upsert_catalog_entry_from_parquet(
            db,
            market=market,
            symbol=symbol,
            timeframe=timeframe,
            session=session,
            year=int(year),
            file_path=file_path,
        )


def _market_data_sync_lock_key(job: MarketDataSyncJob) -> str:
    return (
        "market_data_sync:"
        f"{job.provider.strip().lower()}:"
        f"{job.market.strip().lower()}:"
        f"{job.symbol.strip().upper()}:"
        f"{job.timeframe.strip().lower()}"
    )


async def _release_market_data_sync_lock(
    *,
    redis_client: Any,
    key: str,
    lock_value: str,
) -> None:
    script = (
        "if redis.call('get', KEYS[1]) == ARGV[1] "
        "then return redis.call('del', KEYS[1]) "
        "else return 0 end"
    )
    try:
        await redis_client.eval(script, 1, key, lock_value)
    except Exception:  # noqa: BLE001
        logger.warning("[market-data-sync] lock release fallback key=%s", key)
        with suppress(Exception):
            await redis_client.delete(key)


async def _commit_if_requested(db: AsyncSession, *, auto_commit: bool) -> None:
    if not auto_commit:
        return
    await db.commit()
