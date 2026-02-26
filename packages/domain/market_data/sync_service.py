"""Missing-range market-data sync job orchestration service."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from packages.shared_settings.schema.settings import settings
from packages.domain.market_data.data import DataLoader
from packages.domain.market_data.data.local_coverage import (
    LocalCoverageInputError,
    MissingRange,
    deserialize_missing_ranges,
    detect_missing_ranges,
    serialize_missing_ranges,
)
from packages.domain.market_data.data.parquet_writer import append_ohlcv_rows
from packages.infra.providers.trading.adapters.base import OhlcvBar
from packages.infra.providers.market_data.alpaca_client import AlpacaMarketDataClient
from packages.infra.providers.market_data.ccxt_rest import CcxtRestProvider
from packages.domain.ports.queue_ports import get_job_queue_ports
from packages.infra.db import session as db_module
from packages.infra.db.models.market_data_sync_chunk import MarketDataSyncChunk
from packages.infra.db.models.market_data_sync_job import MarketDataSyncJob
from packages.infra.observability.logger import logger

_INTERNAL_TO_EXTERNAL_STATUS: dict[str, str] = {
    "queued": "pending",
    "running": "running",
    "completed": "done",
    "failed": "failed",
    "cancelled": "failed",
}

_SUPPORTED_PROVIDERS: frozenset[str] = frozenset({"alpaca", "ccxt"})


@dataclass(frozen=True, slots=True)
class MarketDataSyncJobReceipt:
    """Create-job response payload."""

    job_id: UUID
    status: str
    progress: int


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

    provider_client, close_provider = await _build_provider_client(provider_name)
    loader = DataLoader()

    try:
        total = len(ranges)
        for index, target_range in enumerate(ranges):
            chunk = MarketDataSyncChunk(
                job_id=job.id,
                chunk_index=index,
                chunk_start=target_range.start,
                chunk_end=target_range.end,
                fetched_rows=0,
                written_rows=0,
                status="pending",
                metadata_={},
                error_message=None,
            )
            db.add(chunk)
            await db.flush()

            try:
                bars = await _fetch_provider_range(
                    provider_name=provider_name,
                    provider_client=provider_client,
                    market=job.market,
                    symbol=job.symbol,
                    timeframe=timeframe_key,
                    start=target_range.start,
                    end=target_range.end,
                )
                frame = _bars_to_frame(bars=bars)
                write_result = append_ohlcv_rows(
                    loader=loader,
                    market=job.market,
                    symbol=job.symbol,
                    timeframe=timeframe_key,
                    rows=frame,
                )
                rows_written_total += write_result.rows_written

                chunk.fetched_rows = len(frame)
                chunk.written_rows = write_result.rows_written
                chunk.status = "completed"
                chunk.metadata_ = {
                    "rows_input": write_result.rows_input,
                    "files_touched": write_result.files_touched,
                }
            except Exception as exc:  # noqa: BLE001
                message = f"{type(exc).__name__}: {exc}"
                errors.append(message)
                chunk.status = "failed"
                chunk.error_message = message

            completed = index + 1
            job.rows_written = rows_written_total
            job.progress = min(95, int((completed / total) * 90) + 5)
            job.current_step = f"processing_range_{completed}_of_{total}"
            await _commit_if_requested(db, auto_commit=auto_commit)

        if errors and rows_written_total == 0:
            job.status = "failed"
            job.error_message = errors[0]
        else:
            job.status = "completed"
            job.error_message = "; ".join(errors[:3]) if errors else None

        job.progress = 100
        job.current_step = "completed" if job.status == "completed" else "failed"
        job.completed_at = datetime.now(UTC)
        await _commit_if_requested(db, auto_commit=auto_commit)
        refreshed = await db.scalar(
            select(MarketDataSyncJob)
            .options(selectinload(MarketDataSyncJob.chunks))
            .where(MarketDataSyncJob.id == job.id)
        )
        return _to_view(refreshed or job)
    finally:
        await close_provider()


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
        return [
            item
            for item in parsed
            if item.end >= requested_start and item.start <= requested_end
        ]

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
    return list(coverage.missing_ranges)


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
) -> list[OhlcvBar]:
    bars: dict[datetime, OhlcvBar] = {}
    step = _timeframe_delta(timeframe)
    cursor = _ensure_utc(start)
    boundary = _ensure_utc(end)
    batch_limit = settings.market_data_sync_batch_limit

    while cursor <= boundary:
        if provider_name == "alpaca":
            fetched = await provider_client.fetch_ohlcv(
                symbol=symbol,
                market=market,
                timeframe=_to_alpaca_timeframe(timeframe),
                since=cursor,
                limit=batch_limit,
            )
        else:
            fetched = await provider_client.fetch_ohlcv(
                symbol=symbol,
                market=market,
                timeframe=timeframe,
                since=cursor,
                limit=batch_limit,
            )

        if not fetched:
            break

        normalized = sorted((_normalize_bar(item) for item in fetched), key=lambda item: item.timestamp)
        for item in normalized:
            if item.timestamp < start or item.timestamp > end:
                continue
            bars[item.timestamp] = item

        last_timestamp = max(item.timestamp for item in normalized)
        next_cursor = last_timestamp + step
        if next_cursor <= cursor:
            break
        cursor = next_cursor

        if len(fetched) < batch_limit and last_timestamp < boundary:
            break

    return [bars[key] for key in sorted(bars)]


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


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _to_external_status(status: str) -> str:
    return _INTERNAL_TO_EXTERNAL_STATUS.get(status, status)


async def _commit_if_requested(db: AsyncSession, *, auto_commit: bool) -> None:
    if not auto_commit:
        return
    await db.commit()
