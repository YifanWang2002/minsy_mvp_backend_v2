"""Backtest job orchestration and persistence service."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.engine.backtest.analytics import build_compact_performance_payload
from src.engine.backtest.engine import EventDrivenBacktestEngine
from src.engine.backtest.types import BacktestConfig
from src.engine.data import DataLoader
from src.engine.strategy import parse_strategy_payload
from src.models import database as db_module
from src.models.backtest import BacktestJob
from src.models.strategy import Strategy
from src.util.logger import logger

_INTERNAL_TO_EXTERNAL_STATUS: dict[str, str] = {
    "queued": "pending",
    "running": "running",
    "completed": "done",
    "failed": "failed",
    "cancelled": "failed",
}


@dataclass(frozen=True, slots=True)
class BacktestJobReceipt:
    """Create-job response."""

    job_id: UUID
    strategy_id: UUID
    status: str
    progress: int
    created_at: datetime


@dataclass(frozen=True, slots=True)
class BacktestJobView:
    """Status/result view for one job."""

    job_id: UUID
    strategy_id: UUID
    status: str
    progress: int
    current_step: str | None
    result: dict[str, Any] | None
    error: dict[str, Any] | None
    submitted_at: datetime
    completed_at: datetime | None


class BacktestJobNotFoundError(LookupError):
    """Raised when a backtest job does not exist."""


class BacktestStrategyNotFoundError(LookupError):
    """Raised when strategy does not exist for a job create request."""


async def create_backtest_job(
    db: AsyncSession,
    *,
    strategy_id: UUID,
    start_date: str = "",
    end_date: str = "",
    initial_capital: float = 100_000.0,
    commission_rate: float = 0.0,
    slippage_bps: float = 0.0,
    auto_commit: bool = True,
) -> BacktestJobReceipt:
    """Create a queued backtest job."""

    strategy = await db.scalar(select(Strategy).where(Strategy.id == strategy_id))
    if strategy is None:
        raise BacktestStrategyNotFoundError(f"Strategy not found: {strategy_id}")

    initial_capital_value, commission_rate_value, slippage_bps_value = (
        _validated_backtest_config_values(
            initial_capital=initial_capital,
            commission_rate=commission_rate,
            slippage_bps=slippage_bps,
        )
    )

    config = {
        "start_date": start_date.strip(),
        "end_date": end_date.strip(),
        "initial_capital": initial_capital_value,
        "commission_rate": commission_rate_value,
        "slippage_bps": slippage_bps_value,
    }
    job = BacktestJob(
        strategy_id=strategy.id,
        user_id=strategy.user_id,
        session_id=strategy.session_id,
        status="queued",
        progress=0,
        current_step="pending",
        config=config,
        results=None,
        error_message=None,
    )
    db.add(job)
    await db.flush()

    if auto_commit:
        await db.commit()
        await db.refresh(job)

    return BacktestJobReceipt(
        job_id=job.id,
        strategy_id=job.strategy_id,
        status=_to_external_status(job.status),
        progress=job.progress,
        created_at=_to_utc(job.submitted_at),
    )


async def execute_backtest_job_with_fresh_session(job_id: UUID) -> BacktestJobView:
    """Execute one job with a dedicated database session."""
    if db_module.AsyncSessionLocal is None:
        # Worker tasks may run concurrently across multiple processes.
        # Avoid DDL/schema-migration steps here to prevent lock contention;
        # schema should already be managed by service startup/migrations.
        await db_module.init_postgres(ensure_schema=False)
    assert db_module.AsyncSessionLocal is not None

    async with db_module.AsyncSessionLocal() as session:
        return await execute_backtest_job(session, job_id=job_id, auto_commit=True)


def _enqueue_backtest_job(job_id: UUID) -> str:
    from src.workers.backtest_tasks import enqueue_backtest_job

    return enqueue_backtest_job(job_id)


async def schedule_backtest_job(job_id: UUID) -> str:
    """Enqueue a backtest job for worker-side execution."""
    task_id = _enqueue_backtest_job(job_id)
    logger.info("[backtest] enqueued job_id=%s celery_task_id=%s", job_id, task_id)
    return task_id


async def execute_backtest_job(
    db: AsyncSession,
    *,
    job_id: UUID,
    auto_commit: bool = True,
) -> BacktestJobView:
    """Run one backtest job from queued -> running -> completed/failed."""

    job = await db.scalar(select(BacktestJob).where(BacktestJob.id == job_id))
    if job is None:
        raise BacktestJobNotFoundError(f"Backtest job not found: {job_id}")

    if job.status == "completed" and isinstance(job.results, dict):
        return _to_view(job)

    strategy = await db.scalar(select(Strategy).where(Strategy.id == job.strategy_id))
    if strategy is None:
        _mark_job_failed(
            job,
            error_code="STRATEGY_NOT_FOUND",
            message=f"Strategy not found: {job.strategy_id}",
        )
        if auto_commit:
            await db.commit()
        return _to_view(job)

    _mark_job_running_step(job, progress=5, step="loading_data")
    await _commit_if_requested(db, auto_commit=auto_commit)

    try:
        parsed = parse_strategy_payload(strategy.dsl_payload or {})
        symbol = _pick_symbol(parsed.universe.tickers)
        market = parsed.universe.market
        timeframe = parsed.universe.timeframe

        loader = DataLoader()
        start_date, end_date = _resolve_timerange(
            loader=loader,
            market=market,
            symbol=symbol,
            config=job.config or {},
        )

        _mark_job_running_step(job, progress=20, step="running_engine")
        await _commit_if_requested(db, auto_commit=auto_commit)

        frame = loader.load(
            market=market,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
        )

        config = _build_backtest_config(job.config or {})
        result = EventDrivenBacktestEngine(
            strategy=parsed,
            data=frame,
            config=config,
        ).run()

        _mark_job_running_step(job, progress=95, step="serializing_result")
        await _commit_if_requested(db, auto_commit=auto_commit)

        serialized_result = _serialize_backtest_result(
            result=result,
            market=market,
            symbol=symbol,
            timeframe=timeframe,
        )
        _mark_job_completed(job, result=serialized_result)

    except Exception as exc:  # noqa: BLE001
        _mark_job_failed(
            job,
            error_code="BACKTEST_RUN_ERROR",
            message=f"{type(exc).__name__}: {exc}",
        )

    if auto_commit:
        await db.commit()
        await db.refresh(job)

    return _to_view(job)


async def get_backtest_job_view(
    db: AsyncSession,
    *,
    job_id: UUID,
) -> BacktestJobView:
    """Get status and optional result payload for one job."""

    job = await db.scalar(select(BacktestJob).where(BacktestJob.id == job_id))
    if job is None:
        raise BacktestJobNotFoundError(f"Backtest job not found: {job_id}")
    return _to_view(job)


def _serialize_backtest_result(
    *,
    result: Any,
    market: str,
    symbol: str,
    timeframe: str,
) -> dict[str, Any]:
    payload = {
        "market": market,
        "symbol": symbol,
        "timeframe": timeframe,
        "summary": {
            "total_trades": result.summary.total_trades,
            "winning_trades": result.summary.winning_trades,
            "losing_trades": result.summary.losing_trades,
            "win_rate": result.summary.win_rate,
            "total_pnl": result.summary.total_pnl,
            "total_return_pct": result.summary.total_return_pct,
            "final_equity": result.summary.final_equity,
            "max_drawdown_pct": result.summary.max_drawdown_pct,
        },
        "trades": [
            {
                "side": trade.side.value,
                "entry_time": trade.entry_time.isoformat(),
                "exit_time": trade.exit_time.isoformat(),
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
                "quantity": trade.quantity,
                "bars_held": trade.bars_held,
                "exit_reason": trade.exit_reason,
                "pnl": trade.pnl,
                "pnl_pct": trade.pnl_pct,
                "commission": trade.commission,
            }
            for trade in result.trades
        ],
        "equity_curve": [
            {"timestamp": point.timestamp.isoformat(), "equity": point.equity}
            for point in result.equity_curve
        ],
        "returns": list(result.returns),
        "events": [
            {
                "type": event.type.value,
                "timestamp": event.timestamp.isoformat(),
                "bar_index": event.bar_index,
                "payload": event.payload,
            }
            for event in result.events
        ],
        "performance": result.performance,
        "started_at": result.started_at.isoformat(),
        "finished_at": result.finished_at.isoformat(),
    }
    payload["performance"] = build_compact_performance_payload(payload)
    return payload


def _resolve_timerange(
    *,
    loader: DataLoader,
    market: str,
    symbol: str,
    config: dict[str, Any],
) -> tuple[datetime, datetime]:
    raw_start = str(config.get("start_date", "")).strip()
    raw_end = str(config.get("end_date", "")).strip()

    if raw_start and raw_end:
        start = _parse_dt(raw_start)
        end = _parse_dt(raw_end)
        return _validated_timerange(start=start, end=end)

    metadata = loader.get_symbol_metadata(market, symbol)
    available = metadata.get("available_timerange", {})
    metadata_start = _parse_dt(str(available.get("start")))
    metadata_end = _parse_dt(str(available.get("end")))
    start = _parse_dt(raw_start) if raw_start else metadata_start
    end = _parse_dt(raw_end) if raw_end else metadata_end
    return _validated_timerange(start=start, end=end)


def _parse_dt(value: str) -> datetime:
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _pick_symbol(symbols: tuple[str, ...]) -> str:
    if not symbols:
        raise ValueError("Strategy universe must contain at least one symbol.")
    return symbols[0]


def _validated_backtest_config_values(
    *,
    initial_capital: float,
    commission_rate: float,
    slippage_bps: float,
) -> tuple[float, float, float]:
    initial_capital_value = float(initial_capital)
    commission_rate_value = float(commission_rate)
    slippage_bps_value = float(slippage_bps)

    if initial_capital_value <= 0:
        raise ValueError("initial_capital must be > 0")
    if commission_rate_value < 0:
        raise ValueError("commission_rate must be >= 0")
    if slippage_bps_value < 0:
        raise ValueError("slippage_bps must be >= 0")

    return initial_capital_value, commission_rate_value, slippage_bps_value


def _parse_record_bar_events(config: dict[str, Any]) -> bool:
    raw_record_bar_events = config.get("record_bar_events", False)
    if isinstance(raw_record_bar_events, bool):
        return raw_record_bar_events
    return str(raw_record_bar_events).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _build_backtest_config(config: dict[str, Any]) -> BacktestConfig:
    record_bar_events = _parse_record_bar_events(config)
    return BacktestConfig(
        initial_capital=float(config.get("initial_capital", 100_000.0)),
        commission_rate=float(config.get("commission_rate", 0.0)),
        slippage_bps=float(config.get("slippage_bps", 0.0)),
        record_bar_events=record_bar_events,
    )


def _validated_timerange(*, start: datetime, end: datetime) -> tuple[datetime, datetime]:
    if end < start:
        raise ValueError("end_date must be greater than or equal to start_date")
    return (start, end)


async def _commit_if_requested(
    db: AsyncSession,
    *,
    auto_commit: bool,
) -> None:
    if auto_commit:
        await db.commit()


def _mark_job_running_step(
    job: BacktestJob,
    *,
    progress: int,
    step: str,
) -> None:
    job.status = "running"
    job.progress = progress
    job.current_step = step


def _mark_job_completed(
    job: BacktestJob,
    *,
    result: dict[str, Any],
) -> None:
    job.results = result
    job.status = "completed"
    job.progress = 100
    job.current_step = "done"
    job.completed_at = datetime.now(UTC)
    job.error_message = None


def _mark_job_failed(
    job: BacktestJob,
    *,
    error_code: str,
    message: str,
) -> None:
    job.status = "failed"
    job.progress = 100
    job.current_step = "failed"
    job.completed_at = datetime.now(UTC)
    job.error_message = message
    job.results = {
        "error": {
            "code": error_code,
            "message": message,
        }
    }


def _to_view(job: BacktestJob) -> BacktestJobView:
    results = job.results if isinstance(job.results, dict) else None
    error = None
    if results and isinstance(results.get("error"), dict):
        error = results["error"]
    elif job.error_message:
        error = {
            "code": "BACKTEST_RUN_ERROR",
            "message": job.error_message,
        }

    return BacktestJobView(
        job_id=job.id,
        strategy_id=job.strategy_id,
        status=_to_external_status(job.status),
        progress=job.progress,
        current_step=job.current_step,
        result=results if job.status == "completed" else None,
        error=error,
        submitted_at=_to_utc(job.submitted_at),
        completed_at=_to_utc(job.completed_at) if job.completed_at else None,
    )


def _to_external_status(internal_status: str) -> str:
    return _INTERNAL_TO_EXTERNAL_STATUS.get(internal_status, "failed")


def _to_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
