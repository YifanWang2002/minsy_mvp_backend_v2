"""Backtest analytics endpoints for frontend chart rendering."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.middleware.auth import get_current_user
from src.dependencies import get_db
from src.engine.backtest.analytics import (
    build_backtest_overview,
    compute_entry_hour_pnl_heatmap,
    compute_entry_weekday_pnl,
    compute_equity_curve,
    compute_exit_reason_breakdown,
    compute_holding_period_pnl_bins,
    compute_long_short_breakdown,
    compute_monthly_return_table,
    compute_rolling_metrics,
    compute_underwater_curve,
)
from src.engine.backtest.service import create_backtest_job, execute_backtest_job
from src.models.backtest import BacktestJob
from src.models.strategy import Strategy
from src.models.user import User

router = APIRouter(prefix="/backtests", tags=["backtests"])

_STATUS_TO_EXTERNAL: dict[str, str] = {
    "queued": "pending",
    "running": "running",
    "completed": "done",
    "failed": "failed",
    "cancelled": "failed",
}

_SUPPORTED_ANALYSES: frozenset[str] = frozenset(
    {
        "overview",
        "equity_curve",
        "entry_hour_pnl_heatmap",
        "entry_weekday_pnl",
        "monthly_return_table",
        "holding_period_pnl_bins",
        "long_short_breakdown",
        "exit_reason_breakdown",
        "underwater_curve",
        "rolling_metrics",
    }
)


@router.get("/jobs/{job_id}/analysis/{analysis}")
async def get_backtest_analysis(
    job_id: UUID,
    analysis: str,
    max_points: int = Query(default=240, ge=10, le=5000),
    sampling: str = Query(default="auto"),
    window_bars: int = Query(default=0, ge=0, le=100_000),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    normalized_analysis = analysis.strip().lower()
    if normalized_analysis not in _SUPPORTED_ANALYSES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "BACKTEST_ANALYSIS_UNSUPPORTED",
                "message": f"Unsupported analysis: {analysis}",
                "supported_analyses": sorted(_SUPPORTED_ANALYSES),
            },
        )

    job = await db.scalar(
        select(BacktestJob).where(
            BacktestJob.id == job_id,
            BacktestJob.user_id == user.id,
        )
    )
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "BACKTEST_JOB_NOT_FOUND",
                "message": "Backtest job not found.",
            },
        )

    status_external = _STATUS_TO_EXTERNAL.get(job.status, "pending")
    result_payload = job.results if isinstance(job.results, dict) else None
    if status_external != "done" or result_payload is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "BACKTEST_JOB_NOT_READY",
                "message": "Backtest job is not completed yet.",
                "status": status_external,
            },
        )

    data = _compute_analysis_payload(
        analysis=normalized_analysis,
        result=result_payload,
        max_points=max_points,
        sampling=sampling,
        window_bars=window_bars,
    )

    return {
        "job_id": str(job.id),
        "strategy_id": str(job.strategy_id),
        "status": status_external,
        "analysis": normalized_analysis,
        "data": data,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
    }


@router.post("/jobs/demo/ensure")
async def ensure_demo_backtest_job(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Resolve one completed backtest job for widget-sample demo usage.

    Flow:
      1) Reuse the latest completed job for current user if available.
      2) Otherwise create+execute one real job from the user's latest strategies.
    """

    existing_job = await _find_latest_completed_job(db, user_id=user.id)
    if existing_job is not None:
        return _serialize_demo_job(existing_job, source="existing")

    candidate_strategies = (
        await db.scalars(
            select(Strategy)
            .where(Strategy.user_id == user.id)
            .order_by(Strategy.updated_at.desc(), Strategy.created_at.desc())
            .limit(5)
        )
    ).all()

    if not candidate_strategies:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "DEMO_BACKTEST_STRATEGY_NOT_FOUND",
                "message": "No strategy available to create a demo backtest job.",
            },
        )

    last_error: str | None = None
    for strategy in candidate_strategies:
        try:
            receipt = await create_backtest_job(
                db,
                strategy_id=strategy.id,
                auto_commit=True,
            )
            view = await execute_backtest_job(
                db,
                job_id=receipt.job_id,
                auto_commit=True,
            )
            if view.status == "done" and isinstance(view.result, dict):
                created_job = await db.scalar(
                    select(BacktestJob).where(BacktestJob.id == receipt.job_id)
                )
                if created_job is not None:
                    return _serialize_demo_job(created_job, source="created")
            last_error = _summarize_backtest_error(view.error)
        except Exception as exc:  # noqa: BLE001
            await db.rollback()
            last_error = f"{type(exc).__name__}: {exc}"

    raise HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail={
            "code": "DEMO_BACKTEST_JOB_UNAVAILABLE",
            "message": "Unable to find or create a completed backtest job for demo usage.",
            "last_error": last_error,
        },
    )


async def _find_latest_completed_job(
    db: AsyncSession,
    *,
    user_id: UUID,
) -> BacktestJob | None:
    jobs = (
        await db.scalars(
            select(BacktestJob)
            .where(
                BacktestJob.user_id == user_id,
                BacktestJob.status == "completed",
                BacktestJob.results.is_not(None),
            )
            .order_by(BacktestJob.completed_at.desc(), BacktestJob.submitted_at.desc())
            .limit(20)
        )
    ).all()
    for job in jobs:
        if isinstance(job.results, dict):
            return job
    return None


def _serialize_demo_job(job: BacktestJob, *, source: str) -> dict[str, Any]:
    return {
        "job_id": str(job.id),
        "strategy_id": str(job.strategy_id),
        "status": _STATUS_TO_EXTERNAL.get(job.status, "pending"),
        "source": source,
        "submitted_at": job.submitted_at.isoformat() if job.submitted_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
    }


def _summarize_backtest_error(error: Any) -> str:
    if not isinstance(error, dict):
        return str(error) if error is not None else "unknown"
    code = str(error.get("code", "")).strip()
    message = str(error.get("message", "")).strip()
    if code and message:
        return f"{code}: {message}"
    if code:
        return code
    if message:
        return message
    return "unknown"


def _compute_analysis_payload(
    *,
    analysis: str,
    result: dict[str, Any],
    max_points: int,
    sampling: str,
    window_bars: int,
) -> dict[str, Any]:
    if analysis == "overview":
        return build_backtest_overview(
            result,
            sample_trades=5,
            sample_events=5,
        )
    if analysis == "equity_curve":
        return compute_equity_curve(
            result,
            sampling_mode=sampling,
            max_points=max_points,
        )
    if analysis == "entry_hour_pnl_heatmap":
        return compute_entry_hour_pnl_heatmap(result)
    if analysis == "entry_weekday_pnl":
        return compute_entry_weekday_pnl(result)
    if analysis == "monthly_return_table":
        return compute_monthly_return_table(result)
    if analysis == "holding_period_pnl_bins":
        return compute_holding_period_pnl_bins(result)
    if analysis == "long_short_breakdown":
        return compute_long_short_breakdown(result)
    if analysis == "exit_reason_breakdown":
        return compute_exit_reason_breakdown(result)
    if analysis == "underwater_curve":
        return compute_underwater_curve(
            result,
            max_points=max_points,
        )
    if analysis == "rolling_metrics":
        return compute_rolling_metrics(
            result,
            window_bars=window_bars,
            max_points=max_points,
        )
    raise ValueError(f"Unsupported analysis: {analysis}")
