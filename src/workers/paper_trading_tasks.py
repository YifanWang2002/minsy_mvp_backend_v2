"""Celery tasks for paper-trading runtime orchestration."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from datetime import UTC, datetime
from math import ceil
from typing import Any
from uuid import UUID

from sqlalchemy import Select, select
from sqlalchemy.orm import selectinload

from src.config import settings
from src.engine.execution.deployment_lock import deployment_runtime_lock
from src.engine.execution.runtime_service import process_deployment_signal_cycle
from src.engine.execution.runtime_state_store import runtime_state_store
from src.engine.execution.timeframe_scheduler import (
    should_trigger_cycle,
    timeframe_to_seconds,
)
from src.engine.market_data.runtime import market_data_runtime
from src.models import database as db_module
from src.models.deployment import Deployment
from src.models.deployment_run import DeploymentRun
from src.models.redis import get_sync_redis_client
from src.util.logger import logger
from src.workers.celery_app import celery_app

_PAPER_TRADING_QUEUE_NAME = "paper_trading"


def _latest_run(deployment: Deployment) -> DeploymentRun | None:
    if not deployment.deployment_runs:
        return None
    return sorted(
        deployment.deployment_runs,
        key=lambda item: item.created_at,
        reverse=True,
    )[0]


def _scheduler_state(run: DeploymentRun) -> dict[str, Any]:
    state = run.runtime_state if isinstance(run.runtime_state, dict) else {}
    scheduler = state.get("scheduler")
    if isinstance(scheduler, dict):
        return scheduler
    return {}


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    normalized = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _resolve_timeframe_seconds(deployment: Deployment) -> int:
    payload = deployment.strategy.dsl_payload if isinstance(deployment.strategy.dsl_payload, dict) else {}
    raw = payload.get("timeframe") or deployment.strategy.timeframe or "1m"
    return timeframe_to_seconds(str(raw), default_seconds=60)


def _resolve_runtime_task_expires_seconds(interval_seconds: int | float | None = None) -> int:
    base_seconds = max(1.0, float(settings.paper_trading_runtime_task_expires_seconds))
    if interval_seconds is None:
        return int(base_seconds)
    return max(1, int(max(base_seconds, float(interval_seconds) * 2.0)))


def _paper_trading_queue_backlog() -> int | None:
    try:
        redis = get_sync_redis_client()
        return int(redis.llen(_PAPER_TRADING_QUEUE_NAME))
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[paper-worker] queue backlog probe failed queue=%s error=%s",
            _PAPER_TRADING_QUEUE_NAME,
            type(exc).__name__,
        )
        return None


def _update_scheduler_runtime_state(
    *,
    run: DeploymentRun,
    now: datetime,
    timeframe_seconds: int,
    last_trigger_bucket: int | None = None,
) -> None:
    state = dict(run.runtime_state) if isinstance(run.runtime_state, dict) else {}
    scheduler = dict(_scheduler_state(run))
    scheduler["timeframe_seconds"] = int(timeframe_seconds)
    scheduler["updated_at"] = now.isoformat()
    if last_trigger_bucket is not None:
        scheduler["last_trigger_bucket"] = int(last_trigger_bucket)
        scheduler["last_enqueued_at"] = now.isoformat()
    state["scheduler"] = scheduler
    run.runtime_state = state


async def _maybe_enqueue_runtime_fallback(
    *,
    deployment: Deployment,
    run: DeploymentRun,
    now: datetime,
) -> str | None:
    if deployment.status != "active":
        return None
    if run.status in {"paused", "stopped", "error"}:
        return None

    scheduler = dict(_scheduler_state(run))
    loop_interval_seconds = max(0.2, float(settings.paper_trading_loop_interval_seconds))
    grace_seconds = max(loop_interval_seconds * 2.0, 2.0)

    last_enqueued_at = _parse_datetime(scheduler.get("last_enqueued_at"))
    if last_enqueued_at is not None and (now - last_enqueued_at).total_seconds() <= grace_seconds:
        return None

    fallback_last_enqueued_at = _parse_datetime(scheduler.get("fallback_last_enqueued_at"))
    if fallback_last_enqueued_at is not None and (now - fallback_last_enqueued_at).total_seconds() <= grace_seconds:
        return None

    timeframe_seconds = _resolve_timeframe_seconds(deployment)
    task_id = enqueue_paper_trading_runtime(
        deployment.id,
        delay_seconds=loop_interval_seconds,
        expires_seconds=_resolve_runtime_task_expires_seconds(timeframe_seconds),
    )
    scheduler["timeframe_seconds"] = timeframe_seconds
    scheduler["last_enqueued_at"] = now.isoformat()
    scheduler["fallback_last_enqueued_at"] = now.isoformat()
    scheduler["fallback_task_id"] = task_id
    scheduler["updated_at"] = now.isoformat()

    state = dict(run.runtime_state) if isinstance(run.runtime_state, dict) else {}
    state["scheduler"] = scheduler
    run.runtime_state = state
    return task_id


async def _run_paper_scheduler_tick() -> dict[str, Any]:
    with suppress(Exception):
        await db_module.close_postgres()

    try:
        await db_module.init_postgres(ensure_schema=False)
        assert db_module.AsyncSessionLocal is not None
        async with db_module.AsyncSessionLocal() as session:
            now = datetime.now(UTC)
            queue_backlog = _paper_trading_queue_backlog()
            backlog_soft_limit = max(0, int(settings.paper_trading_queue_backlog_soft_limit))
            backlog_overloaded = (
                queue_backlog is not None
                and backlog_soft_limit > 0
                and queue_backlog >= backlog_soft_limit
            )
            max_enqueues_per_tick = max(0, int(settings.paper_trading_scheduler_max_enqueues_per_tick))
            loop_interval_seconds = max(0.2, float(settings.paper_trading_loop_interval_seconds))
            starting_retry_seconds = max(
                loop_interval_seconds * 2.0,
                float(settings.paper_trading_starting_retry_seconds),
            )
            stmt: Select[tuple[Deployment]] = (
                select(Deployment)
                .options(
                    selectinload(Deployment.strategy),
                    selectinload(Deployment.deployment_runs),
                )
                .where(
                    Deployment.mode == "paper",
                    Deployment.status == "active",
                )
                .order_by(Deployment.created_at.asc())
            )
            deployments = list((await session.scalars(stmt)).all())
            if not deployments:
                return {
                    "status": "ok",
                    "deployments_total": 0,
                    "enqueued": 0,
                    "skipped": 0,
                    "locked": 0,
                    "queue_backlog": queue_backlog,
                    "backlog_overloaded": backlog_overloaded,
                }

            enqueued = 0
            skipped = 0
            locked = 0
            for deployment in deployments:
                lease = await deployment_runtime_lock.acquire(deployment.id)
                if lease is None:
                    locked += 1
                    skipped += 1
                    continue
                try:
                    run = _latest_run(deployment)
                    if run is None:
                        skipped += 1
                        continue
                    if run.status in {"paused", "stopped", "error"}:
                        skipped += 1
                        continue

                    interval_seconds = _resolve_timeframe_seconds(deployment)
                    scheduler = _scheduler_state(run)
                    if max_enqueues_per_tick and enqueued >= max_enqueues_per_tick and run.status != "starting":
                        skipped += 1
                        continue

                    if run.status == "starting":
                        last_enqueued_at = _parse_datetime(scheduler.get("last_enqueued_at"))
                        should_retry_starting = (
                            last_enqueued_at is None
                            or (now - last_enqueued_at).total_seconds() >= starting_retry_seconds
                        )
                        if should_retry_starting:
                            enqueue_paper_trading_runtime(
                                deployment.id,
                                expires_seconds=_resolve_runtime_task_expires_seconds(interval_seconds),
                            )
                            scheduler["timeframe_seconds"] = interval_seconds
                            scheduler["last_enqueued_at"] = now.isoformat()
                            scheduler["starting_retry_enqueued_at"] = now.isoformat()
                            scheduler["updated_at"] = now.isoformat()
                            state = dict(run.runtime_state) if isinstance(run.runtime_state, dict) else {}
                            state["scheduler"] = scheduler
                            run.runtime_state = state
                            await runtime_state_store.upsert(deployment.id, state)
                            enqueued += 1
                        else:
                            _update_scheduler_runtime_state(
                                run=run,
                                now=now,
                                timeframe_seconds=interval_seconds,
                            )
                            payload = dict(run.runtime_state) if isinstance(run.runtime_state, dict) else {}
                            await runtime_state_store.upsert(deployment.id, payload)
                            skipped += 1
                        continue

                    if backlog_overloaded:
                        skipped += 1
                        continue

                    raw_bucket = scheduler.get("last_trigger_bucket")
                    try:
                        last_bucket = int(raw_bucket) if raw_bucket is not None else None
                    except (TypeError, ValueError):
                        last_bucket = None

                    due, bucket = should_trigger_cycle(
                        now=now,
                        interval_seconds=interval_seconds,
                        last_trigger_bucket=last_bucket,
                    )
                    if not due:
                        _update_scheduler_runtime_state(
                            run=run,
                            now=now,
                            timeframe_seconds=interval_seconds,
                        )
                        payload = dict(run.runtime_state) if isinstance(run.runtime_state, dict) else {}
                        await runtime_state_store.upsert(deployment.id, payload)
                        skipped += 1
                        continue

                    enqueue_paper_trading_runtime(
                        deployment.id,
                        expires_seconds=_resolve_runtime_task_expires_seconds(interval_seconds),
                    )
                    _update_scheduler_runtime_state(
                        run=run,
                        now=now,
                        timeframe_seconds=interval_seconds,
                        last_trigger_bucket=bucket,
                    )
                    payload = dict(run.runtime_state) if isinstance(run.runtime_state, dict) else {}
                    await runtime_state_store.upsert(deployment.id, payload)
                    enqueued += 1
                finally:
                    await deployment_runtime_lock.release(lease)

            await session.commit()
            return {
                "status": "ok",
                "deployments_total": len(deployments),
                "enqueued": enqueued,
                "skipped": skipped,
                "locked": locked,
                "queue_backlog": queue_backlog,
                "backlog_overloaded": backlog_overloaded,
            }
    finally:
        with suppress(Exception):
            await db_module.close_postgres()


async def _run_paper_runtime_once(deployment_id: UUID) -> dict[str, Any]:
    with suppress(Exception):
        await db_module.close_postgres()

    try:
        await db_module.init_postgres(ensure_schema=False)
        assert db_module.AsyncSessionLocal is not None
        async with db_module.AsyncSessionLocal() as session:
            deployment = await session.scalar(
                select(Deployment)
                .options(
                    selectinload(Deployment.deployment_runs),
                    selectinload(Deployment.strategy),
                )
                .where(Deployment.id == deployment_id)
            )
            if deployment is None:
                return {"deployment_id": str(deployment_id), "status": "missing"}

            run = _latest_run(deployment)
            if run is None:
                return {"deployment_id": str(deployment_id), "status": "run_missing"}
            run_id = run.id
            deployment_uuid = deployment.id

            if deployment.status != "active":
                market_data_runtime.unsubscribe(f"deployment:{deployment.id}")
                run.status = "stopped" if deployment.status == "stopped" else deployment.status
                await session.commit()
                return {"deployment_id": str(deployment_id), "status": run.status}

            try:
                result = await process_deployment_signal_cycle(
                    session,
                    deployment_id=deployment_id,
                )
            except Exception as exc:  # noqa: BLE001
                with suppress(Exception):
                    await session.rollback()
                reason = f"runtime_exception:{type(exc).__name__}"
                fresh_run = await session.scalar(
                    select(DeploymentRun).where(DeploymentRun.id == run_id).limit(1)
                )
                if fresh_run is None:
                    return {
                        "deployment_id": str(deployment_id),
                        "status": "error",
                        "reason": reason,
                    }
                fresh_run.status = "error"
                state = dict(fresh_run.runtime_state) if isinstance(fresh_run.runtime_state, dict) else {}
                state.update(
                    {
                        "runtime_status": "error",
                        "runtime_reason": reason,
                        "last_runtime_exception": reason,
                        "last_updated_at": datetime.now(UTC).isoformat(),
                    }
                )
                fresh_run.runtime_state = state
                await session.commit()
                await runtime_state_store.upsert(deployment_uuid, state)
                return {
                    "deployment_id": str(deployment_id),
                    "status": "error",
                    "reason": reason,
                }
            fallback_task_id = await _maybe_enqueue_runtime_fallback(
                deployment=deployment,
                run=run,
                now=datetime.now(UTC),
            )
            await session.commit()
            payload: dict[str, Any] = {
                "deployment_id": str(deployment_id),
                "status": run.status,
                "signal": result.signal,
                "reason": result.reason,
            }
            if fallback_task_id is not None:
                payload["fallback_task_id"] = fallback_task_id
            return payload
    finally:
        with suppress(Exception):
            await db_module.close_postgres()


@celery_app.task(name="paper_trading.run_deployment_runtime")
def run_deployment_runtime_task(deployment_id: str) -> dict[str, Any]:
    """Run one runtime heartbeat tick for a deployment."""
    deployment_uuid = UUID(deployment_id)
    logger.info("[paper-worker] running deployment runtime deployment_id=%s", deployment_uuid)
    try:
        return asyncio.run(_run_paper_runtime_once(deployment_uuid))
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "[paper-worker] runtime task failed deployment_id=%s error=%s",
            deployment_uuid,
            type(exc).__name__,
        )
        return {
            "deployment_id": str(deployment_uuid),
            "status": "error",
            "reason": f"runtime_exception:{type(exc).__name__}",
        }


def enqueue_paper_trading_runtime(
    deployment_id: UUID | str,
    *,
    delay_seconds: float | None = None,
    expires_seconds: float | None = None,
) -> str:
    """Enqueue one paper-trading runtime tick."""
    countdown = None
    if delay_seconds is not None:
        countdown = max(0.0, float(delay_seconds))
    expires = _resolve_runtime_task_expires_seconds()
    if expires_seconds is not None:
        expires = max(1, int(float(expires_seconds)))
    if countdown is not None and expires <= ceil(countdown):
        expires = ceil(countdown) + 1
    result = run_deployment_runtime_task.apply_async(
        args=(str(deployment_id),),
        queue=_PAPER_TRADING_QUEUE_NAME,
        countdown=countdown,
        expires=expires,
    )
    return str(result.id)


@celery_app.task(name="paper_trading.scheduler_tick")
def scheduler_tick_task() -> dict[str, Any]:
    """Scan active deployments and enqueue due runtime ticks by timeframe bucket."""
    if not settings.paper_trading_enabled:
        return {
            "status": "disabled",
            "deployments_total": 0,
            "enqueued": 0,
            "skipped": 0,
            "locked": 0,
        }
    logger.info("[paper-worker] running scheduler tick")
    return asyncio.run(_run_paper_scheduler_tick())
