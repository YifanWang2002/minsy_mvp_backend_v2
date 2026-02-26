"""Celery tasks for backtest job execution."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from uuid import UUID

from packages.domain.backtest.service import execute_backtest_job_with_fresh_session
from packages.infra.db import session as db_module
from packages.infra.observability.logger import logger
from apps.worker.common.celery_base import celery_app


async def _run_backtest_job_once(job_uuid: UUID):
    # Celery worker tasks run in separate asyncio.run() loops; recreate async DB
    # resources per task to avoid cross-loop asyncpg futures.
    with suppress(Exception):
        await db_module.close_postgres()

    try:
        return await execute_backtest_job_with_fresh_session(job_uuid)
    finally:
        with suppress(Exception):
            await db_module.close_postgres()


@celery_app.task(
    name="backtest.execute_job",
    # Backtest tasks are memory-heavy and may be killed by OOM. Ack-early avoids
    # broker-side redelivery loops for the same poisoned payload.
    acks_late=False,
    reject_on_worker_lost=False,
)
def execute_backtest_job_task(job_id: str) -> dict[str, str | int]:
    """Execute one backtest job inside a Celery worker process."""
    job_uuid = UUID(job_id)
    logger.info("[backtest-worker] executing job_id=%s", job_uuid)
    view = asyncio.run(_run_backtest_job_once(job_uuid))
    logger.info(
        "[backtest-worker] finished job_id=%s status=%s progress=%s",
        view.job_id,
        view.status,
        view.progress,
    )
    return {
        "job_id": str(view.job_id),
        "status": view.status,
        "progress": view.progress,
        "current_step": view.current_step or "",
    }


def enqueue_backtest_job(job_id: UUID | str) -> str:
    """Enqueue backtest execution and return Celery task id."""
    result = execute_backtest_job_task.apply_async(args=(str(job_id),))
    return str(result.id)
