"""Celery tasks for stress job execution."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from uuid import UUID

from packages.domain.stress import execute_stress_job_with_fresh_session
from packages.infra.db import session as db_module
from packages.infra.observability.logger import logger
from apps.worker.common.celery_base import celery_app


async def _run_stress_job_once(job_uuid: UUID):
    with suppress(Exception):
        await db_module.close_postgres()

    try:
        return await execute_stress_job_with_fresh_session(job_uuid)
    finally:
        with suppress(Exception):
            await db_module.close_postgres()


@celery_app.task(name="stress.execute_job")
def execute_stress_job_task(job_id: str) -> dict[str, str | int]:
    """Execute one stress job in worker process."""

    job_uuid = UUID(job_id)
    logger.info("[stress-worker] executing job_id=%s", job_uuid)
    view = asyncio.run(_run_stress_job_once(job_uuid))
    logger.info(
        "[stress-worker] finished job_id=%s status=%s progress=%s",
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


def enqueue_stress_job(job_id: UUID | str) -> str:
    """Enqueue stress job and return task id."""

    result = execute_stress_job_task.apply_async(args=(str(job_id),), queue="stress")
    return str(result.id)
