"""Celery tasks for asynchronous IM notification delivery."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import Any
from uuid import uuid4

from src.config import settings
from src.models import database as db_module
from src.models.redis import get_sync_redis_client
from src.services.notification_outbox_service import NotificationOutboxService
from src.util.logger import logger
from src.workers.celery_app import celery_app

_DISPATCH_LOCK_KEY = "notifications:dispatch:lock"
_RELEASE_LOCK_SCRIPT = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('DEL', KEYS[1])
end
return 0
"""


def _acquire_dispatch_lock() -> tuple[str | None, bool]:
    """Acquire one short-lived Redis lock to dedupe dispatch workers."""
    token = uuid4().hex
    ttl_seconds = max(
        float(settings.notifications_dispatch_lock_ttl_seconds),
        float(settings.notifications_dispatch_max_runtime_seconds)
        + float(settings.notifications_delivery_timeout_seconds)
        + 2.0,
    )
    ttl_ms = max(1000, int(ttl_seconds * 1000))
    try:
        redis = get_sync_redis_client()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[notifications-worker] lock disabled because redis client is unavailable error=%s",
            type(exc).__name__,
        )
        return None, False

    try:
        acquired = redis.set(_DISPATCH_LOCK_KEY, token, nx=True, px=ttl_ms)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[notifications-worker] lock acquire failed, continuing without lock error=%s",
            type(exc).__name__,
        )
        return None, False
    if not acquired:
        return None, True
    return token, True


def _release_dispatch_lock(token: str) -> None:
    try:
        redis = get_sync_redis_client()
        redis.eval(_RELEASE_LOCK_SCRIPT, 1, _DISPATCH_LOCK_KEY, token)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[notifications-worker] lock release failed error=%s",
            type(exc).__name__,
        )


async def _run_dispatch_once(*, limit: int | None = None) -> dict[str, int]:
    with suppress(Exception):
        await db_module.close_postgres()

    try:
        # Worker tasks may run in parallel; skip DDL checks at task runtime.
        await db_module.init_postgres(ensure_schema=False)
        assert db_module.AsyncSessionLocal is not None
        async with db_module.AsyncSessionLocal() as session:
            service = NotificationOutboxService(session)
            stats = await service.dispatch_due(limit=limit)
            await session.commit()
            return stats
    finally:
        with suppress(Exception):
            await db_module.close_postgres()


@celery_app.task(name="notifications.dispatch_pending")
def dispatch_pending_notifications_task(limit: int | None = None) -> dict[str, Any]:
    """Fetch due outbox items and dispatch them to configured IM channels."""
    if not settings.notifications_enabled:
        return {"status": "disabled", "picked": 0, "sent": 0, "failed": 0, "dead": 0}

    lock_token, lock_enforced = _acquire_dispatch_lock()
    if lock_enforced and lock_token is None:
        return {
            "status": "skipped_lock",
            "picked": 0,
            "sent": 0,
            "failed": 0,
            "dead": 0,
            "deferred": 0,
        }

    logger.info("[notifications-worker] dispatching pending notifications")
    try:
        stats = asyncio.run(_run_dispatch_once(limit=limit))
    finally:
        if lock_token is not None:
            _release_dispatch_lock(lock_token)
    return {"status": "ok", **stats}


def enqueue_notification_dispatch(limit: int | None = None) -> str:
    """Enqueue one asynchronous notification dispatch run."""
    result = dispatch_pending_notifications_task.apply_async(kwargs={"limit": limit}, queue="notifications")
    return str(result.id)
