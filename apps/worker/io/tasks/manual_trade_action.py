"""Celery tasks for asynchronous manual trade execution."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import Any
from uuid import UUID

from sqlalchemy import select

from apps.worker.common.celery_base import celery_app
from packages.domain.trading.runtime.runtime_service import execute_manual_trade_action
from packages.infra.db import session as db_module
from packages.infra.db.models.manual_trade_action import ManualTradeAction
from packages.infra.observability.logger import logger


async def _run_execute_manual_trade_action(action_id: UUID) -> dict[str, Any]:
    with suppress(Exception):
        await db_module.close_postgres()

    try:
        await db_module.init_postgres(ensure_schema=False)
        assert db_module.AsyncSessionLocal is not None
        async with db_module.AsyncSessionLocal() as session:
            action = await session.scalar(
                select(ManualTradeAction).where(ManualTradeAction.id == action_id)
            )
            if action is None:
                return {"action_id": str(action_id), "status": "missing"}

            if action.status in {"accepted", "completed", "rejected", "failed"}:
                return {
                    "action_id": str(action.id),
                    "deployment_id": str(action.deployment_id),
                    "status": action.status,
                }

            if action.status == "pending":
                payload = action.payload if isinstance(action.payload, dict) else {}
                execution = (
                    payload.get("_execution")
                    if isinstance(payload.get("_execution"), dict)
                    else {}
                )
                action.status = "executing"
                action.payload = {
                    **payload,
                    "_execution": {
                        **execution,
                        "status": "executing",
                        "reason": execution.get("reason") or "worker_started",
                    },
                }
                await session.commit()
                await session.refresh(action)

            try:
                result = await execute_manual_trade_action(
                    session,
                    deployment_id=action.deployment_id,
                    action=action,
                )
            except Exception as exc:  # noqa: BLE001
                payload = action.payload if isinstance(action.payload, dict) else {}
                execution = (
                    payload.get("_execution")
                    if isinstance(payload.get("_execution"), dict)
                    else {}
                )
                action.status = "failed"
                action.payload = {
                    **payload,
                    "_execution": {
                        **execution,
                        "status": "failed",
                        "reason": f"execution_exception:{type(exc).__name__}",
                    },
                }
                await session.commit()
                await session.refresh(action)
                return {
                    "action_id": str(action.id),
                    "deployment_id": str(action.deployment_id),
                    "status": "failed",
                    "reason": f"execution_exception:{type(exc).__name__}",
                }

            if result.status == "deferred" and result.reason == "deployment_locked":
                retry_after_seconds_raw = result.metadata.get("retry_after_seconds")
                try:
                    retry_after_seconds = max(1, int(retry_after_seconds_raw))
                except (TypeError, ValueError):
                    retry_after_seconds = 1
                retry_task_id = enqueue_execute_manual_trade_action(
                    action.id,
                    countdown_seconds=retry_after_seconds,
                )
                if retry_task_id is None:
                    payload = action.payload if isinstance(action.payload, dict) else {}
                    execution = (
                        payload.get("_execution")
                        if isinstance(payload.get("_execution"), dict)
                        else {}
                    )
                    action.status = "failed"
                    action.payload = {
                        **payload,
                        "_execution": {
                            **execution,
                            "status": "failed",
                            "reason": "lock_retry_enqueue_failed",
                        },
                    }
                    await session.commit()
                    await session.refresh(action)
                    return {
                        "action_id": str(action.id),
                        "deployment_id": str(action.deployment_id),
                        "status": "failed",
                        "reason": "lock_retry_enqueue_failed",
                    }
                payload = action.payload if isinstance(action.payload, dict) else {}
                execution = (
                    payload.get("_execution")
                    if isinstance(payload.get("_execution"), dict)
                    else {}
                )
                action.payload = {
                    **payload,
                    "_execution": {
                        **execution,
                        "status": "executing",
                        "reason": "waiting_for_runtime_lock",
                        "task_id": retry_task_id,
                    },
                }
                await session.commit()
                await session.refresh(action)
                return {
                    "action_id": str(action.id),
                    "deployment_id": str(action.deployment_id),
                    "status": "deferred",
                    "reason": "deployment_locked",
                    "retry_task_id": retry_task_id,
                    "retry_after_seconds": retry_after_seconds,
                }

            return {
                "action_id": str(action.id),
                "deployment_id": str(action.deployment_id),
                "status": result.status,
                "reason": result.reason,
                "order_id": str(result.order_id)
                if result.order_id is not None
                else None,
            }
    finally:
        with suppress(Exception):
            await db_module.close_postgres()


@celery_app.task(name="paper_trading.execute_manual_trade_action")
def execute_manual_trade_action_task(action_id: str) -> dict[str, Any]:
    """Execute one manual trade action asynchronously."""
    action_uuid = UUID(action_id)
    logger.info("[manual-action-worker] execute action_id=%s", action_uuid)
    return asyncio.run(_run_execute_manual_trade_action(action_uuid))


def enqueue_execute_manual_trade_action(
    action_id: UUID | str,
    *,
    countdown_seconds: int | None = None,
) -> str | None:
    """Enqueue one manual trade action for async execution."""
    try:
        kwargs: dict[str, Any] = {}
        if countdown_seconds is not None and countdown_seconds > 0:
            kwargs["countdown"] = countdown_seconds
        result = execute_manual_trade_action_task.apply_async(
            args=(str(action_id),),
            queue="paper_trading",
            **kwargs,
        )
    except Exception:  # noqa: BLE001
        logger.exception(
            "[manual-action-worker] enqueue execute failed action_id=%s",
            action_id,
        )
        return None
    return str(result.id)
