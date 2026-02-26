"""Celery tasks for trade-approval execution and timeout expiration."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import Any
from uuid import UUID

from packages.shared_settings.schema.settings import settings
from packages.domain.trading.runtime.runtime_service import execute_manual_trade_action
from packages.infra.db import session as db_module
from packages.infra.db.models.manual_trade_action import ManualTradeAction
from packages.domain.trading.services.trade_approval_service import TradeApprovalService
from packages.infra.observability.logger import logger
from apps.worker.common.celery_base import celery_app


async def _run_execute_approved_open(request_id: UUID) -> dict[str, Any]:
    with suppress(Exception):
        await db_module.close_postgres()

    try:
        await db_module.init_postgres(ensure_schema=False)
        assert db_module.AsyncSessionLocal is not None
        async with db_module.AsyncSessionLocal() as session:
            service = TradeApprovalService(session)
            request = await service.mark_executing(request_id=request_id)
            if request is None:
                return {"request_id": str(request_id), "status": "missing"}

            if request.status != "executing":
                return {"request_id": str(request_id), "status": request.status}

            await session.commit()
            await session.refresh(request)

            action = ManualTradeAction(
                user_id=request.user_id,
                deployment_id=request.deployment_id,
                action="open",
                payload={
                    "symbol": request.symbol,
                    "side": request.side,
                    "qty": str(request.qty),
                    "mark_price": float(request.mark_price),
                    "source": "trade_approval",
                    "trade_approval_request_id": str(request.id),
                    "approval_key": request.approval_key,
                },
                status="accepted",
            )
            session.add(action)
            await session.flush()

            try:
                result = await execute_manual_trade_action(
                    session,
                    deployment_id=request.deployment_id,
                    action=action,
                )
            except Exception as exc:  # noqa: BLE001
                await service.mark_failed(request_id=request.id, error=f"execution_exception:{type(exc).__name__}")
                await session.commit()
                return {
                    "request_id": str(request.id),
                    "status": "failed",
                    "reason": f"execution_exception:{type(exc).__name__}",
                }

            if result.status == "completed":
                await service.mark_executed(
                    request_id=request.id,
                    order_id=result.order_id,
                )
                await session.commit()
                return {
                    "request_id": str(request.id),
                    "status": "executed",
                    "order_id": str(result.order_id) if result.order_id is not None else None,
                    "manual_action_id": str(action.id),
                }

            await service.mark_failed(request_id=request.id, error=result.reason)
            await session.commit()
            return {
                "request_id": str(request.id),
                "status": "failed",
                "reason": result.reason,
                "manual_action_id": str(action.id),
            }
    finally:
        with suppress(Exception):
            await db_module.close_postgres()


async def _run_expire_pending(*, limit: int) -> dict[str, Any]:
    with suppress(Exception):
        await db_module.close_postgres()

    try:
        await db_module.init_postgres(ensure_schema=False)
        assert db_module.AsyncSessionLocal is not None
        async with db_module.AsyncSessionLocal() as session:
            service = TradeApprovalService(session)
            expired = await service.expire_due(limit=limit)
            await session.commit()
            return {"status": "ok", "expired": expired}
    finally:
        with suppress(Exception):
            await db_module.close_postgres()


@celery_app.task(name="trade_approval.execute_approved_open")
def execute_approved_open_task(request_id: str) -> dict[str, Any]:
    """Execute one approved OPEN order asynchronously."""
    request_uuid = UUID(request_id)
    logger.info("[trade-approval-worker] execute approved open request_id=%s", request_uuid)
    return asyncio.run(_run_execute_approved_open(request_uuid))


def enqueue_execute_approved_open(request_id: UUID | str) -> str | None:
    """Enqueue execution task for one approved request."""
    try:
        result = execute_approved_open_task.apply_async(
            args=(str(request_id),),
            queue="trade_approval",
        )
    except Exception:  # noqa: BLE001
        logger.exception(
            "[trade-approval-worker] enqueue execute failed request_id=%s",
            request_id,
        )
        return None
    return str(result.id)


@celery_app.task(name="trade_approval.expire_pending")
def expire_pending_trade_approvals_task(limit: int | None = None) -> dict[str, Any]:
    """Mark timed-out pending approvals as expired."""
    if not settings.trading_approval_enabled:
        return {"status": "disabled", "expired": 0}
    safe_limit = max(1, int(limit or 200))
    return asyncio.run(_run_expire_pending(limit=safe_limit))
