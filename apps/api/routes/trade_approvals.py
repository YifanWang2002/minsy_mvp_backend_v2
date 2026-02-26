"""Trade approval list/decision endpoints."""

from __future__ import annotations

from decimal import Decimal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.middleware.auth import get_current_user
from apps.api.schemas.events import TradeApprovalRequestResponse
from apps.api.schemas.requests import TradeApprovalDecisionRequest
from apps.api.dependencies import get_db
from packages.infra.db.models.trade_approval_request import TradeApprovalRequest
from packages.infra.db.models.user import User
from packages.domain.trading.services.trade_approval_service import TradeApprovalService
from apps.api.services.trading_queue_service import enqueue_execute_approved_open

router = APIRouter(prefix="/trade-approvals", tags=["trade_approvals"])

_ALLOWED_STATUSES: frozenset[str] = frozenset(
    {
        "pending",
        "approved",
        "rejected",
        "expired",
        "executing",
        "executed",
        "failed",
        "cancelled",
    }
)


def _to_float(value: Decimal) -> float:
    return float(value)


def _serialize_trade_approval(row: TradeApprovalRequest) -> TradeApprovalRequestResponse:
    return TradeApprovalRequestResponse(
        trade_approval_request_id=row.id,
        user_id=row.user_id,
        deployment_id=row.deployment_id,
        execution_order_id=row.execution_order_id,
        signal=row.signal,
        side=row.side,
        symbol=row.symbol,
        qty=_to_float(row.qty),
        mark_price=_to_float(row.mark_price),
        reason=row.reason,
        timeframe=row.timeframe,
        bar_time=row.bar_time,
        approval_channel=row.approval_channel,
        status=row.status,
        approval_key=row.approval_key,
        requested_at=row.requested_at,
        expires_at=row.expires_at,
        approved_at=row.approved_at,
        rejected_at=row.rejected_at,
        expired_at=row.expired_at,
        executed_at=row.executed_at,
        approved_via=row.approved_via,
        decision_actor=row.decision_actor,
        execution_error=row.execution_error,
        intent_payload=row.intent_payload if isinstance(row.intent_payload, dict) else {},
        metadata=row.metadata_ if isinstance(row.metadata_, dict) else {},
    )


def _parse_status_filter(raw_status: str | None) -> set[str] | None:
    if raw_status is None:
        return None
    statuses = {item.strip().lower() for item in raw_status.split(",") if item.strip()}
    if not statuses:
        return None
    unknown = statuses - _ALLOWED_STATUSES
    if unknown:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "TRADE_APPROVAL_STATUS_INVALID",
                "message": f"Unsupported status filter: {sorted(unknown)}",
            },
        )
    return statuses


@router.get("", response_model=list[TradeApprovalRequestResponse])
async def list_trade_approvals(
    status_filter: str | None = Query(default=None, alias="status"),
    deployment_id: UUID | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[TradeApprovalRequestResponse]:
    service = TradeApprovalService(db)
    rows = await service.list_for_user(
        user_id=user.id,
        statuses=_parse_status_filter(status_filter),
        deployment_id=deployment_id,
        limit=limit,
    )
    await db.commit()
    return [_serialize_trade_approval(row) for row in rows]


@router.post("/{request_id}/approve", response_model=TradeApprovalRequestResponse)
async def approve_trade_approval(
    request_id: UUID,
    payload: TradeApprovalDecisionRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> TradeApprovalRequestResponse:
    service = TradeApprovalService(db)
    row = await service.approve(
        request_id=request_id,
        user_id=user.id,
        via="api",
        actor=str(user.id),
        note=payload.note,
    )
    if row is None:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "TRADE_APPROVAL_NOT_FOUND", "message": "Approval request not found."},
        )

    metadata = row.metadata_ if isinstance(row.metadata_, dict) else {}
    if row.status == "approved" and not metadata.get("execution_task_id"):
        task_id = enqueue_execute_approved_open(row.id)
        if task_id:
            await service.append_execution_task_id(request_id=row.id, task_id=task_id)
    await db.commit()
    return _serialize_trade_approval(row)


@router.post("/{request_id}/reject", response_model=TradeApprovalRequestResponse)
async def reject_trade_approval(
    request_id: UUID,
    payload: TradeApprovalDecisionRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> TradeApprovalRequestResponse:
    service = TradeApprovalService(db)
    row = await service.reject(
        request_id=request_id,
        user_id=user.id,
        via="api",
        actor=str(user.id),
        note=payload.note,
    )
    if row is None:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "TRADE_APPROVAL_NOT_FOUND", "message": "Approval request not found."},
        )
    await db.commit()
    return _serialize_trade_approval(row)
