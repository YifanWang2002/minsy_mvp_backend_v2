"""Deployment lifecycle and trading runtime endpoints."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from math import ceil
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from apps.api.dependencies import get_db
from apps.api.middleware.auth import get_current_user
from apps.api.schemas.events import (
    DeploymentActionResponse,
    DeploymentResponse,
    DeploymentRunResponse,
    DeploymentSignalProcessResponse,
    ManualTradeActionResponse,
    OrderHistoryPageResponse,
    OrderResponse,
    PnlSnapshotResponse,
    PositionResponse,
    SignalResponse,
)
from apps.api.schemas.requests import DeploymentCreateRequest, ManualTradeActionRequest
from apps.api.services.trading_queue_service import (
    enqueue_execute_manual_trade_action,
    enqueue_paper_trading_runtime,
)
from packages.domain.exceptions import DomainError
from packages.domain.trading import deployment_ops as deployment_ops_domain
from packages.domain.trading.runtime.runtime_service import (
    execute_manual_trade_action,
    process_deployment_signal_cycle,
    reconcile_manual_action_terminal_state,
)
from packages.domain.trading.services.trading_event_outbox_service import (
    append_trading_event_snapshot,
)
from packages.infra.db.models.deployment import Deployment
from packages.infra.db.models.deployment_run import DeploymentRun
from packages.infra.db.models.manual_trade_action import ManualTradeAction
from packages.infra.db.models.order import Order
from packages.infra.db.models.pnl_snapshot import PnlSnapshot
from packages.infra.db.models.position import Position
from packages.infra.db.models.signal_event import SignalEvent
from packages.infra.db.models.user import User
from packages.shared_settings.schema.settings import settings

router = APIRouter(prefix="/deployments", tags=["deployments"])


def _deployment_query_options() -> tuple[object, ...]:
    return (
        selectinload(Deployment.deployment_runs).selectinload(
            DeploymentRun.broker_account
        ),
        selectinload(Deployment.strategy),
    )


def _decimal_to_float(value: Decimal | None) -> float | None:
    if value is None:
        return None
    return float(value)


def _as_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
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
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _serialize_run(run: DeploymentRun | None) -> DeploymentRunResponse | None:
    if run is None:
        return None
    state = run.runtime_state if isinstance(run.runtime_state, dict) else {}
    scheduler = (
        state.get("scheduler") if isinstance(state.get("scheduler"), dict) else {}
    )
    return DeploymentRunResponse(
        deployment_run_id=run.id,
        deployment_id=run.deployment_id,
        strategy_id=run.strategy_id,
        broker_account_id=run.broker_account_id,
        status=run.status,
        last_bar_time=run.last_bar_time,
        timeframe_seconds=_as_int(scheduler.get("timeframe_seconds")),
        last_trigger_bucket=_as_int(scheduler.get("last_trigger_bucket")),
        last_enqueued_at=_as_datetime(scheduler.get("last_enqueued_at")),
        runtime_state=state,
        created_at=run.created_at,
        updated_at=run.updated_at,
    )


def _latest_run(deployment: Deployment) -> DeploymentRun | None:
    if not deployment.deployment_runs:
        return None
    return sorted(
        deployment.deployment_runs,
        key=lambda item: item.created_at,
        reverse=True,
    )[0]


def _serialize_deployment(deployment: Deployment) -> DeploymentResponse:
    run = _latest_run(deployment)
    broker_provider = None
    if run is not None and run.broker_account is not None:
        provider = str(run.broker_account.provider).strip().lower()
        broker_provider = provider or None
    strategy_name = None
    if deployment.strategy is not None:
        raw_strategy_name = str(deployment.strategy.name).strip()
        strategy_name = raw_strategy_name or None
    strategy_payload = (
        deployment.strategy.dsl_payload
        if deployment.strategy is not None
        and isinstance(deployment.strategy.dsl_payload, dict)
        else {}
    )
    universe = (
        strategy_payload.get("universe", {})
        if isinstance(strategy_payload.get("universe"), dict)
        else {}
    )
    symbols = (
        universe.get("tickers") if isinstance(universe.get("tickers"), list) else []
    )
    market = universe.get("market") if isinstance(universe.get("market"), str) else None
    timeframe = (
        strategy_payload.get("timeframe")
        if isinstance(strategy_payload.get("timeframe"), str)
        else None
    )
    return DeploymentResponse(
        deployment_id=deployment.id,
        strategy_id=deployment.strategy_id,
        strategy_name=strategy_name,
        user_id=deployment.user_id,
        broker_provider=broker_provider,
        mode=deployment.mode,
        status=deployment.status,
        market=market,
        symbols=[str(symbol) for symbol in symbols],
        timeframe=timeframe,
        capital_allocated=float(deployment.capital_allocated),
        risk_limits=deployment.risk_limits
        if isinstance(deployment.risk_limits, dict)
        else {},
        deployed_at=deployment.deployed_at,
        stopped_at=deployment.stopped_at,
        created_at=deployment.created_at,
        updated_at=deployment.updated_at,
        run=_serialize_run(run),
    )


def _serialize_order(order: Order) -> OrderResponse:
    metadata = order.metadata_ if isinstance(order.metadata_, dict) else {}
    provider_status = metadata.get("provider_status")
    if not isinstance(provider_status, str) or not provider_status.strip():
        provider_status = order.status
    display_price = _decimal_to_float(order.price)
    if display_price is None:
        submitted_mark_price = metadata.get("submitted_mark_price")
        try:
            display_price = (
                float(submitted_mark_price)
                if submitted_mark_price is not None
                else None
            )
        except (TypeError, ValueError):
            display_price = None
    return OrderResponse(
        order_id=order.id,
        deployment_id=order.deployment_id,
        provider_order_id=order.provider_order_id,
        client_order_id=order.client_order_id,
        symbol=order.symbol,
        side=order.side,
        type=order.type,
        qty=float(order.qty),
        price=display_price,
        status=order.status,
        provider_status=str(provider_status),
        reject_reason=order.reject_reason,
        last_sync_at=order.last_sync_at,
        submitted_at=order.submitted_at,
        metadata=metadata,
        created_at=order.created_at,
        updated_at=order.updated_at,
    )


def _serialize_position(position: Position) -> PositionResponse:
    return PositionResponse(
        position_id=position.id,
        deployment_id=position.deployment_id,
        symbol=position.symbol,
        side=position.side,
        qty=float(position.qty),
        avg_entry_price=float(position.avg_entry_price),
        mark_price=float(position.mark_price),
        unrealized_pnl=float(position.unrealized_pnl),
        realized_pnl=float(position.realized_pnl),
        created_at=position.created_at,
        updated_at=position.updated_at,
    )


def _serialize_pnl_snapshot(snapshot: PnlSnapshot) -> PnlSnapshotResponse:
    return PnlSnapshotResponse(
        pnl_snapshot_id=snapshot.id,
        deployment_id=snapshot.deployment_id,
        source="platform_estimate",
        equity=float(snapshot.equity),
        cash=float(snapshot.cash),
        margin_used=float(snapshot.margin_used),
        unrealized_pnl=float(snapshot.unrealized_pnl),
        realized_pnl=float(snapshot.realized_pnl),
        snapshot_time=snapshot.snapshot_time,
        created_at=snapshot.created_at,
        updated_at=snapshot.updated_at,
    )


def _serialize_manual_action(action: ManualTradeAction) -> ManualTradeActionResponse:
    return ManualTradeActionResponse(
        manual_trade_action_id=action.id,
        user_id=action.user_id,
        deployment_id=action.deployment_id,
        action=action.action,
        payload=action.payload if isinstance(action.payload, dict) else {},
        status=action.status,
        created_at=action.created_at,
        updated_at=action.updated_at,
    )


def _serialize_signal(record: SignalEvent) -> SignalResponse:
    return SignalResponse(
        signal_event_id=record.id,
        deployment_id=record.deployment_id,
        signal=record.signal,
        symbol=record.symbol,
        timeframe=record.timeframe,
        bar_time=record.bar_time,
        reason=record.reason,
        metadata=record.metadata_ if isinstance(record.metadata_, dict) else {},
    )


async def _load_owned_deployment(
    db: AsyncSession,
    *,
    deployment_id: UUID,
    user_id: UUID,
) -> Deployment:
    deployment = await db.scalar(
        select(Deployment)
        .options(*_deployment_query_options())
        .where(
            Deployment.id == deployment_id,
            Deployment.user_id == user_id,
        )
    )
    if deployment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "DEPLOYMENT_NOT_FOUND",
                "message": "Deployment not found.",
            },
        )
    return deployment


@router.post("", response_model=DeploymentResponse, status_code=status.HTTP_201_CREATED)
async def create_deployment(
    payload: DeploymentCreateRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> DeploymentResponse:
    try:
        deployment = await deployment_ops_domain.create_deployment(
            db,
            strategy_id=payload.strategy_id,
            broker_account_id=payload.broker_account_id,
            user_id=user.id,
            mode=payload.mode,
            capital_allocated=payload.capital_allocated,
            risk_limits=payload.risk_limits,
            runtime_state=payload.runtime_state,
        )
    except DomainError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    await append_trading_event_snapshot(db, deployment_id=deployment.id)
    return _serialize_deployment(deployment)


@router.get("", response_model=list[DeploymentResponse])
async def list_deployments(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[DeploymentResponse]:
    rows = (
        await db.scalars(
            select(Deployment)
            .options(*_deployment_query_options())
            .where(Deployment.user_id == user.id)
            .order_by(Deployment.created_at.desc()),
        )
    ).all()
    return [_serialize_deployment(row) for row in rows]


@router.get("/{deployment_id}", response_model=DeploymentResponse)
async def get_deployment(
    deployment_id: UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> DeploymentResponse:
    deployment = await _load_owned_deployment(
        db,
        deployment_id=deployment_id,
        user_id=user.id,
    )
    return _serialize_deployment(deployment)


async def _apply_status_transition(
    db: AsyncSession,
    *,
    deployment: Deployment,
    target_status: str,
) -> Deployment:
    try:
        transitioned = await deployment_ops_domain.apply_status_transition(
            db,
            deployment=deployment,
            target_status=target_status,
        )
    except DomainError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    await append_trading_event_snapshot(db, deployment_id=transitioned.id)
    return transitioned


@router.post("/{deployment_id}/start", response_model=DeploymentActionResponse)
async def start_deployment(
    deployment_id: UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> DeploymentActionResponse:
    deployment = await _load_owned_deployment(
        db, deployment_id=deployment_id, user_id=user.id
    )
    deployment = await _apply_status_transition(
        db, deployment=deployment, target_status="active"
    )

    task_id: str | None = None
    if settings.paper_trading_enqueue_on_start:
        task_id = enqueue_paper_trading_runtime(deployment.id)
    return DeploymentActionResponse(
        deployment=_serialize_deployment(deployment),
        queued_task_id=task_id,
    )


@router.post("/{deployment_id}/pause", response_model=DeploymentActionResponse)
async def pause_deployment(
    deployment_id: UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> DeploymentActionResponse:
    deployment = await _load_owned_deployment(
        db, deployment_id=deployment_id, user_id=user.id
    )
    deployment = await _apply_status_transition(
        db, deployment=deployment, target_status="paused"
    )
    return DeploymentActionResponse(deployment=_serialize_deployment(deployment))


@router.post("/{deployment_id}/stop", response_model=DeploymentActionResponse)
async def stop_deployment(
    deployment_id: UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> DeploymentActionResponse:
    deployment = await _load_owned_deployment(
        db, deployment_id=deployment_id, user_id=user.id
    )
    deployment = await _apply_status_transition(
        db, deployment=deployment, target_status="stopped"
    )
    return DeploymentActionResponse(deployment=_serialize_deployment(deployment))


@router.get("/{deployment_id}/orders", response_model=list[OrderResponse])
async def list_deployment_orders(
    deployment_id: UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[OrderResponse]:
    await _load_owned_deployment(db, deployment_id=deployment_id, user_id=user.id)
    rows = (
        await db.scalars(
            select(Order)
            .where(Order.deployment_id == deployment_id)
            .order_by(Order.submitted_at.desc(), Order.id.desc()),
        )
    ).all()
    return [_serialize_order(row) for row in rows]


@router.get(
    "/{deployment_id}/orders/history",
    response_model=OrderHistoryPageResponse,
)
async def list_deployment_orders_history(
    deployment_id: UUID,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=10, ge=1, le=100),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> OrderHistoryPageResponse:
    await _load_owned_deployment(db, deployment_id=deployment_id, user_id=user.id)
    total = (
        await db.scalar(
            select(func.count(Order.id)).where(Order.deployment_id == deployment_id)
        )
    ) or 0
    total_count = int(total)
    total_pages = max(1, ceil(total_count / page_size)) if total_count > 0 else 1
    effective_page = min(page, total_pages)
    offset = (effective_page - 1) * page_size
    rows = (
        await db.scalars(
            select(Order)
            .where(Order.deployment_id == deployment_id)
            .order_by(Order.submitted_at.desc(), Order.id.desc())
            .offset(offset)
            .limit(page_size)
        )
    ).all()
    return OrderHistoryPageResponse(
        items=[_serialize_order(row) for row in rows],
        page=effective_page,
        page_size=page_size,
        total=total_count,
        total_pages=total_pages,
    )


@router.get("/{deployment_id}/positions", response_model=list[PositionResponse])
async def list_deployment_positions(
    deployment_id: UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[PositionResponse]:
    await _load_owned_deployment(db, deployment_id=deployment_id, user_id=user.id)
    rows = (
        await db.scalars(
            select(Position)
            .where(Position.deployment_id == deployment_id)
            .order_by(Position.updated_at.desc()),
        )
    ).all()
    return [_serialize_position(row) for row in rows]


@router.get("/{deployment_id}/pnl", response_model=list[PnlSnapshotResponse])
async def list_deployment_pnl(
    deployment_id: UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[PnlSnapshotResponse]:
    await _load_owned_deployment(db, deployment_id=deployment_id, user_id=user.id)
    rows = (
        await db.scalars(
            select(PnlSnapshot)
            .where(PnlSnapshot.deployment_id == deployment_id)
            .order_by(PnlSnapshot.snapshot_time.desc())
            .limit(500),
        )
    ).all()
    return [_serialize_pnl_snapshot(row) for row in rows]


@router.post("/{deployment_id}/manual-action", response_model=ManualTradeActionResponse)
@router.post(
    "/{deployment_id}/manual-actions", response_model=ManualTradeActionResponse
)
async def create_manual_action(
    deployment_id: UUID,
    payload: ManualTradeActionRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ManualTradeActionResponse:
    deployment = await _load_owned_deployment(
        db, deployment_id=deployment_id, user_id=user.id
    )
    action = ManualTradeAction(
        user_id=user.id,
        deployment_id=deployment_id,
        action=payload.action,
        payload=payload.payload,
        status="pending",
    )
    db.add(action)
    await db.commit()
    await db.refresh(action)
    await append_trading_event_snapshot(db, deployment_id=deployment_id)
    action_name = str(payload.action).strip().lower()
    requires_active_deployment = action_name not in {"stop", "close", "reduce"}
    if requires_active_deployment and deployment.status != "active":
        action.status = "rejected"
        action.payload = {
            **(action.payload if isinstance(action.payload, dict) else {}),
            "_execution": {
                "status": "rejected",
                "reason": f"deployment_{deployment.status}",
            },
        }
        await db.commit()
        await db.refresh(action)
        await append_trading_event_snapshot(db, deployment_id=deployment_id)
        return _serialize_manual_action(action)
    task_id = enqueue_execute_manual_trade_action(action.id)
    if task_id is not None:
        await db.refresh(action)
        if action.status == "pending":
            payload_dict = action.payload if isinstance(action.payload, dict) else {}
            execution = (
                payload_dict.get("_execution")
                if isinstance(payload_dict.get("_execution"), dict)
                else {}
            )
            action.status = "executing"
            action.payload = {
                **payload_dict,
                "_execution": {
                    **execution,
                    "status": "executing",
                    "reason": "queued",
                    "task_id": task_id,
                },
            }
            await db.commit()
            await db.refresh(action)
        await append_trading_event_snapshot(db, deployment_id=deployment_id)
        return _serialize_manual_action(action)
    result = await execute_manual_trade_action(
        db,
        deployment_id=deployment_id,
        action=action,
    )
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
        if retry_task_id is not None:
            payload_dict = action.payload if isinstance(action.payload, dict) else {}
            execution = (
                payload_dict.get("_execution")
                if isinstance(payload_dict.get("_execution"), dict)
                else {}
            )
            action.status = "executing"
            action.payload = {
                **payload_dict,
                "_execution": {
                    **execution,
                    "status": "executing",
                    "reason": "waiting_for_runtime_lock",
                    "task_id": retry_task_id,
                },
            }
            await db.commit()
            await append_trading_event_snapshot(db, deployment_id=deployment_id)
        else:
            payload_dict = action.payload if isinstance(action.payload, dict) else {}
            execution = (
                payload_dict.get("_execution")
                if isinstance(payload_dict.get("_execution"), dict)
                else {}
            )
            action.status = "failed"
            action.payload = {
                **payload_dict,
                "_execution": {
                    **execution,
                    "status": "failed",
                    "reason": "lock_retry_enqueue_failed",
                },
            }
            await db.commit()
            await append_trading_event_snapshot(db, deployment_id=deployment_id)
    await db.refresh(action)
    await append_trading_event_snapshot(db, deployment_id=deployment_id)
    return _serialize_manual_action(action)


@router.get(
    "/{deployment_id}/manual-actions/latest",
    response_model=ManualTradeActionResponse,
)
async def get_latest_manual_action(
    deployment_id: UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ManualTradeActionResponse:
    await _load_owned_deployment(db, deployment_id=deployment_id, user_id=user.id)
    action = await db.scalar(
        select(ManualTradeAction)
        .where(
            ManualTradeAction.deployment_id == deployment_id,
            ManualTradeAction.user_id == user.id,
        )
        .order_by(ManualTradeAction.created_at.desc(), ManualTradeAction.id.desc())
        .limit(1)
    )
    if action is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "MANUAL_ACTION_NOT_FOUND",
                "message": "No manual action found for deployment.",
            },
        )
    await reconcile_manual_action_terminal_state(db, action=action)
    return _serialize_manual_action(action)


@router.get("/{deployment_id}/signals", response_model=list[SignalResponse])
async def list_deployment_signals(
    deployment_id: UUID,
    cursor: str | None = None,
    limit: int = 100,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[SignalResponse]:
    await _load_owned_deployment(db, deployment_id=deployment_id, user_id=user.id)
    safe_limit = min(max(limit, 1), 500)

    stmt = (
        select(SignalEvent)
        .where(SignalEvent.deployment_id == deployment_id)
        .order_by(SignalEvent.created_at.desc(), SignalEvent.id.desc())
        .limit(safe_limit)
    )

    if cursor:
        try:
            cursor_id = UUID(cursor)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "code": "SIGNAL_CURSOR_INVALID",
                    "message": "Cursor must be a valid UUID.",
                },
            ) from exc

        cursor_row = await db.scalar(
            select(SignalEvent).where(
                SignalEvent.id == cursor_id,
                SignalEvent.deployment_id == deployment_id,
            )
        )
        if cursor_row is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "code": "SIGNAL_CURSOR_NOT_FOUND",
                    "message": "Cursor event not found.",
                },
            )
        stmt = stmt.where(
            or_(
                SignalEvent.created_at < cursor_row.created_at,
                and_(
                    SignalEvent.created_at == cursor_row.created_at,
                    SignalEvent.id != cursor_row.id,
                ),
            )
        )

    rows = list((await db.scalars(stmt)).all())
    return [_serialize_signal(record) for record in rows]


@router.post(
    "/{deployment_id}/process-now", response_model=DeploymentSignalProcessResponse
)
async def process_deployment_now(
    deployment_id: UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> DeploymentSignalProcessResponse:
    await _load_owned_deployment(db, deployment_id=deployment_id, user_id=user.id)
    result = await process_deployment_signal_cycle(db, deployment_id=deployment_id)
    return DeploymentSignalProcessResponse(
        deployment_id=result.deployment_id,
        execution_event_id=result.execution_event_id,
        signal=result.signal,
        reason=result.reason,
        order_id=result.order_id,
        idempotent_hit=result.idempotent_hit,
        bar_time=result.bar_time,
        metadata=result.metadata,
    )
