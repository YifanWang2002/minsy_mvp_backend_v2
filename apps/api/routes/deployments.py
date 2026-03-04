"""Deployment lifecycle and trading runtime endpoints."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from apps.api.agents.phases import Phase, can_transition
from apps.api.dependencies import get_db
from apps.api.middleware.auth import get_current_user
from apps.api.schemas.events import (
    DeploymentActionResponse,
    DeploymentResponse,
    DeploymentRunResponse,
    DeploymentSignalProcessResponse,
    ManualTradeActionResponse,
    OrderResponse,
    PnlSnapshotResponse,
    PositionResponse,
    SignalResponse,
)
from apps.api.schemas.requests import DeploymentCreateRequest, ManualTradeActionRequest
from apps.worker.io.tasks.manual_trade_action import enqueue_execute_manual_trade_action
from apps.api.services.trading_queue_service import enqueue_paper_trading_runtime
from packages.core.events.notification_events import EVENT_DEPLOYMENT_STARTED
from packages.domain.notification.services.notification_outbox_service import (
    NotificationOutboxService,
)
from packages.domain.trading import deployment_ops as deployment_ops_domain
from packages.domain.trading.runtime.runtime_service import (
    execute_manual_trade_action,
    process_deployment_signal_cycle,
    reconcile_manual_action_terminal_state,
)
from packages.domain.trading.runtime.timeframe_scheduler import timeframe_to_seconds
from packages.domain.trading.services.broker_provider_service import (
    BrokerProviderService,
)
from packages.domain.trading.services.trading_event_outbox_service import (
    append_trading_event_snapshot,
)
from packages.infra.db.models.broker_account import BrokerAccount
from packages.infra.db.models.deployment import Deployment
from packages.infra.db.models.deployment_run import DeploymentRun
from packages.infra.db.models.manual_trade_action import ManualTradeAction
from packages.infra.db.models.order import Order
from packages.infra.db.models.phase_transition import PhaseTransition
from packages.infra.db.models.pnl_snapshot import PnlSnapshot
from packages.infra.db.models.position import Position
from packages.infra.db.models.session import Session
from packages.infra.db.models.signal_event import SignalEvent
from packages.infra.db.models.strategy import Strategy
from packages.infra.db.models.user import User
from packages.infra.observability.logger import logger
from packages.infra.redis.stores.runtime_state_store import runtime_state_store
from packages.shared_settings.schema.settings import settings

router = APIRouter(prefix="/deployments", tags=["deployments"])

_DEPLOYABLE_STRATEGY_STATUSES = {"validated", "backtested", "deployed"}
_broker_provider_service = BrokerProviderService()


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


def _positive_decimal_or_none(value: Any) -> Decimal | None:
    try:
        parsed = Decimal(str(value))
    except Exception:  # noqa: BLE001
        return None
    if parsed <= 0:
        return None
    return parsed.quantize(Decimal("0.01"))


def _resolve_capital_from_validation_metadata(account: BrokerAccount) -> Decimal | None:
    metadata = (
        account.validation_metadata
        if isinstance(account.validation_metadata, dict)
        else {}
    )
    for key in (
        "paper_equity",
        "paper_cash",
        "paper_buying_power",
        "equity",
        "cash",
        "buying_power",
    ):
        capital = _positive_decimal_or_none(metadata.get(key))
        if capital is not None:
            return capital
    return None


def _resolve_capital_from_account_metadata(account: BrokerAccount) -> Decimal | None:
    metadata = account.metadata_ if isinstance(account.metadata_, dict) else {}
    for key in ("starting_cash", "capital_allocated", "equity", "cash", "buying_power"):
        capital = _positive_decimal_or_none(metadata.get(key))
        if capital is not None:
            return capital
    return None


async def _resolve_deployment_capital_allocated(
    *,
    requested_capital: Decimal,
    account: BrokerAccount,
) -> Decimal:
    if requested_capital > 0:
        return requested_capital.quantize(Decimal("0.01"))

    from_probe = _resolve_capital_from_validation_metadata(account)
    if from_probe is not None:
        return from_probe

    from_account_metadata = _resolve_capital_from_account_metadata(account)
    if from_account_metadata is not None:
        return from_account_metadata

    binding = _broker_provider_service.build_adapter_binding_from_account(account)
    adapter = binding.adapter
    if adapter is None:
        return Decimal("10000.00")

    try:
        state = await adapter.fetch_account_state()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[deployments] failed to fetch account state broker_account_id=%s provider=%s error=%s",
            account.id,
            binding.provider,
            type(exc).__name__,
        )
        return Decimal("10000.00")
    finally:
        await adapter.aclose()

    for candidate in (state.equity, state.cash, state.buying_power):
        capital = _positive_decimal_or_none(candidate)
        if capital is not None:
            return capital
    return Decimal("10000.00")


async def _resolve_broker_provider_for_run(
    db: AsyncSession, *, run: DeploymentRun
) -> str:
    account = await db.scalar(
        select(BrokerAccount).where(BrokerAccount.id == run.broker_account_id)
    )
    if account is None:
        return "unknown"
    provider = str(account.provider).strip().lower()
    return provider or "unknown"


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


def _build_phase_transition_metadata(
    *,
    reason: str,
    source: str,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "reason": reason,
        "source": source,
        "context": context or {},
        "recorded_at": datetime.now(UTC).isoformat(),
    }


def _map_deployment_status_to_phase_status(target_status: str) -> str:
    if target_status == "active":
        return "deployed"
    if target_status in {"paused", "stopped"}:
        return "ready"
    return "ready"


async def _sync_session_deployment_phase_state(
    db: AsyncSession,
    *,
    deployment: Deployment,
    target_status: str,
    at: datetime,
) -> None:
    if deployment.strategy is None:
        return

    session = await db.scalar(
        select(Session).where(
            Session.id == deployment.strategy.session_id,
            Session.user_id == deployment.user_id,
        )
    )
    if session is None:
        return

    phase_status = _map_deployment_status_to_phase_status(target_status)
    artifacts = dict(session.artifacts or {})
    deployment_block_raw = artifacts.get(Phase.DEPLOYMENT.value)
    deployment_block = (
        dict(deployment_block_raw)
        if isinstance(deployment_block_raw, dict)
        else {"profile": {}, "missing_fields": ["deployment_status"], "runtime": {}}
    )
    profile_raw = deployment_block.get("profile")
    profile = dict(profile_raw) if isinstance(profile_raw, dict) else {}
    profile.update(
        {
            "strategy_id": str(deployment.strategy_id),
            "deployment_id": str(deployment.id),
            "deployment_mode": deployment.mode,
            "deployment_status": phase_status,
            "deployment_last_action": target_status,
            "deployment_last_action_at": at.isoformat(),
        }
    )
    deployment_block["profile"] = profile
    deployment_block["missing_fields"] = []
    runtime_raw = deployment_block.get("runtime")
    runtime = dict(runtime_raw) if isinstance(runtime_raw, dict) else {}
    runtime.update(
        {
            "deployment_id": str(deployment.id),
            "status": deployment.status,
            "phase_status": phase_status,
            "last_action": target_status,
            "updated_at": at.isoformat(),
        }
    )
    deployment_block["runtime"] = runtime
    artifacts[Phase.DEPLOYMENT.value] = deployment_block
    session.artifacts = artifacts
    session.last_activity_at = at

    if session.current_phase != Phase.DEPLOYMENT.value and can_transition(
        session.current_phase,
        Phase.DEPLOYMENT.value,
    ):
        from_phase = session.current_phase
        session.current_phase = Phase.DEPLOYMENT.value
        session.previous_response_id = None
        db.add(
            PhaseTransition(
                session_id=session.id,
                from_phase=from_phase,
                to_phase=Phase.DEPLOYMENT.value,
                trigger="user_action",
                metadata_=_build_phase_transition_metadata(
                    reason="deployment_api_status_sync",
                    source="api",
                    context={
                        "deployment_id": str(deployment.id),
                        "target_status": target_status,
                    },
                ),
            )
        )

    next_meta = dict(session.metadata_ or {})
    next_meta["deployment_status"] = phase_status
    next_meta["deployment_status_updated_at"] = at.isoformat()
    session.metadata_ = next_meta


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
    if payload.mode != "paper":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "DEPLOYMENT_MODE_NOT_SUPPORTED",
                "message": "Only paper deployment mode is supported.",
            },
        )

    strategy = await db.scalar(
        select(Strategy).where(
            Strategy.id == payload.strategy_id,
            Strategy.user_id == user.id,
        )
    )
    if strategy is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "STRATEGY_NOT_FOUND", "message": "Strategy not found."},
        )
    if str(strategy.status).strip().lower() not in _DEPLOYABLE_STRATEGY_STATUSES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "STRATEGY_NOT_DEPLOYABLE",
                "message": "Strategy is not ready for deployment.",
            },
        )
    compatibility = deployment_ops_domain.assess_strategy_runtime_compatibility(
        strategy.dsl_payload if isinstance(strategy.dsl_payload, dict) else {},
        fallback_timeframe=strategy.timeframe,
    )
    if compatibility.status == "blocked":
        error_code, message = deployment_ops_domain.build_runtime_compatibility_error(
            compatibility
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": error_code,
                "message": message,
                "blockers": list(compatibility.blockers),
            },
        )

    account = await db.scalar(
        select(BrokerAccount).where(
            BrokerAccount.id == payload.broker_account_id,
            BrokerAccount.user_id == user.id,
        )
    )
    if account is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "BROKER_ACCOUNT_NOT_FOUND",
                "message": "Broker account not found.",
            },
        )
    if account.mode != "paper":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "BROKER_ACCOUNT_MODE_NOT_SUPPORTED",
                "message": "Only paper broker accounts are supported for deployment.",
            },
        )
    if account.mode != payload.mode:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "BROKER_ACCOUNT_MODE_MISMATCH",
                "message": "Broker account mode does not match deployment mode.",
            },
        )
    if str(account.status).strip().lower() != "active":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "BROKER_ACCOUNT_INACTIVE",
                "message": "Broker account must be active before deployment.",
            },
        )

    capital_resolution = await deployment_ops_domain.resolve_deployment_capital(
        requested_capital=payload.capital_allocated,
        account=account,
    )
    capital_allocated = capital_resolution.amount
    resolved_runtime_state = (
        dict(payload.runtime_state) if isinstance(payload.runtime_state, dict) else {}
    )
    resolved_runtime_state["capital_resolution"] = {
        "source": capital_resolution.source,
        "resolved_amount": format(capital_allocated, "f"),
        "requested_amount": format(
            payload.capital_allocated.quantize(Decimal("0.01")), "f"
        ),
    }

    deployment = Deployment(
        strategy_id=payload.strategy_id,
        user_id=user.id,
        mode=payload.mode,
        status="pending",
        risk_limits=payload.risk_limits,
        capital_allocated=capital_allocated,
    )
    db.add(deployment)
    await db.flush()

    run = DeploymentRun(
        deployment_id=deployment.id,
        strategy_id=deployment.strategy_id,
        broker_account_id=payload.broker_account_id,
        status="stopped",
        runtime_state=resolved_runtime_state,
    )
    db.add(run)
    await db.commit()
    deployment = await db.scalar(
        select(Deployment)
        .options(*_deployment_query_options())
        .where(Deployment.id == deployment.id)
    )
    assert deployment is not None
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
    run = _latest_run(deployment)
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "DEPLOYMENT_RUN_MISSING",
                "message": "Deployment run is missing.",
            },
        )

    now = datetime.now(UTC)
    timeframe_raw = (
        deployment.strategy.dsl_payload.get("timeframe")
        if deployment.strategy is not None
        and isinstance(deployment.strategy.dsl_payload, dict)
        else deployment.strategy.timeframe
        if deployment.strategy is not None
        else "1m"
    )
    timeframe_seconds = timeframe_to_seconds(
        str(timeframe_raw or "1m"), default_seconds=60
    )
    state = dict(run.runtime_state) if isinstance(run.runtime_state, dict) else {}
    scheduler = (
        state.get("scheduler") if isinstance(state.get("scheduler"), dict) else {}
    )
    scheduler = dict(scheduler)
    execution = (
        state.get("execution") if isinstance(state.get("execution"), dict) else {}
    )
    execution = dict(execution)
    if target_status == "active":
        compatibility = deployment_ops_domain.assess_strategy_runtime_compatibility(
            deployment.strategy.dsl_payload
            if deployment.strategy is not None
            and isinstance(deployment.strategy.dsl_payload, dict)
            else {},
            fallback_timeframe=deployment.strategy.timeframe
            if deployment.strategy is not None
            else None,
        )
        if compatibility.status == "blocked":
            error_code, message = (
                deployment_ops_domain.build_runtime_compatibility_error(compatibility)
            )
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "code": error_code,
                    "message": message,
                    "blockers": list(compatibility.blockers),
                },
            )
        if not settings.paper_trading_enabled and deployment.mode == "paper":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "code": "PAPER_TRADING_DISABLED",
                    "message": "Paper trading is currently disabled.",
                },
            )
        broker_account = run.broker_account
        if broker_account is None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "code": "BROKER_ACCOUNT_NOT_FOUND",
                    "message": "Broker account is missing for this deployment run.",
                },
            )
        if str(broker_account.status).strip().lower() != "active":
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "code": "BROKER_ACCOUNT_INACTIVE",
                    "message": "Broker account must be active before starting deployment.",
                },
            )
        deployment.status = "active"
        deployment.deployed_at = deployment.deployed_at or now
        deployment.stopped_at = None
        run.status = "starting"
        if deployment.strategy is not None:
            deployment.strategy.status = "deployed"
        scheduler.update(
            {
                "timeframe_seconds": timeframe_seconds,
                "last_trigger_bucket": None,
                "last_enqueued_at": None,
                "fallback_last_enqueued_at": None,
                "updated_at": now.isoformat(),
            }
        )
        broker_provider = await _resolve_broker_provider_for_run(db, run=run)
        execution.update(
            {
                "order_execution_mode": "broker"
                if settings.paper_trading_execute_orders
                else "simulated",
                "market_data_source": f"{broker_provider}_rest",
                "broker_provider": broker_provider,
                "updated_at": now.isoformat(),
            }
        )
    elif target_status == "paused":
        deployment.status = "paused"
        run.status = "paused"
        scheduler.update({"updated_at": now.isoformat()})
        execution["updated_at"] = now.isoformat()
    elif target_status == "stopped":
        deployment.status = "stopped"
        deployment.stopped_at = now
        run.status = "stopped"
        scheduler.update({"updated_at": now.isoformat()})
        execution["updated_at"] = now.isoformat()
    else:
        raise ValueError(f"Unsupported target status: {target_status}")

    state["scheduler"] = scheduler
    state["execution"] = execution
    run.runtime_state = state

    if target_status == "active":
        await _enqueue_deployment_started_notification(
            db,
            deployment=deployment,
            at=now,
        )

    await _sync_session_deployment_phase_state(
        db,
        deployment=deployment,
        target_status=target_status,
        at=now,
    )

    await db.commit()
    await runtime_state_store.upsert(deployment.id, state)
    await append_trading_event_snapshot(db, deployment_id=deployment.id)
    reloaded = await db.scalar(
        select(Deployment)
        .options(*_deployment_query_options())
        .where(Deployment.id == deployment.id)
    )
    if reloaded is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "DEPLOYMENT_NOT_FOUND",
                "message": "Deployment not found after status transition.",
            },
        )
    return reloaded


async def _enqueue_deployment_started_notification(
    db: AsyncSession,
    *,
    deployment: Deployment,
    at: datetime,
) -> None:
    if not settings.notifications_enabled:
        return

    strategy_payload = (
        deployment.strategy.dsl_payload
        if deployment.strategy is not None
        and isinstance(deployment.strategy.dsl_payload, dict)
        else {}
    )
    universe = (
        strategy_payload.get("universe")
        if isinstance(strategy_payload.get("universe"), dict)
        else {}
    )
    symbols = (
        universe.get("tickers") if isinstance(universe.get("tickers"), list) else []
    )
    timeframe = (
        strategy_payload.get("timeframe")
        if isinstance(strategy_payload.get("timeframe"), str)
        else None
    )

    payload = {
        "deployment_id": str(deployment.id),
        "strategy_id": str(deployment.strategy_id),
        "mode": deployment.mode,
        "symbols": [str(symbol) for symbol in symbols],
        "timeframe": timeframe or "1m",
        "deployed_at": at.isoformat(),
    }
    service = NotificationOutboxService(db)
    await service.enqueue_event_for_user(
        user_id=deployment.user_id,
        event_type=EVENT_DEPLOYMENT_STARTED,
        event_key=f"deployment_started:{deployment.id}:{at.isoformat()}",
        payload=payload,
    )


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
            .order_by(Order.submitted_at.desc()),
        )
    ).all()
    return [_serialize_order(row) for row in rows]


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
    await db.refresh(action)
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
