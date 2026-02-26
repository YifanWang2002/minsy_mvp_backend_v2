"""Deployment lifecycle domain operations shared by API and MCP."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from packages.shared_settings.schema.settings import settings
from packages.domain.exceptions import DomainError
from packages.infra.providers.trading.adapters.alpaca_trading import AlpacaTradingAdapter
from packages.infra.providers.trading.credentials import CredentialCipher
from packages.core.events.notification_events import EVENT_DEPLOYMENT_STARTED
from packages.domain.notification.services.notification_outbox_service import NotificationOutboxService
from packages.domain.trading.runtime.timeframe_scheduler import timeframe_to_seconds
from packages.infra.db.models.broker_account import BrokerAccount
from packages.infra.db.models.deployment import Deployment
from packages.infra.db.models.deployment_run import DeploymentRun
from packages.infra.db.models.order import Order
from packages.infra.db.models.phase_transition import PhaseTransition
from packages.infra.db.models.position import Position
from packages.infra.db.models.session import Session
from packages.infra.db.models.strategy import Strategy
from packages.infra.observability.logger import logger
from packages.infra.redis.stores.runtime_state_store import runtime_state_store

_DEPLOYABLE_STRATEGY_STATUSES = {"validated", "backtested", "deployed"}
_DEPLOYMENT_PHASE = "deployment"
_VALID_PHASE_TRANSITIONS: dict[str, set[str]] = {
    "kyc": {"pre_strategy", "error"},
    "pre_strategy": {"strategy", "kyc", "error"},
    "strategy": {"pre_strategy", "deployment", "error"},
    "stress_test": {"strategy", "error"},
    "deployment": {"strategy", "completed", "error"},
    "error": {"kyc", "pre_strategy", "strategy"},
    "completed": set(),
}


def _raise_error(*, status_code: int, code: str, message: str) -> None:
    raise DomainError(status_code=status_code, code=code, message=message)


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


def _extract_credential_value(credentials: dict[str, Any], *keys: str) -> str:
    for key in keys:
        raw = credentials.get(key)
        if isinstance(raw, str):
            normalized = raw.strip()
            if normalized:
                return normalized
    return ""


def _resolve_capital_from_validation_metadata(account: BrokerAccount) -> Decimal | None:
    metadata = account.validation_metadata if isinstance(account.validation_metadata, dict) else {}
    for key in ("paper_equity", "paper_cash", "paper_buying_power", "equity", "cash", "buying_power"):
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

    if account.provider != "alpaca" or account.last_validated_status != "paper_probe_ok":
        return Decimal("10000.00")

    try:
        credentials = CredentialCipher().decrypt(account.encrypted_credentials)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[deployment_ops] decrypt broker credentials failed broker_account_id=%s error=%s",
            account.id,
            type(exc).__name__,
        )
        return Decimal("10000.00")

    api_key = _extract_credential_value(credentials, "APCA-API-KEY-ID", "api_key", "key")
    api_secret = _extract_credential_value(credentials, "APCA-API-SECRET-KEY", "api_secret", "secret")
    trading_base_url = _extract_credential_value(credentials, "trading_base_url", "base_url")
    if not trading_base_url:
        trading_base_url = settings.alpaca_paper_trading_base_url
    if not api_key or not api_secret:
        return Decimal("10000.00")

    adapter = AlpacaTradingAdapter(
        api_key=api_key,
        api_secret=api_secret,
        trading_base_url=trading_base_url,
    )
    try:
        state = await adapter.fetch_account_state()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[deployment_ops] fetch account state failed broker_account_id=%s error=%s",
            account.id,
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


def _latest_run(deployment: Deployment) -> DeploymentRun | None:
    if not deployment.deployment_runs:
        return None
    return sorted(
        deployment.deployment_runs,
        key=lambda item: item.created_at,
        reverse=True,
    )[0]


def _serialize_run(run: DeploymentRun | None) -> dict[str, Any] | None:
    if run is None:
        return None
    state = run.runtime_state if isinstance(run.runtime_state, dict) else {}
    scheduler = state.get("scheduler") if isinstance(state.get("scheduler"), dict) else {}
    return {
        "deployment_run_id": run.id,
        "deployment_id": run.deployment_id,
        "strategy_id": run.strategy_id,
        "broker_account_id": run.broker_account_id,
        "status": run.status,
        "last_bar_time": run.last_bar_time,
        "timeframe_seconds": _as_int(scheduler.get("timeframe_seconds")),
        "last_trigger_bucket": _as_int(scheduler.get("last_trigger_bucket")),
        "last_enqueued_at": _as_datetime(scheduler.get("last_enqueued_at")),
        "runtime_state": state,
        "created_at": run.created_at,
        "updated_at": run.updated_at,
    }


def serialize_deployment(deployment: Deployment) -> dict[str, Any]:
    run = _latest_run(deployment)
    strategy_payload = (
        deployment.strategy.dsl_payload
        if deployment.strategy is not None and isinstance(deployment.strategy.dsl_payload, dict)
        else {}
    )
    universe = strategy_payload.get("universe", {}) if isinstance(strategy_payload.get("universe"), dict) else {}
    symbols = universe.get("tickers") if isinstance(universe.get("tickers"), list) else []
    market = universe.get("market") if isinstance(universe.get("market"), str) else None
    timeframe = strategy_payload.get("timeframe") if isinstance(strategy_payload.get("timeframe"), str) else None
    return {
        "deployment_id": deployment.id,
        "strategy_id": deployment.strategy_id,
        "user_id": deployment.user_id,
        "mode": deployment.mode,
        "status": deployment.status,
        "market": market,
        "symbols": [str(symbol) for symbol in symbols],
        "timeframe": timeframe,
        "capital_allocated": float(deployment.capital_allocated),
        "risk_limits": deployment.risk_limits if isinstance(deployment.risk_limits, dict) else {},
        "deployed_at": deployment.deployed_at,
        "stopped_at": deployment.stopped_at,
        "created_at": deployment.created_at,
        "updated_at": deployment.updated_at,
        "run": _serialize_run(run),
    }


def serialize_position(position: Position) -> dict[str, Any]:
    return {
        "position_id": position.id,
        "deployment_id": position.deployment_id,
        "symbol": position.symbol,
        "side": position.side,
        "qty": float(position.qty),
        "avg_entry_price": float(position.avg_entry_price),
        "mark_price": float(position.mark_price),
        "unrealized_pnl": float(position.unrealized_pnl),
        "realized_pnl": float(position.realized_pnl),
        "created_at": position.created_at,
        "updated_at": position.updated_at,
    }


def serialize_order(order: Order) -> dict[str, Any]:
    metadata = order.metadata_ if isinstance(order.metadata_, dict) else {}
    provider_status = metadata.get("provider_status")
    if not isinstance(provider_status, str) or not provider_status.strip():
        provider_status = order.status
    return {
        "order_id": order.id,
        "deployment_id": order.deployment_id,
        "provider_order_id": order.provider_order_id,
        "client_order_id": order.client_order_id,
        "symbol": order.symbol,
        "side": order.side,
        "type": order.type,
        "qty": float(order.qty),
        "price": _decimal_to_float(order.price),
        "status": order.status,
        "provider_status": str(provider_status),
        "reject_reason": order.reject_reason,
        "last_sync_at": order.last_sync_at,
        "submitted_at": order.submitted_at,
        "metadata": metadata,
        "created_at": order.created_at,
        "updated_at": order.updated_at,
    }


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


def _can_transition(from_phase: str, to_phase: str) -> bool:
    targets = _VALID_PHASE_TRANSITIONS.get(from_phase, set())
    return to_phase in targets


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
    deployment_block_raw = artifacts.get(_DEPLOYMENT_PHASE)
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
    artifacts[_DEPLOYMENT_PHASE] = deployment_block
    session.artifacts = artifacts
    session.last_activity_at = at

    if session.current_phase != _DEPLOYMENT_PHASE and _can_transition(session.current_phase, _DEPLOYMENT_PHASE):
        from_phase = session.current_phase
        session.current_phase = _DEPLOYMENT_PHASE
        session.previous_response_id = None
        db.add(
            PhaseTransition(
                session_id=session.id,
                from_phase=from_phase,
                to_phase=_DEPLOYMENT_PHASE,
                trigger="user_action",
                metadata_=_build_phase_transition_metadata(
                    reason="deployment_status_sync",
                    source="domain",
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


async def load_owned_deployment(
    db: AsyncSession,
    *,
    deployment_id: UUID,
    user_id: UUID,
) -> Deployment:
    deployment = await db.scalar(
        select(Deployment)
        .options(selectinload(Deployment.deployment_runs), selectinload(Deployment.strategy))
        .where(
            Deployment.id == deployment_id,
            Deployment.user_id == user_id,
        )
    )
    if deployment is None:
        _raise_error(
            status_code=404,
            code="DEPLOYMENT_NOT_FOUND",
            message="Deployment not found.",
        )
    return deployment


async def create_deployment(
    db: AsyncSession,
    *,
    strategy_id: UUID,
    broker_account_id: UUID,
    user_id: UUID,
    mode: str,
    capital_allocated: Decimal,
    risk_limits: dict[str, Any],
    runtime_state: dict[str, Any],
) -> Deployment:
    if mode != "paper":
        _raise_error(
            status_code=422,
            code="DEPLOYMENT_MODE_NOT_SUPPORTED",
            message="Only paper deployment mode is supported.",
        )

    strategy = await db.scalar(
        select(Strategy).where(
            Strategy.id == strategy_id,
            Strategy.user_id == user_id,
        )
    )
    if strategy is None:
        _raise_error(
            status_code=404,
            code="STRATEGY_NOT_FOUND",
            message="Strategy not found.",
        )
    if str(strategy.status).strip().lower() not in _DEPLOYABLE_STRATEGY_STATUSES:
        _raise_error(
            status_code=422,
            code="STRATEGY_NOT_DEPLOYABLE",
            message="Strategy is not ready for deployment.",
        )

    account = await db.scalar(
        select(BrokerAccount).where(
            BrokerAccount.id == broker_account_id,
            BrokerAccount.user_id == user_id,
        )
    )
    if account is None:
        _raise_error(
            status_code=404,
            code="BROKER_ACCOUNT_NOT_FOUND",
            message="Broker account not found.",
        )
    if account.mode != "paper":
        _raise_error(
            status_code=422,
            code="BROKER_ACCOUNT_MODE_NOT_SUPPORTED",
            message="Only paper broker accounts are supported for deployment.",
        )
    if account.mode != mode:
        _raise_error(
            status_code=422,
            code="BROKER_ACCOUNT_MODE_MISMATCH",
            message="Broker account mode does not match deployment mode.",
        )

    resolved_capital_allocated = await _resolve_deployment_capital_allocated(
        requested_capital=capital_allocated,
        account=account,
    )

    deployment = Deployment(
        strategy_id=strategy_id,
        user_id=user_id,
        mode=mode,
        status="pending",
        risk_limits=risk_limits,
        capital_allocated=resolved_capital_allocated,
    )
    db.add(deployment)
    await db.flush()

    run = DeploymentRun(
        deployment_id=deployment.id,
        strategy_id=deployment.strategy_id,
        broker_account_id=broker_account_id,
        status="stopped",
        runtime_state=runtime_state,
    )
    db.add(run)
    await db.commit()
    reloaded = await db.scalar(
        select(Deployment)
        .options(selectinload(Deployment.deployment_runs), selectinload(Deployment.strategy))
        .where(Deployment.id == deployment.id)
    )
    if reloaded is None:
        _raise_error(
            status_code=404,
            code="DEPLOYMENT_NOT_FOUND",
            message="Deployment not found after creation.",
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
        if deployment.strategy is not None and isinstance(deployment.strategy.dsl_payload, dict)
        else {}
    )
    universe = strategy_payload.get("universe") if isinstance(strategy_payload.get("universe"), dict) else {}
    symbols = universe.get("tickers") if isinstance(universe.get("tickers"), list) else []
    timeframe = strategy_payload.get("timeframe") if isinstance(strategy_payload.get("timeframe"), str) else None

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


async def apply_status_transition(
    db: AsyncSession,
    *,
    deployment: Deployment,
    target_status: str,
) -> Deployment:
    run = _latest_run(deployment)
    if run is None:
        _raise_error(
            status_code=409,
            code="DEPLOYMENT_RUN_MISSING",
            message="Deployment run is missing.",
        )

    now = datetime.now(UTC)
    timeframe_raw = (
        deployment.strategy.dsl_payload.get("timeframe")
        if deployment.strategy is not None and isinstance(deployment.strategy.dsl_payload, dict)
        else deployment.strategy.timeframe if deployment.strategy is not None else "1m"
    )
    timeframe_seconds = timeframe_to_seconds(str(timeframe_raw or "1m"), default_seconds=60)
    state = dict(run.runtime_state) if isinstance(run.runtime_state, dict) else {}
    scheduler = state.get("scheduler") if isinstance(state.get("scheduler"), dict) else {}
    scheduler = dict(scheduler)
    execution = state.get("execution") if isinstance(state.get("execution"), dict) else {}
    execution = dict(execution)
    if target_status == "active":
        if not settings.paper_trading_enabled and deployment.mode == "paper":
            _raise_error(
                status_code=409,
                code="PAPER_TRADING_DISABLED",
                message="Paper trading is currently disabled.",
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
        execution.update(
            {
                "order_execution_mode": "broker" if settings.paper_trading_execute_orders else "simulated",
                "market_data_source": "alpaca_rest",
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
    reloaded = await db.scalar(
        select(Deployment)
        .options(selectinload(Deployment.deployment_runs), selectinload(Deployment.strategy))
        .where(Deployment.id == deployment.id)
    )
    if reloaded is None:
        _raise_error(
            status_code=404,
            code="DEPLOYMENT_NOT_FOUND",
            message="Deployment not found after status transition.",
        )
    return reloaded
