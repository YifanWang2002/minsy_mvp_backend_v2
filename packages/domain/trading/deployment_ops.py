"""Deployment lifecycle domain operations shared by API and MCP."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from packages.core.events.notification_events import EVENT_DEPLOYMENT_STARTED
from packages.domain.billing.quota_service import QuotaService
from packages.domain.billing.usage_service import UsageMetric, UsageService
from packages.domain.exceptions import DomainError
from packages.domain.notification.services.notification_outbox_service import (
    NotificationOutboxService,
)
from packages.domain.trading.runtime.timeframe_scheduler import (
    SUPPORTED_RUNTIME_TIMEFRAMES,
    timeframe_to_seconds,
)
from packages.domain.trading.services.broker_provider_service import (
    BrokerProviderService,
)
from packages.infra.db.models.broker_account import BrokerAccount
from packages.infra.db.models.deployment import Deployment
from packages.infra.db.models.deployment_run import DeploymentRun
from packages.infra.db.models.order import Order
from packages.infra.db.models.phase_transition import PhaseTransition
from packages.infra.db.models.position import Position
from packages.infra.db.models.session import Session
from packages.infra.db.models.strategy import Strategy
from packages.infra.db.models.user import User
from packages.infra.observability.logger import logger
from packages.infra.redis.stores.runtime_state_store import runtime_state_store
from packages.shared_settings.schema.settings import settings

_DEPLOYABLE_STRATEGY_STATUSES = {"validated", "backtested", "deployed"}
_DEPLOYMENT_PHASE = "deployment"
_broker_provider_service = BrokerProviderService()
_VALID_PHASE_TRANSITIONS: dict[str, set[str]] = {
    "kyc": {"pre_strategy", "error"},
    "pre_strategy": {"strategy", "kyc", "error"},
    "strategy": {"pre_strategy", "deployment", "error"},
    "stress_test": {"strategy", "error"},
    "deployment": {"strategy", "completed", "error"},
    "error": {"kyc", "pre_strategy", "strategy"},
    "completed": set(),
}


@dataclass(frozen=True, slots=True)
class DeploymentRuntimeCompatibility:
    """Deploy-time compatibility snapshot for the current paper runtime."""

    status: str
    blockers: tuple[str, ...]
    blocker_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DeploymentCapitalResolution:
    """Resolved deployment capital amount plus provenance."""

    amount: Decimal
    source: str


@dataclass(frozen=True, slots=True)
class BrokerCapitalBudget:
    """Broker capital budget snapshot for deployment allocation checks."""

    total_capital: Decimal
    reserved_capital: Decimal
    remaining_capital: Decimal
    reservation_statuses: tuple[str, ...]


_BROKER_CAPITAL_RESERVATION_STATUSES: tuple[str, ...] = (
    "pending",
    "active",
    "paused",
    "error",
)


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


def _non_negative_decimal(value: Any) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except Exception:  # noqa: BLE001
        return Decimal("0.00")
    if parsed <= 0:
        return Decimal("0.00")
    return parsed.quantize(Decimal("0.01"))


def _normalize_reservation_statuses(
    statuses: tuple[str, ...] | list[str] | None,
) -> tuple[str, ...]:
    if not statuses:
        return _BROKER_CAPITAL_RESERVATION_STATUSES
    normalized = tuple(
        status.strip().lower()
        for status in statuses
        if isinstance(status, str) and status.strip()
    )
    return normalized or _BROKER_CAPITAL_RESERVATION_STATUSES


async def resolve_broker_capital_budget(
    db: AsyncSession,
    *,
    account: BrokerAccount,
    total_capital: Decimal | None = None,
    reservation_statuses: tuple[str, ...] | list[str] | None = None,
) -> BrokerCapitalBudget:
    normalized_statuses = _normalize_reservation_statuses(reservation_statuses)
    resolved_total_capital = _non_negative_decimal(total_capital)
    if resolved_total_capital <= 0:
        capital_resolution = await resolve_deployment_capital(
            requested_capital=Decimal("0.00"),
            account=account,
        )
        resolved_total_capital = _non_negative_decimal(capital_resolution.amount)

    reserved_rows_subquery = (
        select(
            Deployment.id.label("deployment_id"),
            Deployment.capital_allocated.label("capital_allocated"),
        )
        .join(DeploymentRun, DeploymentRun.deployment_id == Deployment.id)
        .where(
            DeploymentRun.broker_account_id == account.id,
            Deployment.user_id == account.user_id,
            Deployment.mode == account.mode,
            Deployment.status.in_(normalized_statuses),
        )
        .group_by(Deployment.id, Deployment.capital_allocated)
        .subquery()
    )
    reserved_raw = await db.scalar(
        select(func.coalesce(func.sum(reserved_rows_subquery.c.capital_allocated), 0))
    )
    reserved_capital = _non_negative_decimal(reserved_raw)
    remaining_capital = resolved_total_capital - reserved_capital
    if remaining_capital < 0:
        remaining_capital = Decimal("0.00")
    remaining_capital = remaining_capital.quantize(Decimal("0.01"))

    return BrokerCapitalBudget(
        total_capital=resolved_total_capital.quantize(Decimal("0.01")),
        reserved_capital=reserved_capital,
        remaining_capital=remaining_capital,
        reservation_statuses=normalized_statuses,
    )


def resolve_deployment_capital_with_budget(
    *,
    requested_capital: Decimal,
    budget: BrokerCapitalBudget,
    auto_resolution: DeploymentCapitalResolution,
) -> DeploymentCapitalResolution:
    requested = requested_capital.quantize(Decimal("0.01"))
    remaining = budget.remaining_capital.quantize(Decimal("0.01"))
    if remaining <= 0:
        _raise_error(
            status_code=422,
            code="DEPLOYMENT_CAPITAL_EXCEEDS_REMAINING_BUDGET",
            message="No remaining broker capital budget available for this account.",
        )

    if requested > 0:
        if requested > remaining:
            _raise_error(
                status_code=422,
                code="DEPLOYMENT_CAPITAL_EXCEEDS_REMAINING_BUDGET",
                message="Requested capital exceeds remaining broker capital budget.",
            )
        return DeploymentCapitalResolution(amount=requested, source="requested")

    resolved_auto = auto_resolution.amount.quantize(Decimal("0.01"))
    capped = min(resolved_auto, remaining)
    if capped <= 0:
        _raise_error(
            status_code=422,
            code="DEPLOYMENT_CAPITAL_EXCEEDS_REMAINING_BUDGET",
            message="No remaining broker capital budget available for this account.",
        )
    source = auto_resolution.source
    if capped < resolved_auto:
        source = f"{source}:capped_by_remaining_budget"
    return DeploymentCapitalResolution(amount=capped, source=source)


def _resolve_capital_from_validation_metadata(account: BrokerAccount) -> Decimal | None:
    metadata = account.validation_metadata if isinstance(account.validation_metadata, dict) else {}
    for key in ("paper_equity", "paper_cash", "paper_buying_power", "equity", "cash", "buying_power"):
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
    resolution = await resolve_deployment_capital(
        requested_capital=requested_capital,
        account=account,
    )
    return resolution.amount


async def resolve_deployment_capital(
    *,
    requested_capital: Decimal,
    account: BrokerAccount,
) -> DeploymentCapitalResolution:
    if requested_capital > 0:
        return DeploymentCapitalResolution(
            amount=requested_capital.quantize(Decimal("0.01")),
            source="requested",
        )

    from_probe = _resolve_capital_from_validation_metadata(account)
    if from_probe is not None:
        return DeploymentCapitalResolution(amount=from_probe, source="validation_metadata")

    from_account_metadata = _resolve_capital_from_account_metadata(account)
    if from_account_metadata is not None:
        return DeploymentCapitalResolution(amount=from_account_metadata, source="account_metadata")

    binding = _broker_provider_service.build_adapter_binding_from_account(account)
    adapter = binding.adapter
    if adapter is None:
        return DeploymentCapitalResolution(amount=Decimal("10000.00"), source="fallback_default")

    try:
        state = await adapter.fetch_account_state()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[deployment_ops] fetch account state failed broker_account_id=%s provider=%s error=%s",
            account.id,
            binding.provider,
            type(exc).__name__,
        )
        return DeploymentCapitalResolution(amount=Decimal("10000.00"), source="fallback_default")
    finally:
        await adapter.aclose()

    for candidate in (state.equity, state.cash, state.buying_power):
        capital = _positive_decimal_or_none(candidate)
        if capital is not None:
            return DeploymentCapitalResolution(amount=capital, source="adapter_fetch")
    return DeploymentCapitalResolution(amount=Decimal("10000.00"), source="fallback_default")


async def _resolve_broker_provider_for_run(db: AsyncSession, *, run: DeploymentRun) -> str:
    account = await db.scalar(select(BrokerAccount).where(BrokerAccount.id == run.broker_account_id))
    if account is None:
        return "unknown"
    provider = str(account.provider).strip().lower()
    return provider or "unknown"


def assess_strategy_runtime_compatibility(
    strategy_payload: dict[str, Any] | None,
    *,
    fallback_timeframe: str | None = None,
) -> DeploymentRuntimeCompatibility:
    payload = strategy_payload if isinstance(strategy_payload, dict) else {}
    blockers: list[str] = []
    blocker_codes: list[str] = []
    timeframe = str(payload.get("timeframe") or fallback_timeframe or "").strip().lower()
    if timeframe and timeframe not in SUPPORTED_RUNTIME_TIMEFRAMES:
        blocker_codes.append("DEPLOYMENT_RUNTIME_UNSUPPORTED_TIMEFRAME")
        blockers.append(
            "Paper trading timeframe must match DSL-supported values: "
            f"{', '.join(SUPPORTED_RUNTIME_TIMEFRAMES)}."
        )

    universe = payload.get("universe") if isinstance(payload.get("universe"), dict) else {}
    raw_tickers = universe.get("tickers")
    tickers = raw_tickers if isinstance(raw_tickers, list) else []
    normalized_tickers = [str(item).strip() for item in tickers if str(item).strip()]
    if len(normalized_tickers) > 1:
        blocker_codes.append("DEPLOYMENT_RUNTIME_UNSUPPORTED_MULTI_SYMBOL")
        blockers.append(
            "Paper trading currently supports only one symbol per deployment."
        )

    return DeploymentRuntimeCompatibility(
        status="blocked" if blockers else "ok",
        blockers=tuple(blockers),
        blocker_codes=tuple(blocker_codes),
    )


def build_runtime_compatibility_error(
    compatibility: DeploymentRuntimeCompatibility,
) -> tuple[str, str]:
    if compatibility.status != "blocked":
        return ("", "")
    codes = set(compatibility.blocker_codes)
    if codes == {"DEPLOYMENT_RUNTIME_UNSUPPORTED_MULTI_SYMBOL"}:
        return (
            "DEPLOYMENT_RUNTIME_UNSUPPORTED_MULTI_SYMBOL",
            compatibility.blockers[0],
        )
    if codes == {"DEPLOYMENT_RUNTIME_UNSUPPORTED_EXIT_RULE"}:
        return (
            "DEPLOYMENT_RUNTIME_UNSUPPORTED_EXIT_RULE",
            compatibility.blockers[0],
        )
    if codes == {"DEPLOYMENT_RUNTIME_UNSUPPORTED_TIMEFRAME"}:
        return (
            "DEPLOYMENT_RUNTIME_UNSUPPORTED_TIMEFRAME",
            compatibility.blockers[0],
        )
    return (
        "DEPLOYMENT_RUNTIME_UNSUPPORTED",
        compatibility.blockers[0]
        if compatibility.blockers
        else "Strategy is not compatible with the current paper runtime.",
    )


def _latest_run(deployment: Deployment) -> DeploymentRun | None:
    if not deployment.deployment_runs:
        return None
    return sorted(
        deployment.deployment_runs,
        key=lambda item: item.created_at,
        reverse=True,
    )[0]


def _deployment_query_options() -> tuple[object, ...]:
    return (
        selectinload(Deployment.deployment_runs).selectinload(DeploymentRun.broker_account),
        selectinload(Deployment.strategy),
    )


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
        .options(*_deployment_query_options())
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
    compatibility = assess_strategy_runtime_compatibility(
        strategy.dsl_payload if isinstance(strategy.dsl_payload, dict) else {},
        fallback_timeframe=strategy.timeframe,
    )
    if compatibility.status == "blocked":
        error_code, message = build_runtime_compatibility_error(compatibility)
        _raise_error(
            status_code=422,
            code=error_code,
            message=message,
        )

    account = await db.scalar(
        select(BrokerAccount)
        .where(
            BrokerAccount.id == broker_account_id,
            BrokerAccount.user_id == user_id,
        )
        .with_for_update()
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
    if str(account.status).strip().lower() != "active":
        _raise_error(
            status_code=422,
            code="BROKER_ACCOUNT_INACTIVE",
            message="Broker account must be active before deployment.",
        )

    owner = await db.scalar(select(User).where(User.id == user_id))
    tier = owner.current_tier if owner is not None else "free"
    quota_service = QuotaService(UsageService(db))
    await quota_service.assert_quota_available(
        user_id=user_id,
        tier=tier,
        metric=UsageMetric.DEPLOYMENTS_RUNNING_COUNT,
        increment=1,
    )

    auto_capital_resolution = await resolve_deployment_capital(
        requested_capital=Decimal("0.00"),
        account=account,
    )
    capital_budget = await resolve_broker_capital_budget(
        db,
        account=account,
        total_capital=auto_capital_resolution.amount,
    )
    capital_resolution = resolve_deployment_capital_with_budget(
        requested_capital=capital_allocated,
        budget=capital_budget,
        auto_resolution=auto_capital_resolution,
    )
    resolved_capital_allocated = capital_resolution.amount
    resolved_runtime_state = dict(runtime_state) if isinstance(runtime_state, dict) else {}
    resolved_runtime_state["capital_resolution"] = {
        "source": capital_resolution.source,
        "resolved_amount": format(resolved_capital_allocated, "f"),
        "requested_amount": format(capital_allocated.quantize(Decimal("0.01")), "f"),
    }
    resolved_runtime_state["capital_budget"] = {
        "total_capital": format(capital_budget.total_capital, "f"),
        "reserved_capital": format(capital_budget.reserved_capital, "f"),
        "remaining_capital": format(capital_budget.remaining_capital, "f"),
        "reservation_statuses": list(capital_budget.reservation_statuses),
    }

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
        runtime_state=resolved_runtime_state,
    )
    db.add(run)
    await db.commit()
    reloaded = await db.scalar(
        select(Deployment)
        .options(*_deployment_query_options())
        .where(Deployment.id == deployment.id)
    )
    if reloaded is None:
        _raise_error(
            status_code=404,
            code="DEPLOYMENT_NOT_FOUND",
            message="Deployment not found after creation.",
        )
    return reloaded


async def _assert_running_deployment_quota_for_activation(
    db: AsyncSession,
    *,
    user_id: UUID,
    current_status: str,
) -> None:
    normalized_status = str(current_status).strip().lower()
    # pending/paused/error are already counted by deployments_running_count.
    if normalized_status in {"active", "pending", "paused", "error"}:
        return

    owner = await db.scalar(select(User).where(User.id == user_id))
    tier = owner.current_tier if owner is not None else "free"
    quota_service = QuotaService(UsageService(db))
    await quota_service.assert_quota_available(
        user_id=user_id,
        tier=tier,
        metric=UsageMetric.DEPLOYMENTS_RUNNING_COUNT,
        increment=1,
    )


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
        await _assert_running_deployment_quota_for_activation(
            db,
            user_id=deployment.user_id,
            current_status=str(deployment.status),
        )
        compatibility = assess_strategy_runtime_compatibility(
            deployment.strategy.dsl_payload
            if deployment.strategy is not None and isinstance(deployment.strategy.dsl_payload, dict)
            else {},
            fallback_timeframe=deployment.strategy.timeframe if deployment.strategy is not None else None,
        )
        if compatibility.status == "blocked":
            error_code, message = build_runtime_compatibility_error(compatibility)
            _raise_error(
                status_code=422,
                code=error_code,
                message=message,
            )
        if not settings.paper_trading_enabled and deployment.mode == "paper":
            _raise_error(
                status_code=409,
                code="PAPER_TRADING_DISABLED",
                message="Paper trading is currently disabled.",
            )
        # Use explicit query instead of relationship lazy-loading so async callers
        # (including MCP tools) do not hit MissingGreenlet when validating status.
        broker_account = await db.scalar(
            select(BrokerAccount).where(
                BrokerAccount.id == run.broker_account_id,
                BrokerAccount.user_id == deployment.user_id,
            )
        )
        if broker_account is None:
            _raise_error(
                status_code=409,
                code="BROKER_ACCOUNT_NOT_FOUND",
                message="Broker account is missing for this deployment run.",
            )
        if str(broker_account.status).strip().lower() != "active":
            _raise_error(
                status_code=422,
                code="BROKER_ACCOUNT_INACTIVE",
                message="Broker account must be active before starting deployment.",
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
                "order_execution_mode": "broker" if settings.paper_trading_execute_orders else "simulated",
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
    reloaded = await db.scalar(
        select(Deployment)
        .options(*_deployment_query_options())
        .where(Deployment.id == deployment.id)
    )
    if reloaded is None:
        _raise_error(
            status_code=404,
            code="DEPLOYMENT_NOT_FOUND",
            message="Deployment not found after status transition.",
        )
    return reloaded
