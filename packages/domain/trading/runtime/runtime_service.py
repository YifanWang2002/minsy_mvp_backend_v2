"""Runtime orchestration for one deployment bar-close cycle."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from packages.infra.providers.trading.adapters.alpaca_trading import AlpacaTradingAdapter
from packages.infra.providers.trading.adapters.base import (
    AccountState,
    BrokerAdapter,
    OhlcvBar,
    OrderIntent,
    PositionRecord,
)
from packages.infra.providers.trading.credentials import CredentialCipher
from packages.infra.redis.locks.deployment_lock import deployment_runtime_lock
from packages.infra.redis.stores.runtime_state_store import runtime_state_store
from packages.infra.redis.stores.signal_store import SignalRecord, signal_store
from packages.shared_settings.schema.settings import settings
from packages.domain.trading.runtime.circuit_breaker import (
    CircuitBreakerOpenError,
    execute_with_retry,
    get_broker_request_circuit_breaker,
)
from packages.domain.trading.runtime.kill_switch import RuntimeKillSwitch
from packages.domain.trading.runtime.order_manager import OrderManager
from packages.domain.trading.runtime.order_state_machine import (
    apply_order_status_transition,
    normalize_order_status,
)
from packages.domain.trading.runtime.risk_gate import RiskConfig, RiskContext, RiskGate
from packages.domain.trading.runtime.signal_runtime import LiveSignalRuntime
from packages.domain.market_data.runtime import RuntimeBar, market_data_runtime
from packages.domain.trading.pnl.service import PnlService
from packages.infra.db.models.broker_account import BrokerAccount
from packages.infra.db.models.deployment import Deployment
from packages.infra.db.models.deployment_run import DeploymentRun
from packages.infra.db.models.fill import Fill
from packages.infra.db.models.manual_trade_action import ManualTradeAction
from packages.infra.db.models.order import Order
from packages.infra.db.models.order_state_transition import OrderStateTransition
from packages.infra.db.models.position import Position
from packages.infra.db.models.signal_event import SignalEvent
from packages.core.events.notification_events import (
    EVENT_POSITION_CLOSED,
    EVENT_POSITION_OPENED,
)
from packages.domain.notification.services.notification_outbox_service import NotificationOutboxService
from packages.domain.trading.services.trade_approval_service import TradeApprovalService
from packages.domain.trading.services.trading_preference_service import TradingPreferenceService


@dataclass(frozen=True, slots=True)
class ProcessCycleResult:
    deployment_id: UUID
    signal: str
    reason: str
    execution_event_id: UUID | None = None
    order_id: UUID | None = None
    idempotent_hit: bool = False
    bar_time: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ManualActionExecutionResult:
    action_id: UUID
    status: str
    reason: str
    order_id: UUID | None = None
    idempotent_hit: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AlpacaCredentialBundle:
    api_key: str
    api_secret: str
    trading_base_url: str


def _latest_run(deployment: Deployment) -> DeploymentRun | None:
    if not deployment.deployment_runs:
        return None
    return sorted(
        deployment.deployment_runs,
        key=lambda item: item.created_at,
        reverse=True,
    )[0]


def _resolve_scope(deployment: Deployment) -> tuple[str, str, str]:
    payload = deployment.strategy.dsl_payload if isinstance(deployment.strategy.dsl_payload, dict) else {}
    universe = payload.get("universe", {}) if isinstance(payload.get("universe"), dict) else {}
    market = str(universe.get("market") or "stocks").strip().lower()
    tickers = universe.get("tickers") if isinstance(universe.get("tickers"), list) else []
    symbol = str(tickers[0]).strip().upper() if tickers else deployment.strategy.symbols[0].strip().upper()
    timeframe = str(payload.get("timeframe") or deployment.strategy.timeframe or "1m").strip().lower()
    return market, symbol, timeframe


def _resolve_position_side(positions: list[Position], symbol: str) -> str:
    for position in positions:
        if position.symbol.upper() != symbol.upper():
            continue
        qty = float(position.qty)
        if qty <= 0:
            continue
        side = position.side.strip().lower()
        if side in {"long", "short"}:
            return side
    return "flat"


def _resolve_position_qty(positions: list[Position], symbol: str) -> float:
    for position in positions:
        if position.symbol.upper() == symbol.upper():
            return float(position.qty)
    return 0.0


def _merge_seed_bars(
    *,
    historical_bars: list[OhlcvBar],
    latest_bar: OhlcvBar | None,
) -> list[OhlcvBar]:
    ordered = sorted(historical_bars, key=lambda item: item.timestamp)
    if latest_bar is None:
        return ordered
    if not ordered:
        return [latest_bar]
    if latest_bar.timestamp < ordered[-1].timestamp:
        return ordered
    deduped: dict[datetime, OhlcvBar] = {bar.timestamp: bar for bar in ordered}
    deduped[latest_bar.timestamp] = latest_bar
    return [deduped[ts] for ts in sorted(deduped)]


def _resolve_mark_price(
    *,
    market: str,
    symbol: str,
    timeframe: str,
    payload: dict[str, Any],
    positions: list[Position],
) -> Decimal:
    explicit_mark = payload.get("mark_price")
    if explicit_mark is not None:
        try:
            mark = Decimal(str(explicit_mark))
        except (InvalidOperation, ValueError, TypeError):
            return Decimal("0")
        return mark if mark > 0 else Decimal("0")

    bars = market_data_runtime.get_recent_bars(
        market=market,
        symbol=symbol,
        timeframe=timeframe,
        limit=1,
    )
    if bars:
        mark = Decimal(str(bars[-1].close))
        if mark > 0:
            return mark

    for position in positions:
        if position.symbol.upper() != symbol.upper():
            continue
        mark = Decimal(str(position.mark_price))
        if mark > 0:
            return mark
        avg = Decimal(str(position.avg_entry_price))
        if avg > 0:
            return avg
    return Decimal("0")


async def _resolve_mark_price_with_provider_fallback(
    *,
    market: str,
    symbol: str,
    timeframe: str,
    payload: dict[str, Any],
    positions: list[Position],
    adapter: AlpacaTradingAdapter | None,
) -> Decimal:
    mark = _resolve_mark_price(
        market=market,
        symbol=symbol,
        timeframe=timeframe,
        payload=payload,
        positions=positions,
    )
    if mark > 0:
        return mark
    if adapter is None:
        return Decimal("0")

    try:
        quote = await adapter.fetch_latest_quote(symbol)
    except Exception:  # noqa: BLE001
        quote = None
    if quote is not None and quote.last is not None:
        quote_last = Decimal(str(quote.last))
        if quote_last > 0:
            market_data_runtime.upsert_quote(
                market=market,
                symbol=symbol,
                quote=quote,
            )
            return quote_last

    try:
        latest_bar = await adapter.fetch_latest_1m_bar(symbol)
    except Exception:  # noqa: BLE001
        latest_bar = None
    if latest_bar is None:
        try:
            bars_1m = await adapter.fetch_ohlcv_1m(symbol, limit=1)
        except Exception:  # noqa: BLE001
            bars_1m = []
        if not bars_1m:
            return Decimal("0")
        latest_bar = bars_1m[-1]

    market_data_runtime.ingest_1m_bar(
        market=market,
        symbol=symbol,
        bar=latest_bar,
    )
    close = Decimal(str(latest_bar.close))
    return close if close > 0 else Decimal("0")


def _to_decimal_or_none(value: object) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None


def _parse_iso_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
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
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _to_float_or_none(value: Decimal | float | int | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _should_sync_broker_account(
    *,
    run: DeploymentRun,
    now: datetime,
) -> bool:
    state = run.runtime_state if isinstance(run.runtime_state, dict) else {}
    broker_state = state.get("broker_account") if isinstance(state.get("broker_account"), dict) else {}
    last_fetched = _parse_iso_datetime(broker_state.get("fetched_at"))
    if last_fetched is None:
        return True
    interval_seconds = max(1.0, float(settings.paper_trading_broker_account_sync_interval_seconds))
    return (now - last_fetched).total_seconds() >= interval_seconds


def _set_broker_account_state(
    *,
    run: DeploymentRun,
    payload: dict[str, Any],
) -> None:
    state = dict(run.runtime_state) if isinstance(run.runtime_state, dict) else {}
    state["broker_account"] = payload
    run.runtime_state = state


def _build_broker_account_payload(
    *,
    account_state: AccountState,
    positions: list[PositionRecord],
    fetched_at: datetime,
) -> dict[str, Any]:
    unrealized_total = sum((position.unrealized_pnl for position in positions), Decimal("0"))
    realized_total = sum((position.realized_pnl for position in positions), Decimal("0"))
    symbols = sorted({position.symbol.upper() for position in positions if position.symbol})
    return {
        "provider": "alpaca",
        "source": "broker_reported",
        "sync_status": "ok",
        "fetched_at": fetched_at.isoformat(),
        "equity": _to_float_or_none(account_state.equity),
        "cash": _to_float_or_none(account_state.cash),
        "buying_power": _to_float_or_none(account_state.buying_power),
        "margin_used": _to_float_or_none(account_state.margin_used),
        "unrealized_pnl": _to_float_or_none(unrealized_total),
        "realized_pnl": _to_float_or_none(realized_total),
        "positions_count": len(positions),
        "symbols": symbols[:20],
    }


def _mark_broker_account_sync_error(
    *,
    run: DeploymentRun,
    error: str,
) -> None:
    state = run.runtime_state if isinstance(run.runtime_state, dict) else {}
    existing = state.get("broker_account") if isinstance(state.get("broker_account"), dict) else {}
    payload = dict(existing)
    payload.update(
        {
            "provider": "alpaca",
            "source": "broker_reported",
            "sync_status": "error",
            "error": error[:500],
            "updated_at": datetime.now(UTC).isoformat(),
        }
    )
    _set_broker_account_state(run=run, payload=payload)


async def _sync_broker_account_snapshot(
    *,
    run: DeploymentRun,
    adapter: BrokerAdapter,
) -> dict[str, Any]:
    fetched_at = datetime.now(UTC)
    account_state = await adapter.fetch_account_state()
    try:
        positions = await adapter.fetch_positions()
    except Exception:  # noqa: BLE001
        positions = []
    payload = _build_broker_account_payload(
        account_state=account_state,
        positions=positions,
        fetched_at=fetched_at,
    )
    _set_broker_account_state(run=run, payload=payload)
    return payload


def _build_risk_config(risk_limits: dict[str, Any]) -> RiskConfig:
    return RiskConfig(
        max_position_notional=(
            float(risk_limits["max_position_notional"])
            if risk_limits.get("max_position_notional") is not None
            else None
        ),
        max_symbol_exposure_pct=(
            float(risk_limits["max_symbol_exposure_pct"])
            if risk_limits.get("max_symbol_exposure_pct") is not None
            else None
        ),
        min_order_qty=float(risk_limits.get("min_order_qty", 0.0)),
        max_daily_loss=(
            float(risk_limits["max_daily_loss"]) if risk_limits.get("max_daily_loss") is not None else None
        ),
    )


def _extract_credential_value(credentials: dict[str, Any], *keys: str) -> str:
    for key in keys:
        raw = credentials.get(key)
        if isinstance(raw, str):
            normalized = raw.strip()
            if normalized:
                return normalized
    return ""


async def _resolve_alpaca_credentials_for_run(
    *,
    db: AsyncSession,
    run: DeploymentRun,
) -> AlpacaCredentialBundle | None:
    account = await db.scalar(select(BrokerAccount).where(BrokerAccount.id == run.broker_account_id))
    if account is None or account.provider != "alpaca":
        return None
    try:
        credentials = CredentialCipher().decrypt(account.encrypted_credentials)
    except Exception:  # noqa: BLE001
        return None
    api_key = _extract_credential_value(
        credentials,
        "APCA-API-KEY-ID",
        "api_key",
        "key",
    )
    api_secret = _extract_credential_value(
        credentials,
        "APCA-API-SECRET-KEY",
        "api_secret",
        "secret",
    )
    if not api_key or not api_secret:
        return None
    trading_base_url = _extract_credential_value(
        credentials,
        "trading_base_url",
        "base_url",
    )
    if not trading_base_url:
        trading_base_url = settings.alpaca_trading_base_url
    return AlpacaCredentialBundle(
        api_key=api_key,
        api_secret=api_secret,
        trading_base_url=trading_base_url,
    )


async def _build_alpaca_adapter_for_market_data(
    *,
    db: AsyncSession,
    run: DeploymentRun,
) -> AlpacaTradingAdapter | None:
    bundle = await _resolve_alpaca_credentials_for_run(db=db, run=run)
    if bundle is None:
        return None
    return AlpacaTradingAdapter(
        api_key=bundle.api_key,
        api_secret=bundle.api_secret,
        trading_base_url=bundle.trading_base_url,
    )


async def _build_adapter_if_enabled(
    *,
    db: AsyncSession,
    run: DeploymentRun,
) -> BrokerAdapter | None:
    if not settings.paper_trading_execute_orders:
        return None
    bundle = await _resolve_alpaca_credentials_for_run(db=db, run=run)
    if bundle is None:
        return None
    return AlpacaTradingAdapter(
        api_key=bundle.api_key,
        api_secret=bundle.api_secret,
        trading_base_url=bundle.trading_base_url,
    )


async def _submit_order_with_resilience(
    *,
    db: AsyncSession,
    deployment_id: str,
    intent: OrderIntent,
    adapter: BrokerAdapter | None,
) -> Any:
    breaker = get_broker_request_circuit_breaker()

    async def _operation() -> Any:
        manager = OrderManager()
        return await manager.submit_order_intent(
            db=db,
            deployment_id=deployment_id,
            intent=intent,
            adapter=adapter,
        )

    return await execute_with_retry(
        _operation,
        breaker=breaker,
        max_attempts=settings.paper_trading_broker_retry_max_attempts,
        base_backoff_seconds=settings.paper_trading_broker_retry_backoff_seconds,
    )


async def _seed_runtime_market_data_from_provider(
    *,
    db: AsyncSession,
    run: DeploymentRun,
    market: str,
    symbol: str,
    timeframe: str,
    limit: int,
) -> tuple[list[RuntimeBar], dict[str, Any], AlpacaTradingAdapter | None]:
    metadata: dict[str, Any] = {
        "market_data_fallback": "not_attempted",
        "market_data_source": (
            "redis_first"
            if settings.effective_market_data_redis_read_enabled
            else "runtime_cache"
        ),
    }
    adapter = await _build_alpaca_adapter_for_market_data(db=db, run=run)
    if adapter is None:
        metadata["market_data_fallback"] = "adapter_unavailable"
        return [], metadata, None

    metadata["market_data_fallback"] = "attempted"
    metadata["market_data_source"] = "alpaca_rest_fallback"

    bars_1m: list[OhlcvBar] = []
    try:
        bars_1m = await adapter.fetch_ohlcv_1m(symbol, limit=max(limit, 2))
    except Exception as exc:  # noqa: BLE001
        metadata["market_data_fallback_error"] = type(exc).__name__
        bars_1m = []

    latest_bar: OhlcvBar | None = None
    try:
        latest_bar = await adapter.fetch_latest_1m_bar(symbol)
    except Exception as exc:  # noqa: BLE001
        metadata["market_data_fallback_latest_1m_error"] = type(exc).__name__

    if bars_1m:
        historical_latest_1m = max(item.timestamp for item in bars_1m)
        metadata["market_data_fallback_historical_latest_1m"] = historical_latest_1m.isoformat()
    if latest_bar is not None:
        metadata["market_data_fallback_latest_1m"] = latest_bar.timestamp.isoformat()
    ordered = _merge_seed_bars(
        historical_bars=bars_1m,
        latest_bar=latest_bar,
    )
    if latest_bar is not None:
        metadata["market_data_fallback_latest_1m_merged"] = (
            "yes" if ordered and ordered[-1].timestamp == latest_bar.timestamp else "no"
        )

    if ordered:
        for bar in ordered:
            market_data_runtime.ingest_1m_bar(
                market=market,
                symbol=symbol,
                bar=bar,
            )
        metadata["market_data_fallback_ingested_1m"] = len(ordered)
        metadata["market_data_fallback_latest_1m"] = ordered[-1].timestamp.isoformat()
    else:
        metadata["market_data_fallback_ingested_1m"] = 0

    try:
        quote = await adapter.fetch_latest_quote(symbol)
    except Exception as exc:  # noqa: BLE001
        metadata["market_data_fallback_quote_error"] = type(exc).__name__
        quote = None
    if quote is not None:
        market_data_runtime.upsert_quote(
            market=market,
            symbol=symbol,
            quote=quote,
        )
        metadata["market_data_fallback_quote"] = "yes"
    else:
        metadata["market_data_fallback_quote"] = "no"

    bars = market_data_runtime.get_recent_bars(
        market=market,
        symbol=symbol,
        timeframe=timeframe,
        limit=limit,
    )
    if bars:
        metadata["market_data_fallback"] = "hydrated"
        metadata["market_data_fallback_bar_time"] = bars[-1].timestamp.isoformat()
    else:
        metadata["market_data_fallback"] = "empty"
    return bars, metadata, adapter


async def _refresh_symbol_mark_price_and_unrealized(
    *,
    db: AsyncSession,
    deployment_id: UUID,
    positions: list[Position],
    symbol: str,
    mark_price: Decimal,
) -> None:
    if mark_price <= 0:
        return

    target_symbol = symbol.strip().upper()
    has_open_position = False
    price_changed = False
    for position in positions:
        if position.symbol.strip().upper() != target_symbol:
            continue
        qty = Decimal(str(position.qty))
        if qty <= 0:
            continue
        has_open_position = True
        current_mark = Decimal(str(position.mark_price))
        if current_mark != mark_price:
            position.mark_price = mark_price
            price_changed = True

    if not has_open_position or not price_changed:
        return

    pnl_service = PnlService()
    await pnl_service.build_snapshot(db, deployment_id=deployment_id)


async def _persist_runtime_state(
    *,
    deployment_id: UUID,
    run: DeploymentRun | None,
    status: str,
    reason: str,
    signal: str | None = None,
    bar_time: datetime | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    now = datetime.now(UTC)
    payload = dict(run.runtime_state) if run is not None and isinstance(run.runtime_state, dict) else {}
    payload.update(
        {
            "runtime_status": status,
            "runtime_reason": reason,
            "updated_at": now.isoformat(),
        }
    )
    if signal is not None:
        payload["runtime_signal"] = signal
    if bar_time is not None:
        payload["runtime_bar_time"] = bar_time.isoformat()
    if metadata:
        payload["runtime_metadata"] = metadata
    await runtime_state_store.upsert(deployment_id, payload)
    await runtime_state_store.publish_live_trading_health(
        {
            "deployment_id": str(deployment_id),
            "runtime_status": status,
            "runtime_reason": reason,
            "runtime_bar_time": bar_time.isoformat() if bar_time is not None else None,
            "updated_at": now.isoformat(),
        }
    )


def _provider_status(order: Order) -> str:
    metadata = order.metadata_ if isinstance(order.metadata_, dict) else {}
    value = metadata.get("provider_status")
    if isinstance(value, str) and value.strip():
        return value.strip().lower()
    return str(order.status).strip().lower()


def _sync_order_provider_state(
    order: Order,
    *,
    provider_status: str,
    reject_reason: str | None = None,
    at: datetime | None = None,
) -> None:
    synced_at = at or datetime.now(UTC)
    metadata = dict(order.metadata_) if isinstance(order.metadata_, dict) else {}
    metadata["provider_status"] = provider_status
    metadata["provider_status_updated_at"] = synced_at.isoformat()
    order.metadata_ = metadata
    order.last_sync_at = synced_at
    order.provider_updated_at = synced_at
    if reject_reason is not None:
        text = reject_reason.strip()
        order.reject_reason = text[:255] if text else None


async def _append_order_state_transition(
    db: AsyncSession,
    *,
    order: Order,
    target_status: str,
    reason: str,
    extra_metadata: dict[str, Any] | None = None,
) -> None:
    transition = apply_order_status_transition(
        order,
        target_status=target_status,
        reason=reason,
        extra_metadata=extra_metadata,
    )
    transition_at = datetime.fromisoformat(str(transition["ts"]))
    db.add(
        OrderStateTransition(
            order_id=order.id,
            from_status=str(transition["from"]),
            to_status=str(transition["to"]),
            reason=str(transition["reason"]),
            transitioned_at=transition_at,
            metadata_=extra_metadata or {},
        )
    )


async def _record_signal_event(
    db: AsyncSession,
    *,
    deployment_id: UUID,
    symbol: str,
    timeframe: str,
    signal: str,
    reason: str,
    bar_time: datetime | None,
    metadata: dict[str, Any] | None = None,
) -> UUID:
    row = SignalEvent(
        deployment_id=deployment_id,
        symbol=symbol,
        timeframe=timeframe,
        signal=signal,
        reason=reason[:255],
        bar_time=bar_time or datetime.now(UTC),
        metadata_=metadata or {},
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return row.id


async def _enqueue_position_notification(
    db: AsyncSession,
    *,
    deployment: Deployment,
    order: Order,
    signal: str,
    qty: Decimal,
    fill_price: Decimal,
    reason: str,
    occurred_at: datetime,
    remaining_qty: Decimal | None = None,
    realized_pnl: Decimal | None = None,
) -> None:
    if not settings.notifications_enabled:
        return

    signal_key = str(signal).strip().upper()
    if signal_key == "OPEN_LONG":
        event_type = EVENT_POSITION_OPENED
        event_key = f"position_opened:{order.id}"
        payload = {
            "deployment_id": str(deployment.id),
            "order_id": str(order.id),
            "symbol": order.symbol,
            "side": "long",
            "qty": float(qty),
            "price": float(fill_price),
            "reason": reason,
            "occurred_at": occurred_at.isoformat(),
        }
    elif signal_key == "OPEN_SHORT":
        event_type = EVENT_POSITION_OPENED
        event_key = f"position_opened:{order.id}"
        payload = {
            "deployment_id": str(deployment.id),
            "order_id": str(order.id),
            "symbol": order.symbol,
            "side": "short",
            "qty": float(qty),
            "price": float(fill_price),
            "reason": reason,
            "occurred_at": occurred_at.isoformat(),
        }
    elif signal_key == "CLOSE":
        event_type = EVENT_POSITION_CLOSED
        event_key = f"position_closed:{order.id}"
        payload = {
            "deployment_id": str(deployment.id),
            "order_id": str(order.id),
            "symbol": order.symbol,
            "qty": float(qty),
            "exit_price": float(fill_price),
            "remaining_qty": float(remaining_qty) if remaining_qty is not None else None,
            "realized_pnl_delta": float(realized_pnl) if realized_pnl is not None else None,
            "reason": reason,
            "occurred_at": occurred_at.isoformat(),
        }
    else:
        return

    service = NotificationOutboxService(db)
    await service.enqueue_event_for_user(
        user_id=deployment.user_id,
        event_type=event_type,
        event_key=event_key,
        payload=payload,
    )


async def sync_order_status_from_adapter(
    db: AsyncSession,
    *,
    order: Order,
    adapter: BrokerAdapter,
) -> Order:
    """Sync one order status snapshot from provider and persist transitions."""
    now = datetime.now(UTC)
    if not order.provider_order_id:
        order.last_sync_at = now
        await db.commit()
        await db.refresh(order)
        return order

    state = await adapter.fetch_order(order.provider_order_id)
    if state is None:
        order.last_sync_at = now
        await db.commit()
        await db.refresh(order)
        return order

    provider_status = normalize_order_status(str(state.status))
    _sync_order_provider_state(
        order,
        provider_status=provider_status,
        reject_reason=state.reject_reason,
        at=state.provider_updated_at or now,
    )
    if state.avg_fill_price is not None:
        order.price = Decimal(str(state.avg_fill_price))
    if state.submitted_at is not None:
        order.submitted_at = state.submitted_at

    if provider_status != order.status:
        try:
            await _append_order_state_transition(
                db,
                order=order,
                target_status=provider_status,
                reason="provider_order_sync",
                extra_metadata={"provider_status": provider_status},
            )
        except ValueError:
            # Preserve provider snapshot while keeping local status when transition is invalid.
            pass

    await db.commit()
    await db.refresh(order)
    return order


async def process_deployment_signal_cycle(
    db: AsyncSession,
    *,
    deployment_id: UUID,
) -> ProcessCycleResult:
    lease = await deployment_runtime_lock.acquire(deployment_id)
    if lease is None:
        deployment = await db.scalar(
            select(Deployment)
            .options(
                selectinload(Deployment.strategy),
            )
            .where(Deployment.id == deployment_id)
        )
        event_id: UUID | None = None
        response_metadata: dict[str, Any] = {}
        if deployment is not None and deployment.strategy is not None:
            _, symbol, timeframe = _resolve_scope(deployment)
            event_id = await _record_signal_event(
                db,
                deployment_id=deployment.id,
                symbol=symbol,
                timeframe=timeframe,
                signal="NOOP",
                reason="deployment_locked",
                bar_time=None,
                metadata={},
            )
            response_metadata["execution_event_id"] = str(event_id)
        await _persist_runtime_state(
            deployment_id=deployment_id,
            run=None,
            status="locked",
            reason="deployment_locked",
        )
        return ProcessCycleResult(
            deployment_id=deployment_id,
            signal="NOOP",
            reason="deployment_locked",
            execution_event_id=event_id,
            metadata=response_metadata,
        )

    try:
        deployment = await db.scalar(
            select(Deployment)
            .options(
                selectinload(Deployment.strategy),
                selectinload(Deployment.deployment_runs),
                selectinload(Deployment.positions),
            )
            .where(Deployment.id == deployment_id)
        )
        if deployment is None:
            raise ValueError("Deployment not found.")

        run = _latest_run(deployment)
        if run is None:
            raise ValueError("Deployment run not found.")
        market, symbol, timeframe = _resolve_scope(deployment)
        market_data_adapter: AlpacaTradingAdapter | None = None
        market_data_metadata: dict[str, Any] = {}
        broker_sync_metadata: dict[str, Any] = {}

        async def _build_cycle_result(
            *,
            signal: str,
            reason: str,
            bar_time: datetime | None = None,
            metadata: dict[str, Any] | None = None,
            order_id: UUID | None = None,
            idempotent_hit: bool = False,
        ) -> ProcessCycleResult:
            event_metadata = dict(metadata) if isinstance(metadata, dict) else {}
            if order_id is not None:
                event_metadata.setdefault("order_id", str(order_id))
            event_id = await _record_signal_event(
                db,
                deployment_id=deployment.id,
                symbol=symbol,
                timeframe=timeframe,
                signal=signal,
                reason=reason,
                bar_time=bar_time,
                metadata=event_metadata,
            )
            response_metadata = dict(event_metadata)
            response_metadata["execution_event_id"] = str(event_id)
            return ProcessCycleResult(
                deployment_id=deployment.id,
                signal=signal,
                reason=reason,
                execution_event_id=event_id,
                order_id=order_id,
                idempotent_hit=idempotent_hit,
                bar_time=bar_time,
                metadata=response_metadata,
            )

        def _combine_cycle_metadata(extra: dict[str, Any] | None = None) -> dict[str, Any]:
            payload = dict(market_data_metadata)
            payload.update(broker_sync_metadata)
            if isinstance(extra, dict):
                payload.update(extra)
            return payload

        if deployment.status != "active":
            # Keep runtime market-data subscriptions aligned with deployment lifecycle.
            market_data_runtime.unsubscribe(f"deployment:{deployment.id}")
            await _persist_runtime_state(
                deployment_id=deployment.id,
                run=run,
                status=deployment.status,
                reason=f"deployment_{deployment.status}",
            )
            return await _build_cycle_result(
                signal="NOOP",
                reason=f"deployment_{deployment.status}",
            )

        kill_decision = RuntimeKillSwitch().evaluate(
            user_id=deployment.user_id,
            deployment_id=deployment.id,
        )
        if not kill_decision.allowed:
            run.status = "paused"
            run.runtime_state = {
                **(run.runtime_state if isinstance(run.runtime_state, dict) else {}),
                "kill_switch_reason": kill_decision.reason,
                "last_updated_at": datetime.now(UTC).isoformat(),
            }
            await db.commit()
            await _persist_runtime_state(
                deployment_id=deployment.id,
                run=run,
                status="paused",
                reason=kill_decision.reason,
            )
            return await _build_cycle_result(
                signal="NOOP",
                reason=kill_decision.reason,
            )

        run.status = "running"
        run.runtime_state = {
            **(run.runtime_state if isinstance(run.runtime_state, dict) else {}),
            "last_updated_at": datetime.now(UTC).isoformat(),
        }
        if _should_sync_broker_account(run=run, now=datetime.now(UTC)):
            broker_sync_adapter: BrokerAdapter | None = None
            close_broker_sync_adapter = False
            try:
                # Account sync should not depend on order-execution toggle.
                broker_sync_adapter = await _build_alpaca_adapter_for_market_data(db=db, run=run)
                if broker_sync_adapter is not None:
                    close_broker_sync_adapter = True
                    broker_snapshot = await _sync_broker_account_snapshot(
                        run=run,
                        adapter=broker_sync_adapter,
                    )
                    broker_sync_metadata["broker_account_sync"] = "ok"
                    broker_sync_metadata["broker_account_fetched_at"] = broker_snapshot.get("fetched_at")
                else:
                    broker_sync_metadata["broker_account_sync"] = "skipped_adapter_unavailable"
            except Exception as exc:  # noqa: BLE001
                _mark_broker_account_sync_error(
                    run=run,
                    error=f"broker_account_sync_failed:{type(exc).__name__}",
                )
                broker_sync_metadata["broker_account_sync"] = f"error:{type(exc).__name__}"
            finally:
                if close_broker_sync_adapter and broker_sync_adapter is not None:
                    with suppress(Exception):
                        await broker_sync_adapter.aclose()

        market_data_runtime.subscribe(
            f"deployment:{deployment.id}",
            [symbol],
            market=market,
        )

        bars = market_data_runtime.get_recent_bars(
            market=market,
            symbol=symbol,
            timeframe=timeframe,
            limit=500,
        )
        if (
            settings.effective_market_data_redis_read_enabled
            and settings.effective_market_data_runtime_fail_fast_on_redis_error
            and market_data_runtime.redis_read_error_recent()
        ):
            redis_status = market_data_runtime.redis_data_plane_status()
            fail_fast_metadata = _combine_cycle_metadata(
                {
                    "market_data_fail_fast": "yes",
                    "market_data_source": "redis_first",
                    "redis_data_plane": redis_status,
                }
            )
            run.status = "paused"
            run.runtime_state = {
                **(run.runtime_state if isinstance(run.runtime_state, dict) else {}),
                "market_data_fail_fast_reason": "market_data_redis_unavailable",
                "last_updated_at": datetime.now(UTC).isoformat(),
            }
            await db.commit()
            await _persist_runtime_state(
                deployment_id=deployment.id,
                run=run,
                status="paused",
                reason="market_data_redis_unavailable",
                metadata=fail_fast_metadata,
            )
            return await _build_cycle_result(
                signal="NOOP",
                reason="market_data_redis_unavailable",
                metadata=fail_fast_metadata,
            )
        if not bars:
            bars, market_data_metadata, market_data_adapter = await _seed_runtime_market_data_from_provider(
                db=db,
                run=run,
                market=market,
                symbol=symbol,
                timeframe=timeframe,
                limit=500,
            )
        if not bars:
            no_market_data_metadata = _combine_cycle_metadata()
            await _persist_runtime_state(
                deployment_id=deployment.id,
                run=run,
                status="running",
                reason="no_market_data",
                metadata=no_market_data_metadata or None,
            )
            return await _build_cycle_result(
                signal="NOOP",
                reason="no_market_data",
                metadata=no_market_data_metadata or None,
            )

        await _refresh_symbol_mark_price_and_unrealized(
            db=db,
            deployment_id=deployment.id,
            positions=deployment.positions,
            symbol=symbol,
            mark_price=Decimal(str(bars[-1].close)),
        )

        position_side = _resolve_position_side(deployment.positions, symbol)
        signal_runtime = LiveSignalRuntime()
        decision = signal_runtime.evaluate(
            strategy_payload=deployment.strategy.dsl_payload,
            bars=bars,
            current_position_side=position_side,
        )
        decision_metadata = _combine_cycle_metadata(decision.metadata)
        bar_time = decision.bar_time
        if bar_time is not None:
            run.last_bar_time = bar_time
        run.status = "running"
        run.runtime_state = {
            **(run.runtime_state if isinstance(run.runtime_state, dict) else {}),
            "last_signal": decision.signal,
            "last_signal_reason": decision.reason,
            "last_updated_at": datetime.now(UTC).isoformat(),
        }

        await signal_store.add_async(
            SignalRecord(
                deployment_id=deployment.id,
                signal=decision.signal,
                symbol=symbol,
                timeframe=timeframe,
                bar_time=bar_time or bars[-1].timestamp,
                reason=decision.reason,
                metadata=decision_metadata,
            )
        )

        if decision.signal not in {"OPEN_LONG", "OPEN_SHORT", "CLOSE"}:
            await db.commit()
            await _persist_runtime_state(
                deployment_id=deployment.id,
                run=run,
                status="running",
                reason=decision.reason,
                signal=decision.signal,
                bar_time=bar_time,
                metadata=decision_metadata or None,
            )
            return await _build_cycle_result(
                signal=decision.signal,
                reason=decision.reason,
                bar_time=bar_time,
                metadata=decision_metadata or None,
            )

        last_close = bars[-1].close
        order_qty = float((deployment.risk_limits or {}).get("order_qty", 1.0))
        current_symbol_qty = _resolve_position_qty(deployment.positions, symbol)

        if decision.signal == "CLOSE":
            if current_symbol_qty <= 0:
                await db.commit()
                await _persist_runtime_state(
                    deployment_id=deployment.id,
                    run=run,
                    status="running",
                    reason="no_open_position_to_close",
                    signal="NOOP",
                    bar_time=bar_time,
                )
                return await _build_cycle_result(
                    signal="NOOP",
                    reason="no_open_position_to_close",
                    bar_time=bar_time,
                )
            order_qty = current_symbol_qty

        if decision.signal in {"OPEN_LONG", "OPEN_SHORT"}:
            risk_gate = RiskGate()
            risk_decision = risk_gate.evaluate(
                config=_build_risk_config(
                    deployment.risk_limits if isinstance(deployment.risk_limits, dict) else {}
                ),
                context=RiskContext(
                    cash=float(deployment.capital_allocated),
                    equity=float(deployment.capital_allocated),
                    current_symbol_notional=current_symbol_qty * float(last_close),
                    requested_qty=order_qty,
                    mark_price=float(last_close),
                ),
            )
            if not risk_decision.allowed:
                await db.commit()
                await _persist_runtime_state(
                    deployment_id=deployment.id,
                    run=run,
                    status="running",
                    reason=f"risk_rejected:{risk_decision.reason}",
                    signal="NOOP",
                    bar_time=bar_time,
                )
                return await _build_cycle_result(
                    signal="NOOP",
                    reason=f"risk_rejected:{risk_decision.reason}",
                    bar_time=bar_time,
                )

            preference = await TradingPreferenceService(db).get_view(user_id=deployment.user_id)
            if settings.trading_approval_enabled and preference.open_approval_required:
                approval_service = TradeApprovalService(db)
                approval_side = "short" if decision.signal == "OPEN_SHORT" else "long"
                approval_request, _ = await approval_service.create_or_get_open_request(
                    user_id=deployment.user_id,
                    deployment_id=deployment.id,
                    signal=decision.signal,
                    side=approval_side,
                    symbol=symbol,
                    qty=Decimal(str(order_qty)),
                    mark_price=Decimal(str(last_close)),
                    reason=decision.reason,
                    timeframe=timeframe,
                    bar_time=bar_time,
                    approval_channel=preference.approval_channel,
                    approval_timeout_seconds=preference.approval_timeout_seconds,
                    intent_payload={
                        "signal": decision.signal,
                        "reason": decision.reason,
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "qty": order_qty,
                        "mark_price": float(last_close),
                        "bar_time": bar_time.isoformat() if bar_time is not None else None,
                    },
                )
                approval_reason = (
                    "approval_pending"
                    if approval_request.status == "pending"
                    else f"approval_{approval_request.status}"
                )
                approval_metadata = {
                    "approval_request_id": str(approval_request.id),
                    "approval_status": approval_request.status,
                    "approval_expires_at": approval_request.expires_at.isoformat(),
                    "approval_channel": approval_request.approval_channel,
                    "execution_mode": preference.execution_mode,
                }
                await db.commit()
                await _persist_runtime_state(
                    deployment_id=deployment.id,
                    run=run,
                    status="running",
                    reason=approval_reason,
                    signal="NOOP",
                    bar_time=bar_time,
                    metadata=approval_metadata,
                )
                return await _build_cycle_result(
                    signal="NOOP",
                    reason=approval_reason,
                    bar_time=bar_time,
                    metadata=approval_metadata,
                )

        side = "buy"
        if decision.signal == "OPEN_SHORT":
            side = "sell"
        elif decision.signal == "CLOSE":
            side = "sell" if position_side == "long" else "buy"

        bar_epoch = int((bar_time or bars[-1].timestamp).timestamp())
        client_order_id = f"{deployment.id.hex[:8]}-{bar_epoch}-{decision.signal.lower()}"
        intent = OrderIntent(
            client_order_id=client_order_id,
            symbol=symbol,
            side=side,
            qty=Decimal(str(order_qty)),
            order_type="market",
            metadata={"signal": decision.signal, "reason": decision.reason, "timeframe": timeframe},
        )

        adapter: BrokerAdapter | None
        close_adapter_after_submit = False
        if settings.paper_trading_execute_orders and market_data_adapter is not None:
            adapter = market_data_adapter
        else:
            adapter = await _build_adapter_if_enabled(db=db, run=run)
            close_adapter_after_submit = adapter is not None and adapter is not market_data_adapter
        try:
            submit_result = await _submit_order_with_resilience(
                db=db,
                deployment_id=str(deployment.id),
                intent=intent,
                adapter=adapter,
            )
            if adapter is not None:
                try:
                    await sync_order_status_from_adapter(
                        db,
                        order=submit_result.order,
                        adapter=adapter,
                    )
                except Exception:  # noqa: BLE001
                    # Keep local submit snapshot when immediate provider sync fails.
                    await db.rollback()
        except CircuitBreakerOpenError:
            await db.commit()
            broker_circuit_metadata = _combine_cycle_metadata()
            await _persist_runtime_state(
                deployment_id=deployment.id,
                run=run,
                status="running",
                reason="broker_circuit_open",
                signal="NOOP",
                bar_time=bar_time,
                metadata=broker_circuit_metadata or None,
            )
            return await _build_cycle_result(
                signal="NOOP",
                reason="broker_circuit_open",
                bar_time=bar_time,
                metadata=broker_circuit_metadata or None,
            )
        except Exception as exc:  # noqa: BLE001
            await db.commit()
            detail = str(exc).strip()
            reason = (
                f"order_submit_failed:{type(exc).__name__}:{detail[:240]}"
                if detail
                else f"order_submit_failed:{type(exc).__name__}"
            )
            submit_failed_metadata = _combine_cycle_metadata()
            await _persist_runtime_state(
                deployment_id=deployment.id,
                run=run,
                status="running",
                reason=reason,
                signal="NOOP",
                bar_time=bar_time,
                metadata=submit_failed_metadata or None,
            )
            return await _build_cycle_result(
                signal="NOOP",
                reason=reason,
                bar_time=bar_time,
                metadata=submit_failed_metadata or None,
            )
        finally:
            if close_adapter_after_submit and adapter is not None:
                await adapter.aclose()

        normalized_status = normalize_order_status(str(submit_result.order.status))
        if normalized_status in {"rejected", "canceled", "cancelled"}:
            reason = "order_rejected" if normalized_status == "rejected" else "order_canceled"
            reject_reason = submit_result.order.reject_reason if normalized_status == "rejected" else None
            _sync_order_provider_state(
                submit_result.order,
                provider_status=normalized_status,
                reject_reason=reject_reason,
            )
            rejected_metadata = _combine_cycle_metadata(
                {"order_id": str(submit_result.order.id), "provider_status": normalized_status}
            )
            await db.commit()
            await _persist_runtime_state(
                deployment_id=deployment.id,
                run=run,
                status="running",
                reason=reason,
                signal="NOOP",
                bar_time=bar_time,
                metadata=rejected_metadata,
            )
            return await _build_cycle_result(
                signal="NOOP",
                reason=reason,
                order_id=submit_result.order.id,
                idempotent_hit=submit_result.idempotent_hit,
                bar_time=bar_time,
                metadata=_combine_cycle_metadata({"provider_status": normalized_status}),
            )

        if submit_result.idempotent_hit:
            _sync_order_provider_state(
                submit_result.order,
                provider_status=_provider_status(submit_result.order),
            )
            idempotent_metadata = _combine_cycle_metadata({"idempotent_hit": True})
            await db.commit()
            await _persist_runtime_state(
                deployment_id=deployment.id,
                run=run,
                status="running",
                reason=decision.reason,
                signal=decision.signal,
                bar_time=bar_time,
                metadata=idempotent_metadata,
            )
            return await _build_cycle_result(
                signal=decision.signal,
                reason=decision.reason,
                order_id=submit_result.order.id,
                idempotent_hit=True,
                bar_time=bar_time,
                metadata=idempotent_metadata,
            )

        if adapter is not None and normalized_status != "filled":
            pending_metadata = _combine_cycle_metadata(
                {
                    "order_id": str(submit_result.order.id),
                    "provider_status": _provider_status(submit_result.order),
                }
            )
            await _persist_runtime_state(
                deployment_id=deployment.id,
                run=run,
                status="running",
                reason="order_pending_sync",
                signal=decision.signal,
                bar_time=bar_time,
                metadata=pending_metadata,
            )
            return await _build_cycle_result(
                signal=decision.signal,
                reason="order_pending_sync",
                order_id=submit_result.order.id,
                bar_time=bar_time,
                metadata=_combine_cycle_metadata({"provider_status": _provider_status(submit_result.order)}),
            )

        fill_price = Decimal(str(last_close))
        if submit_result.order.price is not None and Decimal(str(submit_result.order.price)) > 0:
            fill_price = Decimal(str(submit_result.order.price))
        submit_result.order.price = fill_price
        _sync_order_provider_state(
            submit_result.order,
            provider_status="filled",
        )
        await _append_order_state_transition(
            db,
            order=submit_result.order,
            target_status="filled",
            reason="runtime_fill_simulated",
            extra_metadata={"fill_price": str(fill_price), "fill_qty": str(order_qty)},
        )
        fill_row = Fill(
            order_id=submit_result.order.id,
            provider_fill_id=(
                f"sync-{submit_result.order.provider_order_id}-{bar_epoch}"
                if adapter is not None and submit_result.order.provider_order_id
                else f"sim-{submit_result.order.id.hex[:16]}-{bar_epoch}"
            ),
            fill_price=fill_price,
            fill_qty=Decimal(str(order_qty)),
            fee=Decimal("0"),
            filled_at=bar_time or bars[-1].timestamp,
        )
        db.add(fill_row)
        await _apply_position_after_fill(
            db=db,
            deployment_id=deployment.id,
            symbol=symbol,
            signal=decision.signal,
            fill_qty=Decimal(str(order_qty)),
            fill_price=fill_price,
            current_position_side=position_side,
            fill_fee=Decimal("0"),
        )
        updated_position = await db.scalar(
            select(Position).where(
                Position.deployment_id == deployment.id,
                Position.symbol == symbol,
            )
        )
        remaining_qty = (
            Decimal(str(updated_position.qty))
            if updated_position is not None
            else None
        )
        realized_pnl = (
            Decimal(str(updated_position.realized_pnl))
            if updated_position is not None
            else None
        )
        await _enqueue_position_notification(
            db,
            deployment=deployment,
            order=submit_result.order,
            signal=decision.signal,
            qty=Decimal(str(order_qty)),
            fill_price=fill_price,
            reason=decision.reason,
            occurred_at=bar_time or bars[-1].timestamp,
            remaining_qty=remaining_qty,
            realized_pnl=realized_pnl,
        )

        pnl_service = PnlService()
        snapshot = await pnl_service.build_snapshot(db, deployment_id=deployment.id)
        await pnl_service.persist_snapshot(db, snapshot=snapshot)
        success_metadata = _combine_cycle_metadata(
            {"order_id": str(submit_result.order.id), "idempotent_hit": submit_result.idempotent_hit}
        )
        await _persist_runtime_state(
            deployment_id=deployment.id,
            run=run,
            status="running",
            reason=decision.reason,
            signal=decision.signal,
            bar_time=bar_time,
            metadata=success_metadata,
        )

        return await _build_cycle_result(
            signal=decision.signal,
            reason=decision.reason,
            order_id=submit_result.order.id,
            idempotent_hit=submit_result.idempotent_hit,
            bar_time=bar_time,
            metadata=_combine_cycle_metadata({"idempotent_hit": submit_result.idempotent_hit}),
        )
    finally:
        if "market_data_adapter" in locals() and market_data_adapter is not None:
            try:
                await market_data_adapter.aclose()
            except Exception:  # noqa: BLE001
                pass
        await deployment_runtime_lock.release(lease)


async def _apply_position_after_fill(
    *,
    db: AsyncSession,
    deployment_id: UUID,
    symbol: str,
    signal: str,
    fill_qty: Decimal,
    fill_price: Decimal,
    current_position_side: str,
    fill_fee: Decimal = Decimal("0"),
) -> None:
    position = await db.scalar(
        select(Position).where(
            Position.deployment_id == deployment_id,
            Position.symbol == symbol,
        )
    )
    if position is None:
        position = Position(
            deployment_id=deployment_id,
            symbol=symbol,
            side="flat",
            qty=Decimal("0"),
            avg_entry_price=Decimal("0"),
            mark_price=fill_price,
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
        )
        db.add(position)
        await db.flush()

    if signal == "OPEN_LONG":
        previous_qty = Decimal(str(position.qty))
        total_qty = previous_qty + fill_qty
        if total_qty > 0:
            weighted_cost = (Decimal(str(position.avg_entry_price)) * previous_qty) + (fill_price * fill_qty)
            position.avg_entry_price = weighted_cost / total_qty
        position.qty = total_qty
        position.side = "long"
        position.mark_price = fill_price
        position.unrealized_pnl = Decimal("0")
        if fill_fee > 0:
            position.realized_pnl = Decimal(str(position.realized_pnl)) - fill_fee
        return

    if signal == "OPEN_SHORT":
        previous_qty = Decimal(str(position.qty))
        total_qty = previous_qty + fill_qty
        if total_qty > 0:
            weighted_cost = (Decimal(str(position.avg_entry_price)) * previous_qty) + (fill_price * fill_qty)
            position.avg_entry_price = weighted_cost / total_qty
        position.qty = total_qty
        position.side = "short"
        position.mark_price = fill_price
        position.unrealized_pnl = Decimal("0")
        if fill_fee > 0:
            position.realized_pnl = Decimal(str(position.realized_pnl)) - fill_fee
        return

    if signal == "CLOSE":
        existing_qty = Decimal(str(position.qty))
        if existing_qty <= 0:
            return
        close_qty = min(existing_qty, fill_qty)
        if close_qty <= 0:
            return
        remaining_qty = max(existing_qty - close_qty, Decimal("0"))
        realized = Decimal(str(position.realized_pnl))
        avg_entry = Decimal(str(position.avg_entry_price))
        if current_position_side == "short":
            realized += (avg_entry - fill_price) * close_qty
        else:
            realized += (fill_price - avg_entry) * close_qty
        if fill_fee > 0:
            realized -= fill_fee
        position.realized_pnl = realized
        position.qty = remaining_qty
        position.side = "flat" if remaining_qty <= 0 else current_position_side
        if remaining_qty <= 0:
            position.avg_entry_price = Decimal("0")
        position.mark_price = fill_price
        position.unrealized_pnl = Decimal("0")
        return


async def execute_manual_trade_action(
    db: AsyncSession,
    *,
    deployment_id: UUID,
    action: ManualTradeAction,
) -> ManualActionExecutionResult:
    lease = await deployment_runtime_lock.acquire(deployment_id)
    market_data_adapter: AlpacaTradingAdapter | None = None
    if lease is None:
        payload = action.payload if isinstance(action.payload, dict) else {}
        action.status = "rejected"
        action.payload = {
            **payload,
            "_execution": {"status": "rejected", "reason": "deployment_locked"},
        }
        await db.commit()
        await db.refresh(action)
        await _persist_runtime_state(
            deployment_id=deployment_id,
            run=None,
            status="manual_action_rejected",
            reason="deployment_locked",
        )
        return ManualActionExecutionResult(
            action_id=action.id,
            status="rejected",
            reason="deployment_locked",
        )

    try:
        deployment = await db.scalar(
            select(Deployment)
            .options(
                selectinload(Deployment.strategy),
                selectinload(Deployment.deployment_runs),
                selectinload(Deployment.positions),
            )
            .where(Deployment.id == deployment_id)
        )
        if deployment is None:
            raise ValueError("Deployment not found.")

        payload = action.payload if isinstance(action.payload, dict) else {}
        market, default_symbol, timeframe = _resolve_scope(deployment)
        symbol = str(payload.get("symbol") or default_symbol).strip().upper()
        current_position_side = _resolve_position_side(deployment.positions, symbol)
        current_symbol_qty = Decimal(str(_resolve_position_qty(deployment.positions, symbol)))

        async def _reject(reason: str) -> ManualActionExecutionResult:
            action.status = "rejected"
            action.payload = {
                **payload,
                "_execution": {"status": "rejected", "reason": reason},
            }
            await db.commit()
            await db.refresh(action)
            await _persist_runtime_state(
                deployment_id=deployment.id,
                run=run,
                status="manual_action_rejected",
                reason=reason,
            )
            return ManualActionExecutionResult(
                action_id=action.id,
                status="rejected",
                reason=reason,
            )

        run = _latest_run(deployment)
        if run is not None:
            market_data_adapter = await _build_alpaca_adapter_for_market_data(db=db, run=run)
        if action.action == "stop":
            deployment.status = "stopped"
            deployment.stopped_at = datetime.now(UTC)
            if run is not None:
                run.status = "stopped"
            action.status = "completed"
            action.payload = {
                **payload,
                "_execution": {"status": "completed", "reason": "deployment_stopped"},
            }
            await db.commit()
            await db.refresh(action)
            await _persist_runtime_state(
                deployment_id=deployment.id,
                run=run,
                status="stopped",
                reason="deployment_stopped",
            )
            return ManualActionExecutionResult(
                action_id=action.id,
                status="completed",
                reason="deployment_stopped",
            )

        if deployment.status != "active":
            return await _reject(f"deployment_{deployment.status}")

        kill_decision = RuntimeKillSwitch().evaluate(
            user_id=deployment.user_id,
            deployment_id=deployment.id,
        )
        if not kill_decision.allowed:
            return await _reject(kill_decision.reason)

        requested_qty = _to_decimal_or_none(payload.get("qty"))
        if requested_qty is not None and requested_qty <= 0:
            return await _reject("invalid_qty")

        signal: str
        qty: Decimal
        apply_risk = False

        if action.action == "open":
            side_hint = str(payload.get("side") or "long").strip().lower()
            if side_hint in {"long", "buy"}:
                signal = "OPEN_LONG"
            elif side_hint in {"short", "sell"}:
                signal = "OPEN_SHORT"
            else:
                return await _reject("invalid_side")

            qty = requested_qty or Decimal(str((deployment.risk_limits or {}).get("order_qty", 1)))
            if qty <= 0:
                return await _reject("invalid_qty")
            apply_risk = True
        elif action.action in {"close", "reduce"}:
            if current_position_side == "flat" or current_symbol_qty <= 0:
                return await _reject("no_open_position_to_close")
            signal = "CLOSE"
            if action.action == "reduce" and requested_qty is None:
                return await _reject("qty_required_for_reduce")
            qty = requested_qty or current_symbol_qty
            qty = min(qty, current_symbol_qty)
            if qty <= 0:
                return await _reject("invalid_qty")
        else:
            return await _reject("unsupported_action")

        mark_price = await _resolve_mark_price_with_provider_fallback(
            market=market,
            symbol=symbol,
            timeframe=timeframe,
            payload=payload,
            positions=deployment.positions,
            adapter=market_data_adapter,
        )
        if mark_price <= 0:
            return await _reject("invalid_mark_price")

        if apply_risk:
            risk_gate = RiskGate()
            risk_decision = risk_gate.evaluate(
                config=_build_risk_config(deployment.risk_limits if isinstance(deployment.risk_limits, dict) else {}),
                context=RiskContext(
                    cash=float(deployment.capital_allocated),
                    equity=float(deployment.capital_allocated),
                    current_symbol_notional=float(current_symbol_qty * mark_price),
                    requested_qty=float(qty),
                    mark_price=float(mark_price),
                ),
            )
            if not risk_decision.allowed:
                return await _reject(f"risk_rejected:{risk_decision.reason}")

        side = "buy"
        if signal == "OPEN_SHORT":
            side = "sell"
        elif signal == "CLOSE":
            side = "sell" if current_position_side == "long" else "buy"

        intent = OrderIntent(
            client_order_id=f"manual-{action.id.hex}",
            symbol=symbol,
            side=side,
            qty=qty,
            order_type="market",
            metadata={
                "signal": signal,
                "manual_action_id": str(action.id),
                "action": action.action,
            },
        )

        adapter: BrokerAdapter | None
        close_adapter_after_submit = False
        if settings.paper_trading_execute_orders and market_data_adapter is not None:
            adapter = market_data_adapter
        else:
            adapter = await _build_adapter_if_enabled(db=db, run=run) if run is not None else None
            close_adapter_after_submit = adapter is not None and adapter is not market_data_adapter
        try:
            submit_result = await _submit_order_with_resilience(
                db=db,
                deployment_id=str(deployment.id),
                intent=intent,
                adapter=adapter,
            )
        except CircuitBreakerOpenError:
            return await _reject("broker_circuit_open")
        except Exception as exc:  # noqa: BLE001
            detail = str(exc).strip()
            reason = (
                f"order_submit_failed:{type(exc).__name__}:{detail[:240]}"
                if detail
                else f"order_submit_failed:{type(exc).__name__}"
            )
            return await _reject(reason)
        finally:
            if close_adapter_after_submit and adapter is not None:
                await adapter.aclose()

        normalized_status = str(submit_result.order.status).strip().lower()
        if normalized_status in {"rejected", "canceled", "cancelled"}:
            return await _reject("order_rejected" if normalized_status == "rejected" else "order_canceled")

        order_id = submit_result.order.id
        if not submit_result.idempotent_hit:
            fill_fee = _to_decimal_or_none(payload.get("fee")) or Decimal("0")
            fill_row = Fill(
                order_id=order_id,
                provider_fill_id=f"sim-manual-{order_id.hex[:16]}",
                fill_price=mark_price,
                fill_qty=qty,
                fee=fill_fee,
                filled_at=datetime.now(UTC),
            )
            db.add(fill_row)
            submit_result.order.price = mark_price
            _sync_order_provider_state(
                submit_result.order,
                provider_status="filled",
            )
            await _append_order_state_transition(
                db,
                order=submit_result.order,
                target_status="filled",
                reason="manual_action_fill_simulated",
                extra_metadata={"fill_price": str(mark_price), "fill_qty": str(qty)},
            )
            await _apply_position_after_fill(
                db=db,
                deployment_id=deployment.id,
                symbol=symbol,
                signal=signal,
                fill_qty=qty,
                fill_price=mark_price,
                current_position_side=current_position_side,
                fill_fee=fill_fee,
            )
            updated_position = await db.scalar(
                select(Position).where(
                    Position.deployment_id == deployment.id,
                    Position.symbol == symbol,
                )
            )
            remaining_qty = (
                Decimal(str(updated_position.qty))
                if updated_position is not None
                else None
            )
            realized_pnl = (
                Decimal(str(updated_position.realized_pnl))
                if updated_position is not None
                else None
            )
            await _enqueue_position_notification(
                db,
                deployment=deployment,
                order=submit_result.order,
                signal=signal,
                qty=qty,
                fill_price=mark_price,
                reason="manual_action",
                occurred_at=datetime.now(UTC),
                remaining_qty=remaining_qty,
                realized_pnl=realized_pnl,
            )
            pnl_service = PnlService()
            snapshot = await pnl_service.build_snapshot(db, deployment_id=deployment.id)
            await pnl_service.persist_snapshot(db, snapshot=snapshot)

        action.status = "completed"
        action.payload = {
            **payload,
            "_execution": {
                "status": "completed",
                "reason": "ok",
                "symbol": symbol,
                "signal": signal,
                "qty": str(qty),
                "order_id": str(order_id),
                "idempotent_hit": submit_result.idempotent_hit,
            },
        }
        await db.commit()
        await db.refresh(action)
        await _persist_runtime_state(
            deployment_id=deployment.id,
            run=run,
            status="manual_action_completed",
            reason="ok",
            signal=signal,
            metadata={"symbol": symbol, "qty": str(qty), "order_id": str(order_id)},
        )
        return ManualActionExecutionResult(
            action_id=action.id,
            status="completed",
            reason="ok",
            order_id=order_id,
            idempotent_hit=submit_result.idempotent_hit,
            metadata={"signal": signal, "symbol": symbol, "qty": str(qty)},
        )
    finally:
        if market_data_adapter is not None:
            try:
                await market_data_adapter.aclose()
            except Exception:  # noqa: BLE001
                pass
        await deployment_runtime_lock.release(lease)
