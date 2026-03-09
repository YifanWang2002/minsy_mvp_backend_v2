"""Runtime orchestration for one deployment bar-close cycle."""

from __future__ import annotations

import logging
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_object_session
from sqlalchemy.orm import selectinload

from packages.core.events.notification_events import (
    EVENT_POSITION_CLOSED,
    EVENT_POSITION_OPENED,
)
from packages.domain.market_data.runtime import RuntimeBar, market_data_runtime
from packages.domain.notification.services.notification_outbox_service import (
    NotificationOutboxService,
)
from packages.domain.trading.pnl.service import PnlService, PortfolioSnapshot
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
from packages.domain.trading.runtime.position_sizing import (
    PositionSizingRequest,
    resolve_open_position_size,
)
from packages.domain.trading.runtime.risk_gate import RiskConfig, RiskContext, RiskGate
from packages.domain.trading.runtime.signal_runtime import LiveSignalRuntime
from packages.domain.trading.runtime.timeframe_scheduler import (
    normalize_runtime_timeframe,
)
from packages.domain.trading.services.broker_provider_service import (
    BrokerProviderService,
)
from packages.domain.trading.services.trade_approval_service import TradeApprovalService
from packages.domain.trading.services.trading_event_outbox_service import (
    append_trading_event_snapshot,
)
from packages.domain.trading.services.trading_preference_service import (
    TradingPreferenceService,
)
from packages.infra.db.models.deployment import Deployment
from packages.infra.db.models.deployment_run import DeploymentRun
from packages.infra.db.models.fill import Fill
from packages.infra.db.models.manual_trade_action import ManualTradeAction
from packages.infra.db.models.order import Order
from packages.infra.db.models.order_state_transition import OrderStateTransition
from packages.infra.db.models.position import Position
from packages.infra.db.models.signal_event import SignalEvent
from packages.infra.providers.trading.adapters.base import (
    AccountState,
    BrokerAdapter,
    OhlcvBar,
    OrderIntent,
    PositionRecord,
)
from packages.infra.redis.locks.deployment_lock import deployment_runtime_lock
from packages.infra.redis.stores.runtime_state_store import runtime_state_store
from packages.infra.redis.stores.signal_store import SignalRecord, signal_store
from packages.shared_settings.schema.settings import settings

logger = logging.getLogger(__name__)

_MANUAL_ACTION_LOCK_RETRY_MAX_ATTEMPTS = 8
_MANUAL_ACTION_LOCK_RETRY_DELAY_SECONDS = 1


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
class RuntimeAccountBudget:
    cash: Decimal
    equity: Decimal
    source: str


@dataclass(frozen=True, slots=True)
class ExecutionPriceSnapshot:
    mark_price: Decimal
    source: str
    timestamp: datetime | None = None
    bid: Decimal | None = None
    ask: Decimal | None = None
    last: Decimal | None = None


_broker_provider_service = BrokerProviderService()


def _latest_run(deployment: Deployment) -> DeploymentRun | None:
    if not deployment.deployment_runs:
        return None
    return sorted(
        deployment.deployment_runs,
        key=lambda item: item.created_at,
        reverse=True,
    )[0]


def _resolve_scope(deployment: Deployment) -> tuple[str, str, str]:
    payload = (
        deployment.strategy.dsl_payload
        if isinstance(deployment.strategy.dsl_payload, dict)
        else {}
    )
    universe = (
        payload.get("universe", {}) if isinstance(payload.get("universe"), dict) else {}
    )
    market = str(universe.get("market") or "stocks").strip().lower()
    tickers = (
        universe.get("tickers") if isinstance(universe.get("tickers"), list) else []
    )
    symbol = (
        str(tickers[0]).strip().upper()
        if tickers
        else deployment.strategy.symbols[0].strip().upper()
    )
    timeframe = normalize_runtime_timeframe(
        str(payload.get("timeframe") or deployment.strategy.timeframe or "1m"),
    )
    return market, symbol, timeframe


def _canonical_symbol(value: str) -> str:
    return (
        str(value or "")
        .strip()
        .upper()
        .replace("/", "")
        .replace("-", "")
        .replace("_", "")
        .replace(" ", "")
    )


def _base_asset_symbol(value: str) -> str:
    text = str(value or "").strip().upper()
    if "/" in text:
        return text.split("/", 1)[0].strip()
    return ""


def _symbols_match(left: str, right: str) -> bool:
    if left.strip().upper() == right.strip().upper():
        return True
    canonical_left = _canonical_symbol(left)
    canonical_right = _canonical_symbol(right)
    if canonical_left and canonical_right and canonical_left == canonical_right:
        return True
    left_base = _base_asset_symbol(left)
    right_base = _base_asset_symbol(right)
    if (
        left_base
        and canonical_right
        and _canonical_symbol(left_base) == canonical_right
    ):
        return True
    if (
        right_base
        and canonical_left
        and _canonical_symbol(right_base) == canonical_left
    ):
        return True
    return False


def _resolve_position_side(positions: list[Position], symbol: str) -> str:
    for position in positions:
        if not _symbols_match(position.symbol, symbol):
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
        if _symbols_match(position.symbol, symbol):
            return float(position.qty)
    return 0.0


def _resolve_position_row(
    positions: list[Position],
    symbol: str,
) -> Position | None:
    for position in positions:
        if _symbols_match(position.symbol, symbol):
            return position
    return None


def _resolve_position_entry_price(
    positions: list[Position],
    symbol: str,
) -> float | None:
    position = _resolve_position_row(positions, symbol)
    if position is None:
        return None
    entry_price = float(position.avg_entry_price)
    return entry_price if entry_price > 0 else None


def _runtime_positive_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _resolve_managed_exit_prices(
    *,
    run: DeploymentRun | None,
    symbol: str,
    current_position_side: str,
) -> tuple[float | None, float | None]:
    if run is None:
        return None, None
    if current_position_side not in {"long", "short"}:
        return None, None
    state = run.runtime_state if isinstance(run.runtime_state, dict) else {}
    managed_exit = (
        state.get("managed_exit")
        if isinstance(state.get("managed_exit"), dict)
        else {}
    )
    managed_symbol = str(managed_exit.get("symbol") or "").strip().upper()
    if managed_symbol and not _symbols_match(managed_symbol, symbol):
        return None, None
    managed_side = str(managed_exit.get("side") or "").strip().lower()
    if managed_side and managed_side != current_position_side:
        return None, None
    return (
        _runtime_positive_float(managed_exit.get("stop_price")),
        _runtime_positive_float(managed_exit.get("take_price")),
    )


def _set_managed_exit_state(
    *,
    run: DeploymentRun | None,
    symbol: str,
    side: str,
    targets: dict[str, Any],
) -> None:
    if run is None:
        return
    state = dict(run.runtime_state) if isinstance(run.runtime_state, dict) else {}
    payload = dict(targets)
    payload["symbol"] = symbol.strip().upper()
    payload["side"] = side.strip().lower()
    state["managed_exit"] = payload
    run.runtime_state = state


def _clear_managed_exit_state(
    *,
    run: DeploymentRun | None,
) -> None:
    if run is None or not isinstance(run.runtime_state, dict):
        return
    state = dict(run.runtime_state)
    state.pop("managed_exit", None)
    run.runtime_state = state


def _sync_managed_exit_state_from_position(
    *,
    run: DeploymentRun | None,
    deployment: Deployment,
    symbol: str,
    bars: list[RuntimeBar] | None,
    position: Position | None,
) -> None:
    if run is None:
        return
    if position is None:
        _clear_managed_exit_state(run=run)
        return
    qty = Decimal(str(position.qty))
    side = str(position.side).strip().lower()
    if qty <= 0 or side not in {"long", "short"}:
        _clear_managed_exit_state(run=run)
        return
    if not bars:
        return

    strategy_payload = (
        deployment.strategy.dsl_payload
        if deployment.strategy is not None
        and isinstance(deployment.strategy.dsl_payload, dict)
        else {}
    )
    entry_signal = "OPEN_SHORT" if side == "short" else "OPEN_LONG"
    entry_price = _runtime_positive_float(position.avg_entry_price)
    if entry_price is None:
        return
    targets = LiveSignalRuntime().build_managed_exit_targets(
        strategy_payload=strategy_payload,
        bars=bars,
        signal=entry_signal,
        entry_price=entry_price,
    )
    if not targets:
        _clear_managed_exit_state(run=run)
        return
    _set_managed_exit_state(
        run=run,
        symbol=symbol,
        side=side,
        targets=targets,
    )


def _load_cached_bars_for_managed_exits(
    *,
    deployment: Deployment,
    market: str,
    symbol: str,
    timeframe: str,
) -> list[RuntimeBar]:
    strategy_payload = (
        deployment.strategy.dsl_payload
        if deployment.strategy is not None
        and isinstance(deployment.strategy.dsl_payload, dict)
        else {}
    )
    signal_runtime = LiveSignalRuntime()
    required_signal_bars = signal_runtime.required_bars(
        strategy_payload=strategy_payload,
    )
    market_data_limit = max(500, int(required_signal_bars))
    market_data_limit = min(
        market_data_limit,
        max(500, int(settings.market_data_ring_capacity_1m)),
    )
    return market_data_runtime.get_recent_bars(
        market=market,
        symbol=symbol,
        timeframe=timeframe,
        limit=market_data_limit,
    )


def _resolve_provider_position_qty(
    positions: list[PositionRecord], symbol: str
) -> Decimal:
    total = Decimal("0")
    for position in positions:
        if not _symbols_match(position.symbol, symbol):
            continue
        qty = Decimal(str(position.qty))
        if qty <= 0:
            continue
        total += qty
    return total


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


def _normalize_snapshot_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    return value.astimezone(UTC) if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _build_execution_price_metadata(
    snapshot: ExecutionPriceSnapshot | None,
) -> dict[str, Any]:
    if snapshot is None:
        return {}
    if snapshot.mark_price <= 0:
        return {}
    metadata: dict[str, Any] = {
        "submitted_mark_price": str(snapshot.mark_price),
        "execution_price": str(snapshot.mark_price),
        "execution_price_source": snapshot.source,
    }
    if snapshot.timestamp is not None:
        metadata["execution_price_timestamp"] = snapshot.timestamp.isoformat()
        metadata["execution_price_age_seconds"] = max(
            0.0, (datetime.now(UTC) - snapshot.timestamp).total_seconds()
        )
    if snapshot.bid is not None and snapshot.bid > 0:
        metadata["execution_quote_bid"] = str(snapshot.bid)
    if snapshot.ask is not None and snapshot.ask > 0:
        metadata["execution_quote_ask"] = str(snapshot.ask)
    if snapshot.last is not None and snapshot.last > 0:
        metadata["execution_quote_last"] = str(snapshot.last)
    return metadata


def _resolve_quote_mark_price(
    *,
    bid: Decimal | None,
    ask: Decimal | None,
    last: Decimal | None,
) -> Decimal:
    bid_value = bid if bid is not None and bid > 0 else None
    ask_value = ask if ask is not None and ask > 0 else None
    last_value = last if last is not None and last > 0 else None
    if bid_value is not None and ask_value is not None:
        lower = min(bid_value, ask_value)
        upper = max(bid_value, ask_value)
        midpoint = (lower + upper) / Decimal("2")
        if last_value is not None and lower <= last_value <= upper:
            return last_value
        if midpoint > 0:
            return midpoint
    if last_value is not None:
        return last_value
    if bid_value is not None:
        return bid_value
    if ask_value is not None:
        return ask_value
    return Decimal("0")


def _resolve_mark_price_snapshot(
    *,
    market: str,
    symbol: str,
    timeframe: str,
    payload: dict[str, Any],
    positions: list[Position],
) -> ExecutionPriceSnapshot:
    explicit_mark = payload.get("mark_price")
    if explicit_mark is not None:
        try:
            mark = Decimal(str(explicit_mark))
        except (InvalidOperation, ValueError, TypeError):
            mark = Decimal("0")
        if mark > 0:
            return ExecutionPriceSnapshot(
                mark_price=mark,
                source="payload.mark_price",
                timestamp=datetime.now(UTC),
            )
        return ExecutionPriceSnapshot(mark_price=Decimal("0"), source="payload.mark_price.invalid")

    bars = market_data_runtime.get_recent_bars(
        market=market,
        symbol=symbol,
        timeframe=timeframe,
        limit=1,
    )
    if bars:
        mark = Decimal(str(bars[-1].close))
        if mark > 0:
            return ExecutionPriceSnapshot(
                mark_price=mark,
                source=f"runtime.{timeframe}.close",
                timestamp=_normalize_snapshot_timestamp(bars[-1].timestamp),
            )

    for position in positions:
        if not _symbols_match(position.symbol, symbol):
            continue
        mark = Decimal(str(position.mark_price))
        if mark > 0:
            return ExecutionPriceSnapshot(
                mark_price=mark,
                source="position.mark_price",
                timestamp=datetime.now(UTC),
            )
        avg = Decimal(str(position.avg_entry_price))
        if avg > 0:
            return ExecutionPriceSnapshot(
                mark_price=avg,
                source="position.avg_entry_price",
                timestamp=datetime.now(UTC),
            )
    return ExecutionPriceSnapshot(mark_price=Decimal("0"), source="unavailable")


async def _resolve_execution_price_snapshot_with_provider_fallback(
    *,
    market: str,
    symbol: str,
    timeframe: str,
    payload: dict[str, Any],
    positions: list[Position],
    adapter: BrokerAdapter | None,
) -> ExecutionPriceSnapshot:
    direct_snapshot = _resolve_mark_price_snapshot(
        market=market,
        symbol=symbol,
        timeframe=timeframe,
        payload=payload,
        positions=positions,
    )
    if direct_snapshot.source.startswith("payload."):
        return direct_snapshot

    if adapter is not None:
        try:
            quote = await adapter.fetch_latest_quote(symbol)
        except Exception:  # noqa: BLE001
            quote = None
        if quote is not None:
            bid = _to_decimal_or_none(getattr(quote, "bid", None))
            ask = _to_decimal_or_none(getattr(quote, "ask", None))
            last = _to_decimal_or_none(getattr(quote, "last", None))
            quote_mark = _resolve_quote_mark_price(bid=bid, ask=ask, last=last)
            if quote_mark > 0:
                market_data_runtime.upsert_quote(
                    market=market,
                    symbol=symbol,
                    quote=quote,
                )
                return ExecutionPriceSnapshot(
                    mark_price=quote_mark,
                    source="provider.quote",
                    timestamp=_normalize_snapshot_timestamp(getattr(quote, "timestamp", None)),
                    bid=bid,
                    ask=ask,
                    last=last,
                )

        try:
            latest_bar = await adapter.fetch_latest_1m_bar(symbol)
        except Exception:  # noqa: BLE001
            latest_bar = None
        if latest_bar is None:
            try:
                bars_1m = await adapter.fetch_ohlcv_1m(symbol, limit=1)
            except Exception:  # noqa: BLE001
                bars_1m = []
            if bars_1m:
                latest_bar = bars_1m[-1]

        if latest_bar is not None:
            market_data_runtime.ingest_1m_bar(
                market=market,
                symbol=symbol,
                bar=latest_bar,
            )
            close = Decimal(str(latest_bar.close))
            if close > 0:
                return ExecutionPriceSnapshot(
                    mark_price=close,
                    source="provider.latest_1m_bar.close",
                    timestamp=_normalize_snapshot_timestamp(latest_bar.timestamp),
                )

    return direct_snapshot


async def _resolve_mark_price_with_provider_fallback(
    *,
    market: str,
    symbol: str,
    timeframe: str,
    payload: dict[str, Any],
    positions: list[Position],
    adapter: BrokerAdapter | None,
) -> Decimal:
    snapshot = await _resolve_execution_price_snapshot_with_provider_fallback(
        market=market,
        symbol=symbol,
        timeframe=timeframe,
        payload=payload,
        positions=positions,
        adapter=adapter,
    )
    return snapshot.mark_price


def _to_decimal_or_none(value: object) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None


def _resolve_provider_filled_qty(order: Order) -> Decimal | None:
    metadata = order.metadata_ if isinstance(order.metadata_, dict) else {}
    parsed = _to_decimal_or_none(metadata.get("provider_filled_qty"))
    if parsed is None or parsed <= 0:
        return None
    return parsed


def _resolve_provider_fill_fee(order: Order) -> Decimal | None:
    metadata = order.metadata_ if isinstance(order.metadata_, dict) else {}
    for key in ("provider_fee", "provider_fill_fee", "fee"):
        parsed = _to_decimal_or_none(metadata.get(key))
        if parsed is None or parsed < 0:
            continue
        return parsed
    return None


async def _resolve_close_qty_against_broker(
    *,
    requested_qty: Decimal,
    symbol: str,
    adapter: BrokerAdapter | None,
) -> tuple[Decimal, dict[str, Any]]:
    if requested_qty <= 0:
        return Decimal("0"), {"close_qty_source": "local_position"}
    if adapter is None:
        return requested_qty, {"close_qty_source": "local_position"}

    metadata: dict[str, Any] = {
        "close_qty_source": "broker_positions",
        "close_qty_local": str(requested_qty),
    }
    try:
        broker_positions = await adapter.fetch_positions()
    except Exception as exc:  # noqa: BLE001
        metadata["close_qty_sync"] = f"error:{type(exc).__name__}"
        return requested_qty, metadata

    broker_qty = _resolve_provider_position_qty(broker_positions, symbol)
    metadata["close_qty_broker"] = str(broker_qty)
    if broker_qty <= 0:
        metadata["close_qty_reconcile_local_flat"] = True
        return Decimal("0"), metadata
    return min(requested_qty, broker_qty), metadata


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


def _resolve_runtime_account_budget_from_run(
    run: DeploymentRun | None,
) -> RuntimeAccountBudget | None:
    if run is None:
        return None
    state = run.runtime_state if isinstance(run.runtime_state, dict) else {}
    broker_state = (
        state.get("broker_account")
        if isinstance(state.get("broker_account"), dict)
        else {}
    )
    if not broker_state:
        return None
    cash = _to_decimal_or_none(broker_state.get("cash"))
    buying_power = _to_decimal_or_none(broker_state.get("buying_power"))
    equity = _to_decimal_or_none(broker_state.get("equity"))
    available_cash = Decimal("0")
    cash_source = ""
    if cash is not None and cash > 0:
        available_cash = cash
        cash_source = "cash"
    elif buying_power is not None and buying_power > 0:
        available_cash = buying_power
        cash_source = "buying_power"
    resolved_equity = equity if equity is not None and equity > 0 else available_cash
    if available_cash <= 0 and resolved_equity <= 0:
        return None
    source = "runtime_state.broker_account"
    if cash_source:
        source = f"{source}.{cash_source}"
    return RuntimeAccountBudget(
        cash=available_cash,
        equity=resolved_equity,
        source=source,
    )


async def _resolve_runtime_account_budget(
    *,
    db: AsyncSession,
    run: DeploymentRun | None,
    adapter: BrokerAdapter | None,
    fallback_capital: Decimal,
) -> tuple[RuntimeAccountBudget, dict[str, Any]]:
    snapshot_budget = _resolve_runtime_account_budget_from_run(run)
    if snapshot_budget is not None:
        return snapshot_budget, {
            "account_budget_source": snapshot_budget.source,
            "account_budget_cash": str(snapshot_budget.cash),
            "account_budget_equity": str(snapshot_budget.equity),
        }

    probe_adapter = adapter
    close_after_probe = False
    if probe_adapter is None and run is not None:
        probe_adapter, _ = await _build_market_data_adapter_for_run(db=db, run=run)
        close_after_probe = probe_adapter is not None and probe_adapter is not adapter

    try:
        if probe_adapter is not None:
            state = await probe_adapter.fetch_account_state()
            available_cash = (
                state.cash
                if state.cash > 0
                else state.buying_power
                if state.buying_power > 0
                else Decimal("0")
            )
            available_equity = state.equity if state.equity > 0 else available_cash
            if available_cash > 0 or available_equity > 0:
                budget = RuntimeAccountBudget(
                    cash=available_cash,
                    equity=available_equity,
                    source="broker_fetch.account_state",
                )
                return budget, {
                    "account_budget_source": budget.source,
                    "account_budget_cash": str(budget.cash),
                    "account_budget_equity": str(budget.equity),
                }
    except Exception as exc:  # noqa: BLE001
        pass
    finally:
        if close_after_probe and probe_adapter is not None:
            with suppress(Exception):
                await probe_adapter.aclose()

    fallback = fallback_capital if fallback_capital > 0 else Decimal("0")
    budget = RuntimeAccountBudget(
        cash=fallback,
        equity=fallback,
        source="deployment.capital_allocated",
    )
    return budget, {
        "account_budget_source": budget.source,
        "account_budget_cash": str(budget.cash),
        "account_budget_equity": str(budget.equity),
    }


def _should_sync_broker_account(
    *,
    run: DeploymentRun,
    now: datetime,
) -> bool:
    state = run.runtime_state if isinstance(run.runtime_state, dict) else {}
    broker_state = (
        state.get("broker_account")
        if isinstance(state.get("broker_account"), dict)
        else {}
    )
    last_fetched = _parse_iso_datetime(broker_state.get("fetched_at"))
    if last_fetched is None:
        return True
    interval_seconds = max(
        1.0, float(settings.paper_trading_broker_account_sync_interval_seconds)
    )
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
    provider: str,
    account_state: AccountState,
    positions: list[PositionRecord],
    fetched_at: datetime,
) -> dict[str, Any]:
    unrealized_total = sum(
        (position.unrealized_pnl for position in positions), Decimal("0")
    )
    realized_total = sum(
        (position.realized_pnl for position in positions), Decimal("0")
    )
    symbols = sorted(
        {position.symbol.upper() for position in positions if position.symbol}
    )
    return {
        "provider": provider,
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
    provider: str,
    error: str,
) -> None:
    state = run.runtime_state if isinstance(run.runtime_state, dict) else {}
    existing = (
        state.get("broker_account")
        if isinstance(state.get("broker_account"), dict)
        else {}
    )
    payload = dict(existing)
    payload.update(
        {
            "provider": provider,
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
    provider: str,
    adapter: BrokerAdapter,
) -> dict[str, Any]:
    fetched_at = datetime.now(UTC)
    account_state = await adapter.fetch_account_state()
    try:
        positions = await adapter.fetch_positions()
    except Exception:  # noqa: BLE001
        positions = []
    payload = _build_broker_account_payload(
        provider=provider,
        account_state=account_state,
        positions=positions,
        fetched_at=fetched_at,
    )
    _set_broker_account_state(run=run, payload=payload)
    return payload


async def refresh_portfolio_snapshot_for_poll(
    db: AsyncSession,
    *,
    deployment: Deployment,
) -> tuple[PortfolioSnapshot, dict[str, Any] | None]:
    """Recompute one deployment snapshot using live broker marks for UI polling."""
    run = _latest_run(deployment)
    broker_payload = (
        run.runtime_state.get("broker_account")
        if run is not None and isinstance(run.runtime_state, dict)
        else None
    )
    adapter: BrokerAdapter | None = None
    close_adapter_after_refresh = False
    provider = "unknown"

    if run is not None:
        adapter, provider = await _build_market_data_adapter_for_run(db=db, run=run)
        close_adapter_after_refresh = adapter is not None

    try:
        if adapter is not None:
            await _sync_pending_orders_for_deployment(
                db,
                deployment=deployment,
                adapter=adapter,
            )
            if run is not None and _should_sync_broker_account(
                run=run, now=datetime.now(UTC)
            ):
                try:
                    broker_payload = await _sync_broker_account_snapshot(
                        run=run,
                        provider=provider,
                        adapter=adapter,
                    )
                except Exception as exc:  # noqa: BLE001
                    _mark_broker_account_sync_error(
                        run=run,
                        provider=provider,
                        error=f"broker_account_sync_failed:{type(exc).__name__}",
                    )
                    broker_payload = (
                        run.runtime_state.get("broker_account")
                        if isinstance(run.runtime_state, dict)
                        else None
                    )

            market, _, timeframe = _resolve_scope(deployment)
            dirty = False
            positions = (
                await db.scalars(
                    select(Position).where(Position.deployment_id == deployment.id)
                )
            ).all()
            for position in positions:
                qty = Decimal(str(position.qty))
                if qty <= 0:
                    continue
                next_mark = await _resolve_mark_price_with_provider_fallback(
                    market=market,
                    symbol=position.symbol,
                    timeframe=timeframe,
                    payload={},
                    positions=positions,
                    adapter=adapter,
                )
                if next_mark <= 0:
                    continue
                if Decimal(str(position.mark_price)) != next_mark:
                    position.mark_price = next_mark
                    dirty = True
            if dirty:
                await db.flush()
    finally:
        if close_adapter_after_refresh and adapter is not None:
            with suppress(Exception):
                await adapter.aclose()

    snapshot = await PnlService().build_snapshot(db, deployment_id=deployment.id)
    return snapshot, broker_payload if isinstance(broker_payload, dict) else None


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
            float(risk_limits["max_daily_loss"])
            if risk_limits.get("max_daily_loss") is not None
            else None
        ),
    )


def _build_position_sizing_request(
    *,
    deployment: Deployment,
    signal: str,
    mark_price: Decimal,
    budget: RuntimeAccountBudget,
) -> PositionSizingRequest:
    strategy_payload = (
        deployment.strategy.dsl_payload
        if deployment.strategy is not None
        and isinstance(deployment.strategy.dsl_payload, dict)
        else {}
    )
    risk_limits = (
        deployment.risk_limits if isinstance(deployment.risk_limits, dict) else {}
    )
    return PositionSizingRequest(
        strategy_payload=strategy_payload,
        signal=signal,
        mark_price=mark_price,
        account_cash=budget.cash,
        account_equity=budget.equity,
        risk_limits=risk_limits,
        default_position_pct=Decimal(str(settings.paper_trading_default_position_pct)),
    )


async def _build_market_data_adapter_for_run(
    *,
    db: AsyncSession,
    run: DeploymentRun,
) -> tuple[BrokerAdapter | None, str]:
    binding = await _broker_provider_service.build_adapter_binding_for_run(
        db=db, run=run
    )
    return binding.adapter, binding.provider


async def _build_adapter_if_enabled(
    *,
    db: AsyncSession,
    run: DeploymentRun,
) -> BrokerAdapter | None:
    if not settings.paper_trading_execute_orders:
        return None
    adapter, _ = await _build_market_data_adapter_for_run(db=db, run=run)
    return adapter


async def _resolve_close_qty_with_optional_adapter(
    *,
    db: AsyncSession,
    run: DeploymentRun | None,
    market_data_adapter: BrokerAdapter | None,
    symbol: str,
    requested_qty: Decimal,
) -> tuple[Decimal, dict[str, Any]]:
    if not settings.paper_trading_execute_orders:
        return requested_qty, {"close_qty_source": "local_position"}

    adapter = market_data_adapter
    close_after_probe = False
    if adapter is None and run is not None:
        adapter = await _build_adapter_if_enabled(db=db, run=run)
        close_after_probe = adapter is not None and adapter is not market_data_adapter

    try:
        return await _resolve_close_qty_against_broker(
            requested_qty=requested_qty,
            symbol=symbol,
            adapter=adapter,
        )
    finally:
        if close_after_probe and adapter is not None:
            with suppress(Exception):
                await adapter.aclose()


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
) -> tuple[list[RuntimeBar], dict[str, Any], BrokerAdapter | None]:
    metadata: dict[str, Any] = {
        "market_data_fallback": "not_attempted",
        "market_data_source": (
            "redis_first"
            if settings.effective_market_data_redis_read_enabled
            else "runtime_cache"
        ),
    }
    adapter, provider = await _build_market_data_adapter_for_run(db=db, run=run)
    metadata["broker_provider"] = provider
    if adapter is None:
        metadata["market_data_fallback"] = "adapter_unavailable"
        return [], metadata, None

    metadata["market_data_fallback"] = "attempted"
    metadata["market_data_source"] = f"{provider}_rest_fallback"

    bars_1m: list[OhlcvBar] = []
    target_bars = max(2, int(limit))
    try:
        bars_1m = await adapter.fetch_ohlcv_1m(symbol, limit=target_bars)
    except Exception as exc:  # noqa: BLE001
        metadata["market_data_fallback_error"] = type(exc).__name__
        bars_1m = []

    if len(bars_1m) < target_bars:
        lookback_minutes = (
            max(target_bars * 5, 24 * 60)
            if market.strip().lower() == "stocks"
            else max(target_bars * 3, 60)
        )
        lookback_since = datetime.now(UTC) - timedelta(minutes=lookback_minutes)
        try:
            backfill_bars = await adapter.fetch_ohlcv_1m(
                symbol,
                since=lookback_since,
                limit=target_bars,
            )
        except Exception as exc:  # noqa: BLE001
            metadata["market_data_fallback_backfill_error"] = type(exc).__name__
            backfill_bars = []
        if backfill_bars:
            merged_bars: dict[datetime, OhlcvBar] = {
                item.timestamp: item for item in bars_1m
            }
            for item in backfill_bars:
                merged_bars[item.timestamp] = item
            ordered_ts = sorted(merged_bars)
            bars_1m = [merged_bars[item] for item in ordered_ts[-target_bars:]]
            metadata["market_data_fallback_backfill_bars"] = len(backfill_bars)
            metadata["market_data_fallback_backfill_since"] = lookback_since.isoformat()

    latest_bar: OhlcvBar | None = None
    try:
        latest_bar = await adapter.fetch_latest_1m_bar(symbol)
    except Exception as exc:  # noqa: BLE001
        metadata["market_data_fallback_latest_1m_error"] = type(exc).__name__

    if bars_1m:
        historical_latest_1m = max(item.timestamp for item in bars_1m)
        metadata["market_data_fallback_historical_latest_1m"] = (
            historical_latest_1m.isoformat()
        )
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
        existing_1m = market_data_runtime.get_recent_bars(
            market=market,
            symbol=symbol,
            timeframe="1m",
            limit=max(target_bars, len(ordered)),
        )
        merged_1m: dict[datetime, RuntimeBar] = {
            item.timestamp: item for item in existing_1m
        }
        for bar in ordered:
            merged_1m[bar.timestamp] = RuntimeBar(
                timestamp=bar.timestamp,
                open=float(bar.open),
                high=float(bar.high),
                low=float(bar.low),
                close=float(bar.close),
                volume=float(bar.volume),
            )
        hydrated_1m = market_data_runtime.hydrate_bars(
            market=market,
            symbol=symbol,
            timeframe="1m",
            bars=[merged_1m[ts] for ts in sorted(merged_1m)],
        )
        metadata["market_data_fallback_ingested_1m"] = len(ordered)
        metadata["market_data_fallback_hydrated_1m"] = len(hydrated_1m)
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
        if not _symbols_match(position.symbol, target_symbol):
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
    snapshot = await pnl_service.build_snapshot(db, deployment_id=deployment_id)
    await pnl_service.persist_snapshot(db, snapshot=snapshot)


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
    payload = (
        dict(run.runtime_state)
        if run is not None and isinstance(run.runtime_state, dict)
        else {}
    )
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
    db = async_object_session(run) if run is not None else None
    if run is not None:
        run.runtime_state = payload
    if db is not None:
        await db.commit()
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
    if db is not None:
        try:
            await append_trading_event_snapshot(db, deployment_id=deployment_id)
        except Exception:  # noqa: BLE001
            await db.rollback()


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
    filled_qty: Decimal | None = None,
    at: datetime | None = None,
) -> None:
    synced_at = at or datetime.now(UTC)
    metadata = dict(order.metadata_) if isinstance(order.metadata_, dict) else {}
    metadata["provider_status"] = provider_status
    metadata["provider_status_updated_at"] = synced_at.isoformat()
    if filled_qty is not None:
        parsed_qty = _to_decimal_or_none(filled_qty)
        if parsed_qty is not None and parsed_qty >= 0:
            metadata["provider_filled_qty"] = str(parsed_qty)
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
            "remaining_qty": float(remaining_qty)
            if remaining_qty is not None
            else None,
            "realized_pnl_delta": float(realized_pnl)
            if realized_pnl is not None
            else None,
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


async def _existing_fill_state(
    db: AsyncSession,
    *,
    order_id: UUID,
) -> tuple[Decimal, Decimal, int]:
    rows = (
        await db.scalars(
            select(Fill)
            .where(Fill.order_id == order_id)
            .order_by(Fill.filled_at.asc(), Fill.id.asc())
        )
    ).all()
    total_qty = Decimal("0")
    total_fee = Decimal("0")
    for row in rows:
        total_qty += Decimal(str(row.fill_qty))
        total_fee += Decimal(str(row.fee))
    return total_qty, total_fee, len(rows)


def _resolve_order_signal(order: Order) -> str | None:
    metadata = order.metadata_ if isinstance(order.metadata_, dict) else {}
    raw_signal = metadata.get("signal")
    signal = str(raw_signal or "").strip().upper()
    if signal in {"OPEN_LONG", "OPEN_SHORT", "CLOSE"}:
        return signal
    return None


async def _mark_trade_approval_executed_for_order(
    db: AsyncSession,
    *,
    order: Order,
    executed_at: datetime | None = None,
) -> None:
    metadata = order.metadata_ if isinstance(order.metadata_, dict) else {}
    raw_request_id = metadata.get("trade_approval_request_id")
    if raw_request_id is None:
        return
    try:
        request_id = UUID(str(raw_request_id))
    except (TypeError, ValueError):
        return

    request = await db.scalar(
        select(TradeApprovalRequest).where(TradeApprovalRequest.id == request_id)
    )
    if request is None:
        return
    if str(request.status).strip().lower() in {
        "executed",
        "failed",
        "rejected",
        "expired",
        "cancelled",
    }:
        return

    request.status = "executed"
    request.executed_at = executed_at or datetime.now(UTC)
    request.execution_order_id = order.id
    request.execution_error = None


async def _reconcile_provider_fill_for_order(
    db: AsyncSession,
    *,
    deployment: Deployment,
    order: Order,
) -> bool:
    signal = _resolve_order_signal(order)
    if signal is None:
        return False

    total_filled_qty = _resolve_provider_filled_qty(order)
    if total_filled_qty is None or total_filled_qty <= 0:
        return False

    existing_qty, existing_fee, existing_count = await _existing_fill_state(
        db,
        order_id=order.id,
    )
    delta_qty = total_filled_qty - existing_qty
    if delta_qty <= 0:
        return False

    fill_price = _to_decimal_or_none(order.price)
    if fill_price is None or fill_price <= 0:
        return False

    total_fee = _resolve_provider_fill_fee(order) or Decimal("0")
    delta_fee = total_fee - existing_fee if total_fee > existing_fee else Decimal("0")

    current_position = await db.scalar(
        select(Position).where(
            Position.deployment_id == deployment.id,
            Position.symbol == order.symbol,
        )
    )
    current_position_side = (
        str(current_position.side).strip().lower()
        if current_position is not None
        else "flat"
    )
    fill_time = (
        order.provider_updated_at
        or order.last_sync_at
        or order.submitted_at
        or datetime.now(UTC)
    )
    provider_fill_id = (
        f"sync-{order.provider_order_id}-{existing_count + 1}"
        if order.provider_order_id
        else f"sync-{order.id.hex[:16]}-{existing_count + 1}"
    )

    db.add(
        Fill(
            order_id=order.id,
            provider_fill_id=provider_fill_id,
            fill_price=fill_price,
            fill_qty=delta_qty,
            fee=delta_fee,
            filled_at=fill_time,
        )
    )
    await _apply_position_after_fill(
        db=db,
        deployment_id=deployment.id,
        symbol=order.symbol,
        signal=signal,
        fill_qty=delta_qty,
        fill_price=fill_price,
        current_position_side=current_position_side,
        fill_fee=delta_fee,
    )
    updated_position = await db.scalar(
        select(Position).where(
            Position.deployment_id == deployment.id,
            Position.symbol == order.symbol,
        )
    )
    remaining_qty = (
        Decimal(str(updated_position.qty)) if updated_position is not None else None
    )
    realized_pnl = (
        Decimal(str(updated_position.realized_pnl))
        if updated_position is not None
        else None
    )
    metadata = order.metadata_ if isinstance(order.metadata_, dict) else {}
    await _enqueue_position_notification(
        db,
        deployment=deployment,
        order=order,
        signal=signal,
        qty=delta_qty,
        fill_price=fill_price,
        reason=str(metadata.get("reason") or "provider_order_sync"),
        occurred_at=fill_time,
        remaining_qty=remaining_qty,
        realized_pnl=realized_pnl,
    )
    if normalize_order_status(str(order.status)) == "filled":
        await _mark_trade_approval_executed_for_order(
            db,
            order=order,
            executed_at=fill_time,
        )
    return True


async def _sync_pending_orders_for_deployment(
    db: AsyncSession,
    *,
    deployment: Deployment,
    adapter: BrokerAdapter | None,
) -> dict[str, Any]:
    if adapter is None:
        return {
            "pending_order_sync": "adapter_unavailable",
            "pending_orders_checked": 0,
        }

    pending_orders = (
        await db.scalars(
            select(Order)
            .where(
                Order.deployment_id == deployment.id,
                Order.status.in_(
                    ("new", "accepted", "pending_new", "partially_filled")
                ),
            )
            .order_by(Order.submitted_at.asc(), Order.id.asc())
            .limit(20)
        )
    ).all()
    if not pending_orders:
        return {"pending_order_sync": "idle", "pending_orders_checked": 0}

    checked = 0
    status_updates = 0
    fill_updates = 0
    sync_errors = 0
    last_error: str | None = None
    for order in pending_orders:
        checked += 1
        previous_status = normalize_order_status(str(order.status))
        try:
            await sync_order_status_from_adapter(
                db,
                order=order,
                adapter=adapter,
            )
        except Exception as exc:  # noqa: BLE001
            sync_errors += 1
            last_error = type(exc).__name__
            logger.warning(
                "pending order sync failed deployment_id=%s order_id=%s provider_order_id=%s error=%s",
                deployment.id,
                order.id,
                order.provider_order_id,
                type(exc).__name__,
            )
            continue
        current_status = normalize_order_status(str(order.status))
        if current_status != previous_status:
            status_updates += 1
        if current_status in {"partially_filled", "filled"}:
            if await _reconcile_provider_fill_for_order(
                db,
                deployment=deployment,
                order=order,
            ):
                fill_updates += 1
        await reconcile_pending_manual_actions_for_order(
            db,
            deployment_id=deployment.id,
            order=order,
        )

    if fill_updates > 0:
        pnl_service = PnlService()
        snapshot = await pnl_service.build_snapshot(db, deployment_id=deployment.id)
        await pnl_service.persist_snapshot(db, snapshot=snapshot)

    result = {
        "pending_order_sync": "ok" if sync_errors == 0 else "partial_error",
        "pending_orders_checked": checked,
        "pending_order_status_updates": status_updates,
        "pending_order_fill_updates": fill_updates,
    }
    if sync_errors > 0:
        result["pending_order_sync_errors"] = sync_errors
        if last_error is not None:
            result["pending_order_sync_last_error"] = last_error
    return result


async def _sync_pending_orders_for_run(
    db: AsyncSession,
    *,
    deployment: Deployment,
    run: DeploymentRun | None,
) -> dict[str, Any]:
    if not settings.paper_trading_execute_orders or run is None:
        return {}

    adapter = await _build_adapter_if_enabled(db=db, run=run)
    if adapter is None:
        return {
            "pending_order_sync": "adapter_unavailable",
            "pending_orders_checked": 0,
        }

    try:
        return await _sync_pending_orders_for_deployment(
            db,
            deployment=deployment,
            adapter=adapter,
        )
    finally:
        with suppress(Exception):
            await adapter.aclose()


async def sync_pending_orders_for_poll(
    db: AsyncSession,
    *,
    deployment: Deployment,
) -> dict[str, Any]:
    """Lightweight pending-order sync for portfolio polling.

    This path intentionally avoids the heavier account/position refresh used by
    ``refresh_portfolio_snapshot_for_poll`` while still allowing accepted orders
    to advance to partially-filled / filled during UI polling.
    """
    run = _latest_run(deployment)
    if run is None:
        return {
            "pending_order_sync": "run_missing",
            "pending_orders_checked": 0,
        }
    return await _sync_pending_orders_for_run(db, deployment=deployment, run=run)


def _manual_action_linked_order_id(action: ManualTradeAction) -> UUID | None:
    payload = action.payload if isinstance(action.payload, dict) else {}
    execution = (
        payload.get("_execution") if isinstance(payload.get("_execution"), dict) else {}
    )
    order_id_raw = execution.get("order_id")
    if order_id_raw is None:
        return None
    try:
        return UUID(str(order_id_raw))
    except (TypeError, ValueError):
        return None


def _manual_action_terminal_state_for_order(order: Order) -> tuple[str, str] | None:
    normalized = normalize_order_status(str(order.status))
    if normalized == "filled":
        return ("completed", "order_filled")
    if normalized == "rejected":
        return ("rejected", "order_rejected")
    if normalized == "canceled":
        return ("rejected", "order_canceled")
    if normalized == "expired":
        return ("rejected", "order_expired")
    return None


def _build_manual_action_terminal_payload(
    action: ManualTradeAction,
    *,
    order: Order,
    next_status: str,
    reason: str,
) -> dict[str, Any]:
    payload = action.payload if isinstance(action.payload, dict) else {}
    execution = (
        payload.get("_execution") if isinstance(payload.get("_execution"), dict) else {}
    )
    provider_status = _provider_status(order)
    next_execution = {
        **execution,
        "status": next_status,
        "reason": reason,
        "order_id": str(order.id),
        "provider_status": provider_status,
    }
    if order.reject_reason:
        next_execution["reject_reason"] = str(order.reject_reason)
    return {
        **payload,
        "_execution": next_execution,
    }


async def reconcile_manual_action_terminal_state(
    db: AsyncSession,
    *,
    action: ManualTradeAction,
    order: Order | None = None,
) -> bool:
    if action.status not in {"pending", "executing", "accepted"}:
        return False
    linked_order_id = _manual_action_linked_order_id(action)
    if linked_order_id is None:
        return False
    resolved_order = order
    if resolved_order is None:
        resolved_order = await db.scalar(
            select(Order).where(Order.id == linked_order_id)
        )
        if resolved_order is None:
            return False
    terminal = _manual_action_terminal_state_for_order(resolved_order)
    if terminal is None:
        return False
    next_status, reason = terminal
    next_payload = _build_manual_action_terminal_payload(
        action,
        order=resolved_order,
        next_status=next_status,
        reason=reason,
    )
    if action.status == next_status and action.payload == next_payload:
        return False
    action.status = next_status
    action.payload = next_payload
    await db.commit()
    await db.refresh(action)
    await append_trading_event_snapshot(db, deployment_id=action.deployment_id)
    return True


async def reconcile_pending_manual_actions_for_order(
    db: AsyncSession,
    *,
    deployment_id: UUID,
    order: Order,
) -> int:
    terminal = _manual_action_terminal_state_for_order(order)
    if terminal is None:
        return 0
    candidates = (
        await db.scalars(
            select(ManualTradeAction)
            .where(
                ManualTradeAction.deployment_id == deployment_id,
                ManualTradeAction.status.in_(("pending", "executing", "accepted")),
            )
            .order_by(
                ManualTradeAction.updated_at.desc(), ManualTradeAction.created_at.desc()
            )
            .limit(20)
        )
    ).all()
    updated = 0
    for action in candidates:
        if _manual_action_linked_order_id(action) != order.id:
            continue
        if await reconcile_manual_action_terminal_state(
            db,
            action=action,
            order=order,
        ):
            updated += 1
    return updated


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

    state = await adapter.fetch_order(
        order.provider_order_id,
        symbol=order.symbol,
    )
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
        filled_qty=state.filled_qty,
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
        runtime_positions = list(deployment.positions)
        market_data_adapter: BrokerAdapter | None = None
        market_data_metadata: dict[str, Any] = {}
        broker_sync_metadata: dict[str, Any] = {}
        pending_order_metadata: dict[str, Any] = {}

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

        def _combine_cycle_metadata(
            extra: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            payload = dict(market_data_metadata)
            payload.update(broker_sync_metadata)
            payload.update(pending_order_metadata)
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
                (
                    broker_sync_adapter,
                    broker_provider,
                ) = await _build_market_data_adapter_for_run(
                    db=db,
                    run=run,
                )
                if broker_sync_adapter is not None:
                    close_broker_sync_adapter = True
                    broker_snapshot = await _sync_broker_account_snapshot(
                        run=run,
                        provider=broker_provider,
                        adapter=broker_sync_adapter,
                    )
                    broker_sync_metadata["broker_account_sync"] = "ok"
                    broker_sync_metadata["broker_account_fetched_at"] = (
                        broker_snapshot.get("fetched_at")
                    )
                    broker_sync_metadata["broker_provider"] = broker_provider
                else:
                    broker_sync_metadata["broker_account_sync"] = (
                        "skipped_adapter_unavailable"
                    )
                    broker_sync_metadata["broker_provider"] = broker_provider
            except Exception as exc:  # noqa: BLE001
                provider_for_error = broker_sync_metadata.get("broker_provider")
                if (
                    not isinstance(provider_for_error, str)
                    or not provider_for_error.strip()
                ):
                    provider_for_error = "unknown"
                _mark_broker_account_sync_error(
                    run=run,
                    provider=provider_for_error,
                    error=f"broker_account_sync_failed:{type(exc).__name__}",
                )
                broker_sync_metadata["broker_account_sync"] = (
                    f"error:{type(exc).__name__}"
                )
            finally:
                if close_broker_sync_adapter and broker_sync_adapter is not None:
                    with suppress(Exception):
                        await broker_sync_adapter.aclose()

        pending_order_metadata = await _sync_pending_orders_for_run(
            db=db,
            deployment=deployment,
            run=run,
        )
        if pending_order_metadata.get("pending_order_fill_updates"):
            runtime_positions = list(
                (
                    await db.scalars(
                        select(Position).where(Position.deployment_id == deployment.id)
                    )
                ).all()
            )

        market_data_runtime.subscribe(
            f"deployment:{deployment.id}",
            [symbol],
            market=market,
        )
        signal_runtime = LiveSignalRuntime()
        required_signal_bars = signal_runtime.required_bars(
            strategy_payload=deployment.strategy.dsl_payload
            if isinstance(deployment.strategy.dsl_payload, dict)
            else {},
        )
        market_data_limit = max(500, int(required_signal_bars))
        market_data_limit = min(
            market_data_limit,
            max(500, int(settings.market_data_ring_capacity_1m)),
        )

        bars = market_data_runtime.get_recent_bars(
            market=market,
            symbol=symbol,
            timeframe=timeframe,
            limit=market_data_limit,
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
        should_seed_market_data = (
            run.status == "starting"
            or len(bars) < required_signal_bars
        )
        if should_seed_market_data:
            (
                bars,
                market_data_metadata,
                market_data_adapter,
            ) = await _seed_runtime_market_data_from_provider(
                db=db,
                run=run,
                market=market,
                symbol=symbol,
                timeframe=timeframe,
                limit=market_data_limit,
            )
            market_data_metadata.setdefault(
                "market_data_seed_reason",
                "starting" if run.status == "starting" else "insufficient_history",
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
            positions=runtime_positions,
            symbol=symbol,
            mark_price=Decimal(str(bars[-1].close)),
        )

        position_side = _resolve_position_side(runtime_positions, symbol)
        if position_side == "flat":
            _clear_managed_exit_state(run=run)
        position_row = _resolve_position_row(runtime_positions, symbol)
        position_entry_price = _resolve_position_entry_price(runtime_positions, symbol)
        managed_stop_price, managed_take_price = _resolve_managed_exit_prices(
            run=run,
            symbol=symbol,
            current_position_side=position_side,
        )
        if (
            position_side in {"long", "short"}
            and managed_stop_price is None
            and managed_take_price is None
        ):
            _sync_managed_exit_state_from_position(
                run=run,
                deployment=deployment,
                symbol=symbol,
                bars=bars,
                position=position_row,
            )
            managed_stop_price, managed_take_price = _resolve_managed_exit_prices(
                run=run,
                symbol=symbol,
                current_position_side=position_side,
            )
        decision = signal_runtime.evaluate(
            strategy_payload=deployment.strategy.dsl_payload,
            bars=bars,
            current_position_side=position_side,
            current_position_entry_price=position_entry_price,
            current_position_stop_price=managed_stop_price,
            current_position_take_price=managed_take_price,
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
        last_close_decimal = Decimal(str(last_close))
        execution_price_snapshot = await _resolve_execution_price_snapshot_with_provider_fallback(
            market=market,
            symbol=symbol,
            timeframe=timeframe,
            payload={},
            positions=runtime_positions,
            adapter=market_data_adapter,
        )
        execution_mark_price = execution_price_snapshot.mark_price
        if execution_mark_price <= 0:
            execution_mark_price = last_close_decimal
            execution_price_snapshot = ExecutionPriceSnapshot(
                mark_price=execution_mark_price,
                source="strategy.last_close",
                timestamp=bar_time or bars[-1].timestamp,
            )
        order_qty = float((deployment.risk_limits or {}).get("order_qty", 1.0))
        current_symbol_qty = _resolve_position_qty(runtime_positions, symbol)
        open_budget: RuntimeAccountBudget | None = None

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
            local_close_qty = Decimal(str(current_symbol_qty))
            (
                order_qty_decimal,
                close_qty_metadata,
            ) = await _resolve_close_qty_with_optional_adapter(
                db=db,
                run=run,
                market_data_adapter=market_data_adapter,
                symbol=symbol,
                requested_qty=local_close_qty,
            )
            decision_metadata.update(close_qty_metadata)
            if order_qty_decimal <= 0:
                await db.commit()
                await _persist_runtime_state(
                    deployment_id=deployment.id,
                    run=run,
                    status="running",
                    reason="no_open_position_to_close",
                    signal="NOOP",
                    bar_time=bar_time,
                    metadata=decision_metadata or None,
                )
                return await _build_cycle_result(
                    signal="NOOP",
                    reason="no_open_position_to_close",
                    bar_time=bar_time,
                    metadata=decision_metadata or None,
                )
            order_qty = float(order_qty_decimal)

        if decision.signal in {"OPEN_LONG", "OPEN_SHORT"}:
            open_budget, budget_metadata = await _resolve_runtime_account_budget(
                db=db,
                run=run,
                adapter=market_data_adapter,
                fallback_capital=Decimal(str(deployment.capital_allocated)),
            )
            decision_metadata.update(budget_metadata)
            sizing_result = resolve_open_position_size(
                request=_build_position_sizing_request(
                    deployment=deployment,
                    signal=decision.signal,
                    mark_price=execution_mark_price,
                    budget=open_budget,
                )
            )
            decision_metadata.update(
                {
                    "position_sizing_source": sizing_result.source,
                    "position_sizing_mode": sizing_result.sizing_mode,
                    "position_sizing_notional": str(sizing_result.notional),
                }
            )
            if isinstance(sizing_result.metadata, dict):
                for key, value in sizing_result.metadata.items():
                    decision_metadata[f"position_sizing_{key}"] = value
            if sizing_result.qty <= 0:
                zero_reason = str(sizing_result.metadata.get("reason") or "zero_qty")
                await db.commit()
                await _persist_runtime_state(
                    deployment_id=deployment.id,
                    run=run,
                    status="running",
                    reason=f"position_sizing_zero:{zero_reason}",
                    signal="NOOP",
                    bar_time=bar_time,
                    metadata=decision_metadata or None,
                )
                return await _build_cycle_result(
                    signal="NOOP",
                    reason=f"position_sizing_zero:{zero_reason}",
                    bar_time=bar_time,
                    metadata=decision_metadata or None,
                )
            order_qty = float(sizing_result.qty)
            risk_gate = RiskGate()
            risk_decision = risk_gate.evaluate(
                config=_build_risk_config(
                    deployment.risk_limits
                    if isinstance(deployment.risk_limits, dict)
                    else {}
                ),
                context=RiskContext(
                    cash=float(open_budget.cash),
                    equity=float(open_budget.equity),
                    current_symbol_notional=current_symbol_qty * float(
                        execution_mark_price
                    ),
                    requested_qty=order_qty,
                    mark_price=float(execution_mark_price),
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

            preference = await TradingPreferenceService(db).get_view(
                user_id=deployment.user_id
            )
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
                    mark_price=execution_mark_price,
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
                        "mark_price": float(execution_mark_price),
                        "bar_time": bar_time.isoformat()
                        if bar_time is not None
                        else None,
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
        client_order_id = (
            f"{deployment.id.hex[:8]}-{bar_epoch}-{decision.signal.lower()}"
        )
        intent = OrderIntent(
            client_order_id=client_order_id,
            symbol=symbol,
            side=side,
            qty=Decimal(str(order_qty)),
            order_type="market",
            metadata={
                "signal": decision.signal,
                "reason": decision.reason,
                "timeframe": timeframe,
                "market": market,
                **_build_execution_price_metadata(execution_price_snapshot),
            },
        )

        adapter: BrokerAdapter | None
        close_adapter_after_submit = False
        if settings.paper_trading_execute_orders and market_data_adapter is not None:
            adapter = market_data_adapter
        else:
            adapter = await _build_adapter_if_enabled(db=db, run=run)
            close_adapter_after_submit = (
                adapter is not None and adapter is not market_data_adapter
            )
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
                    # Rolling back here expires submit_result.order and can trigger
                    # MissingGreenlet on later attribute reads in this cycle.
                    pass
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
            reason = (
                "order_rejected"
                if normalized_status == "rejected"
                else "order_canceled"
            )
            reject_reason = (
                submit_result.order.reject_reason
                if normalized_status == "rejected"
                else None
            )
            _sync_order_provider_state(
                submit_result.order,
                provider_status=normalized_status,
                reject_reason=reject_reason,
            )
            rejected_metadata = _combine_cycle_metadata(
                {
                    "order_id": str(submit_result.order.id),
                    "provider_status": normalized_status,
                }
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
                metadata=_combine_cycle_metadata(
                    {"provider_status": normalized_status}
                ),
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
                metadata=_combine_cycle_metadata(
                    {"provider_status": _provider_status(submit_result.order)}
                ),
            )

        decision_exit_price = (
            _to_decimal_or_none(decision.metadata.get("exit_price"))
            if isinstance(decision.metadata, dict)
            else None
        )
        fill_price = decision_exit_price or Decimal(str(last_close))
        if (
            submit_result.order.price is not None
            and Decimal(str(submit_result.order.price)) > 0
        ):
            fill_price = Decimal(str(submit_result.order.price))
        submit_result.order.price = fill_price
        resolved_fill_qty = _resolve_provider_filled_qty(
            submit_result.order
        ) or Decimal(str(submit_result.order.qty))
        resolved_fill_fee = _resolve_provider_fill_fee(submit_result.order) or Decimal(
            "0"
        )
        _sync_order_provider_state(
            submit_result.order,
            provider_status="filled",
            filled_qty=resolved_fill_qty,
        )
        await _append_order_state_transition(
            db,
            order=submit_result.order,
            target_status="filled",
            reason="runtime_fill_simulated",
            extra_metadata={
                "fill_price": str(fill_price),
                "fill_qty": str(resolved_fill_qty),
            },
        )
        fill_row = Fill(
            order_id=submit_result.order.id,
            provider_fill_id=(
                f"sync-{submit_result.order.provider_order_id}-{bar_epoch}"
                if adapter is not None and submit_result.order.provider_order_id
                else f"sim-{submit_result.order.id.hex[:16]}-{bar_epoch}"
            ),
            fill_price=fill_price,
            fill_qty=resolved_fill_qty,
            fee=resolved_fill_fee,
            filled_at=bar_time or bars[-1].timestamp,
        )
        db.add(fill_row)
        await _mark_trade_approval_executed_for_order(
            db,
            order=submit_result.order,
            executed_at=bar_time or bars[-1].timestamp,
        )
        await _apply_position_after_fill(
            db=db,
            deployment_id=deployment.id,
            symbol=symbol,
            signal=decision.signal,
            fill_qty=resolved_fill_qty,
            fill_price=fill_price,
            current_position_side=position_side,
            fill_fee=resolved_fill_fee,
        )
        updated_position = await db.scalar(
            select(Position).where(
                Position.deployment_id == deployment.id,
                Position.symbol == symbol,
            )
        )
        _sync_managed_exit_state_from_position(
            run=run,
            deployment=deployment,
            symbol=symbol,
            bars=bars,
            position=updated_position,
        )
        remaining_qty = (
            Decimal(str(updated_position.qty)) if updated_position is not None else None
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
            qty=resolved_fill_qty,
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
            {
                "order_id": str(submit_result.order.id),
                "idempotent_hit": submit_result.idempotent_hit,
                "fill_fee": str(resolved_fill_fee),
            }
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
            metadata=_combine_cycle_metadata(
                {"idempotent_hit": submit_result.idempotent_hit}
            ),
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
            weighted_cost = (Decimal(str(position.avg_entry_price)) * previous_qty) + (
                fill_price * fill_qty
            )
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
            weighted_cost = (Decimal(str(position.avg_entry_price)) * previous_qty) + (
                fill_price * fill_qty
            )
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


async def _reconcile_position_flat_from_broker(
    *,
    db: AsyncSession,
    deployment_id: UUID,
    symbol: str,
) -> None:
    position = await db.scalar(
        select(Position).where(
            Position.deployment_id == deployment_id,
            Position.symbol == symbol,
        )
    )
    if position is None:
        return
    position.qty = Decimal("0")
    position.side = "flat"
    position.avg_entry_price = Decimal("0")
    position.unrealized_pnl = Decimal("0")


async def execute_manual_trade_action(
    db: AsyncSession,
    *,
    deployment_id: UUID,
    action: ManualTradeAction,
) -> ManualActionExecutionResult:
    lease = await deployment_runtime_lock.acquire(deployment_id)
    market_data_adapter: BrokerAdapter | None = None
    if lease is None:
        payload = action.payload if isinstance(action.payload, dict) else {}
        execution = (
            payload.get("_execution")
            if isinstance(payload.get("_execution"), dict)
            else {}
        )
        retry_count_raw = (
            execution.get("lock_retry_count") if isinstance(execution, dict) else None
        )
        try:
            retry_count = int(retry_count_raw) if retry_count_raw is not None else 0
        except (TypeError, ValueError):
            retry_count = 0
        next_retry_count = retry_count + 1
        if next_retry_count <= _MANUAL_ACTION_LOCK_RETRY_MAX_ATTEMPTS:
            action.status = "executing"
            action.payload = {
                **payload,
                "_execution": {
                    **execution,
                    "status": "executing",
                    "reason": "waiting_for_runtime_lock",
                    "retryable": True,
                    "lock_retry_count": next_retry_count,
                    "lock_retry_limit": _MANUAL_ACTION_LOCK_RETRY_MAX_ATTEMPTS,
                    "retry_after_seconds": _MANUAL_ACTION_LOCK_RETRY_DELAY_SECONDS,
                },
            }
            await db.commit()
            await db.refresh(action)
            await _persist_runtime_state(
                deployment_id=deployment_id,
                run=None,
                status="manual_action_waiting",
                reason="deployment_locked",
                metadata={
                    "manual_trade_action_id": str(action.id),
                    "lock_retry_count": next_retry_count,
                    "lock_retry_limit": _MANUAL_ACTION_LOCK_RETRY_MAX_ATTEMPTS,
                },
            )
            return ManualActionExecutionResult(
                action_id=action.id,
                status="deferred",
                reason="deployment_locked",
                metadata={
                    "lock_retry_count": next_retry_count,
                    "lock_retry_limit": _MANUAL_ACTION_LOCK_RETRY_MAX_ATTEMPTS,
                    "retry_after_seconds": _MANUAL_ACTION_LOCK_RETRY_DELAY_SECONDS,
                },
            )
        action.status = "rejected"
        action.payload = {
            **payload,
            "_execution": {
                **execution,
                "status": "rejected",
                "reason": "deployment_locked_timeout",
                "lock_retry_count": retry_count,
                "lock_retry_limit": _MANUAL_ACTION_LOCK_RETRY_MAX_ATTEMPTS,
            },
        }
        await db.commit()
        await db.refresh(action)
        await _persist_runtime_state(
            deployment_id=deployment_id,
            run=None,
            status="manual_action_rejected",
            reason="deployment_locked_timeout",
            metadata={
                "manual_trade_action_id": str(action.id),
                "lock_retry_count": retry_count,
                "lock_retry_limit": _MANUAL_ACTION_LOCK_RETRY_MAX_ATTEMPTS,
            },
        )
        return ManualActionExecutionResult(
            action_id=action.id,
            status="rejected",
            reason="deployment_locked_timeout",
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
        current_symbol_qty = Decimal(
            str(_resolve_position_qty(deployment.positions, symbol))
        )

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

        async def _complete_without_order(
            reason: str,
            *,
            metadata: dict[str, Any] | None = None,
        ) -> ManualActionExecutionResult:
            action.status = "completed"
            action.payload = {
                **payload,
                "_execution": {"status": "completed", "reason": reason},
            }
            await db.commit()
            await db.refresh(action)
            await _persist_runtime_state(
                deployment_id=deployment.id,
                run=run,
                status="manual_action_completed",
                reason=reason,
                metadata=metadata,
            )
            return ManualActionExecutionResult(
                action_id=action.id,
                status="completed",
                reason=reason,
                metadata=metadata or {},
            )

        run = _latest_run(deployment)
        if run is not None:
            market_data_adapter, _ = await _build_market_data_adapter_for_run(
                db=db, run=run
            )
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
            if action.action == "open":
                return await _reject(f"deployment_{deployment.status}")

        requested_qty = _to_decimal_or_none(payload.get("qty"))
        if requested_qty is not None and requested_qty <= 0:
            return await _reject("invalid_qty")

        signal: str
        qty: Decimal
        apply_risk = False
        open_budget: RuntimeAccountBudget | None = None
        mark_price: Decimal | None = None
        execution_price_snapshot: ExecutionPriceSnapshot | None = None

        if action.action == "open":
            kill_decision = RuntimeKillSwitch().evaluate(
                user_id=deployment.user_id,
                deployment_id=deployment.id,
            )
            if not kill_decision.allowed:
                return await _reject(kill_decision.reason)
            side_hint = str(payload.get("side") or "long").strip().lower()
            if side_hint in {"long", "buy"}:
                signal = "OPEN_LONG"
            elif side_hint in {"short", "sell"}:
                signal = "OPEN_SHORT"
            else:
                return await _reject("invalid_side")

            execution_price_snapshot = (
                await _resolve_execution_price_snapshot_with_provider_fallback(
                    market=market,
                    symbol=symbol,
                    timeframe=timeframe,
                    payload=payload,
                    positions=deployment.positions,
                    adapter=market_data_adapter,
                )
            )
            mark_price = execution_price_snapshot.mark_price
            if mark_price <= 0:
                return await _reject("invalid_mark_price")
            open_budget, budget_metadata = await _resolve_runtime_account_budget(
                db=db,
                run=run,
                adapter=market_data_adapter,
                fallback_capital=Decimal(str(deployment.capital_allocated)),
            )
            if requested_qty is not None:
                qty = requested_qty
            else:
                sizing_result = resolve_open_position_size(
                    request=_build_position_sizing_request(
                        deployment=deployment,
                        signal=signal,
                        mark_price=mark_price,
                        budget=open_budget,
                    )
                )
                payload = {
                    **payload,
                    "_position_sizing": {
                        "source": sizing_result.source,
                        "mode": sizing_result.sizing_mode,
                        "notional": str(sizing_result.notional),
                        **(
                            dict(sizing_result.metadata)
                            if isinstance(sizing_result.metadata, dict)
                            else {}
                        ),
                    },
                    "_account_budget": budget_metadata,
                }
                qty = sizing_result.qty
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
            qty, close_qty_metadata = await _resolve_close_qty_with_optional_adapter(
                db=db,
                run=run,
                market_data_adapter=market_data_adapter,
                symbol=symbol,
                requested_qty=qty,
            )
            if isinstance(close_qty_metadata, dict) and close_qty_metadata:
                payload = {
                    **payload,
                    "_close_qty": close_qty_metadata,
                }
            if qty <= 0:
                reconcile_local_flat = (
                    isinstance(close_qty_metadata, dict)
                    and close_qty_metadata.get("close_qty_reconcile_local_flat") is True
                )
                if reconcile_local_flat:
                    await _reconcile_position_flat_from_broker(
                        db=db,
                        deployment_id=deployment.id,
                        symbol=symbol,
                    )
                    return await _complete_without_order(
                        "already_flat_on_broker",
                        metadata=(
                            dict(close_qty_metadata)
                            if isinstance(close_qty_metadata, dict)
                            else None
                        ),
                    )
                return await _reject("no_open_position_to_close")
        else:
            return await _reject("unsupported_action")

        if mark_price is None:
            execution_price_snapshot = (
                await _resolve_execution_price_snapshot_with_provider_fallback(
                    market=market,
                    symbol=symbol,
                    timeframe=timeframe,
                    payload=payload,
                    positions=deployment.positions,
                    adapter=market_data_adapter,
                )
            )
            mark_price = execution_price_snapshot.mark_price
        if mark_price <= 0:
            return await _reject("invalid_mark_price")

        if execution_price_snapshot is None:
            execution_price_snapshot = ExecutionPriceSnapshot(
                mark_price=mark_price,
                source="manual.mark_price",
                timestamp=datetime.now(UTC),
            )

        if apply_risk:
            risk_gate = RiskGate()
            risk_budget = open_budget
            if risk_budget is None:
                risk_budget = RuntimeAccountBudget(
                    cash=Decimal(str(deployment.capital_allocated)),
                    equity=Decimal(str(deployment.capital_allocated)),
                    source="deployment.capital_allocated",
                )
            risk_decision = risk_gate.evaluate(
                config=_build_risk_config(
                    deployment.risk_limits
                    if isinstance(deployment.risk_limits, dict)
                    else {}
                ),
                context=RiskContext(
                    cash=float(risk_budget.cash),
                    equity=float(risk_budget.equity),
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
                "market": market,
                **_build_execution_price_metadata(execution_price_snapshot),
                **(
                    {
                        "trade_approval_request_id": str(
                            payload["trade_approval_request_id"]
                        )
                    }
                    if payload.get("trade_approval_request_id") is not None
                    else {}
                ),
            },
        )

        adapter: BrokerAdapter | None
        close_adapter_after_submit = False
        if settings.paper_trading_execute_orders and market_data_adapter is not None:
            adapter = market_data_adapter
        else:
            adapter = (
                await _build_adapter_if_enabled(db=db, run=run)
                if run is not None
                else None
            )
            close_adapter_after_submit = (
                adapter is not None and adapter is not market_data_adapter
            )
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
            return await _reject(
                "order_rejected"
                if normalized_status == "rejected"
                else "order_canceled"
            )

        order_id = submit_result.order.id
        if adapter is not None and normalized_status != "filled":
            if normalized_status == "partially_filled":
                if await _reconcile_provider_fill_for_order(
                    db,
                    deployment=deployment,
                    order=submit_result.order,
                ):
                    pnl_service = PnlService()
                    snapshot = await pnl_service.build_snapshot(
                        db, deployment_id=deployment.id
                    )
                    await pnl_service.persist_snapshot(db, snapshot=snapshot)
            provider_status = _provider_status(submit_result.order)
            action.status = "accepted"
            action.payload = {
                **payload,
                "_execution": {
                    "status": "accepted",
                    "reason": "order_pending_sync",
                    "symbol": symbol,
                    "signal": signal,
                    "qty": str(qty),
                    "order_id": str(order_id),
                    "idempotent_hit": submit_result.idempotent_hit,
                    "provider_status": provider_status,
                },
            }
            await db.commit()
            await db.refresh(action)
            await _persist_runtime_state(
                deployment_id=deployment.id,
                run=run,
                status="manual_action_accepted",
                reason="order_pending_sync",
                signal=signal,
                metadata={
                    "symbol": symbol,
                    "qty": str(qty),
                    "order_id": str(order_id),
                    "provider_status": provider_status,
                },
            )
            return ManualActionExecutionResult(
                action_id=action.id,
                status="accepted",
                reason="order_pending_sync",
                order_id=order_id,
                idempotent_hit=submit_result.idempotent_hit,
                metadata={
                    "signal": signal,
                    "symbol": symbol,
                    "qty": str(qty),
                    "provider_status": provider_status,
                },
            )

        execution_qty = Decimal(str(submit_result.order.qty))
        execution_fill_price = mark_price
        if submit_result.order.price is not None:
            provider_fill_price = Decimal(str(submit_result.order.price))
            if provider_fill_price > 0:
                execution_fill_price = provider_fill_price
        if not submit_result.idempotent_hit:
            fill_fee = (
                _to_decimal_or_none(payload.get("fee"))
                or _resolve_provider_fill_fee(submit_result.order)
                or Decimal("0")
            )
            resolved_fill_qty = _resolve_provider_filled_qty(
                submit_result.order
            ) or Decimal(str(submit_result.order.qty))
            execution_qty = resolved_fill_qty
            fill_row = Fill(
                order_id=order_id,
                provider_fill_id=f"sim-manual-{order_id.hex[:16]}",
                fill_price=execution_fill_price,
                fill_qty=resolved_fill_qty,
                fee=fill_fee,
                filled_at=datetime.now(UTC),
            )
            db.add(fill_row)
            submit_result.order.price = execution_fill_price
            _sync_order_provider_state(
                submit_result.order,
                provider_status="filled",
                filled_qty=resolved_fill_qty,
            )
            await _append_order_state_transition(
                db,
                order=submit_result.order,
                target_status="filled",
                reason="manual_action_fill_confirmed"
                if adapter is not None
                else "manual_action_fill_simulated",
                extra_metadata={
                    "fill_price": str(execution_fill_price),
                    "fill_qty": str(resolved_fill_qty),
                },
            )
            await _mark_trade_approval_executed_for_order(
                db,
                order=submit_result.order,
                executed_at=datetime.now(UTC),
            )
            await _apply_position_after_fill(
                db=db,
                deployment_id=deployment.id,
                symbol=symbol,
                signal=signal,
                fill_qty=resolved_fill_qty,
                fill_price=execution_fill_price,
                current_position_side=current_position_side,
                fill_fee=fill_fee,
            )
            updated_position = await db.scalar(
                select(Position).where(
                    Position.deployment_id == deployment.id,
                    Position.symbol == symbol,
                )
            )
            managed_exit_bars = _load_cached_bars_for_managed_exits(
                deployment=deployment,
                market=market,
                symbol=symbol,
                timeframe=timeframe,
            )
            _sync_managed_exit_state_from_position(
                run=run,
                deployment=deployment,
                symbol=symbol,
                bars=managed_exit_bars,
                position=updated_position,
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
                qty=resolved_fill_qty,
                fill_price=execution_fill_price,
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
                "qty": str(execution_qty),
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
            metadata={
                "symbol": symbol,
                "qty": str(execution_qty),
                "order_id": str(order_id),
            },
        )
        return ManualActionExecutionResult(
            action_id=action.id,
            status="completed",
            reason="ok",
            order_id=order_id,
            idempotent_hit=submit_result.idempotent_hit,
            metadata={"signal": signal, "symbol": symbol, "qty": str(execution_qty)},
        )
    finally:
        if market_data_adapter is not None:
            try:
                await market_data_adapter.aclose()
            except Exception:  # noqa: BLE001
                pass
        await deployment_runtime_lock.release(lease)
