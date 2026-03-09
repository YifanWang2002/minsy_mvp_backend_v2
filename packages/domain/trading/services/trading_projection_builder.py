"""Shared trading projection builder for stream/outbox payloads."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from packages.domain.trading.pnl.service import PortfolioSnapshot
from packages.infra.db.models.deployment import Deployment
from packages.infra.db.models.deployment_run import DeploymentRun
from packages.infra.db.models.fill import Fill
from packages.infra.db.models.manual_trade_action import ManualTradeAction
from packages.infra.db.models.order import Order
from packages.infra.db.models.position import Position
from packages.infra.db.models.trade_approval_request import TradeApprovalRequest


@dataclass(frozen=True, slots=True)
class TradingProjectionEntities:
    """Database rows needed to construct one projection payload."""

    run: DeploymentRun | None
    runtime_state: dict[str, Any]
    scheduler: dict[str, Any]
    positions: list[Position]
    orders: list[Order]
    fills: list[Fill]
    approvals: list[TradeApprovalRequest]
    manual_actions: list[ManualTradeAction]


def _as_optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_iso_or_none(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    text = str(value).strip()
    return text or None


def _as_optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _latest_run(deployment: Deployment) -> DeploymentRun | None:
    if not deployment.deployment_runs:
        return None
    return sorted(
        deployment.deployment_runs,
        key=lambda item: item.created_at,
        reverse=True,
    )[0]


def extract_broker_account_payload(
    *,
    runtime_state: dict[str, Any],
    broker_payload: dict[str, Any] | None = None,
) -> dict[str, object] | None:
    raw = broker_payload if isinstance(broker_payload, dict) else runtime_state.get(
        "broker_account"
    )
    if not isinstance(raw, dict):
        return None

    provider = str(raw.get("provider") or "").strip().lower()
    source = str(raw.get("source") or "").strip()
    sync_status = str(raw.get("sync_status") or "").strip()
    if not provider or not source or not sync_status:
        return None

    symbols_raw = raw.get("symbols")
    symbols = (
        [str(item).upper() for item in symbols_raw if isinstance(item, str)]
        if isinstance(symbols_raw, list)
        else []
    )
    positions_count = raw.get("positions_count")
    try:
        parsed_positions_count = (
            int(positions_count) if positions_count is not None else None
        )
    except (TypeError, ValueError):
        parsed_positions_count = None

    return {
        "provider": provider,
        "source": source,
        "sync_status": sync_status,
        "fetched_at": _as_iso_or_none(raw.get("fetched_at")),
        "equity": _as_optional_float(raw.get("equity")),
        "cash": _as_optional_float(raw.get("cash")),
        "buying_power": _as_optional_float(raw.get("buying_power")),
        "margin_used": _as_optional_float(raw.get("margin_used")),
        "unrealized_pnl": _as_optional_float(raw.get("unrealized_pnl")),
        "realized_pnl": _as_optional_float(raw.get("realized_pnl")),
        "positions_count": parsed_positions_count,
        "symbols": symbols,
        "error": str(raw.get("error"))[:500] if raw.get("error") is not None else None,
        "updated_at": _as_iso_or_none(raw.get("updated_at")),
    }


def _serialize_order(order: Order) -> dict[str, object]:
    metadata = order.metadata_ if isinstance(order.metadata_, dict) else {}
    provider_status = (
        str(metadata.get("provider_status"))
        if metadata.get("provider_status") is not None
        else str(order.status)
    )
    display_price = float(order.price) if order.price is not None else None
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
    return {
        "order_id": str(order.id),
        "symbol": order.symbol,
        "side": order.side,
        "type": order.type,
        "qty": float(order.qty),
        "price": display_price,
        "status": order.status,
        "provider_status": provider_status,
        "reject_reason": order.reject_reason,
        "last_sync_at": order.last_sync_at.isoformat() if order.last_sync_at else None,
        "submitted_at": order.submitted_at.isoformat(),
        "metadata": metadata,
    }


async def load_projection_entities(
    db: AsyncSession,
    *,
    deployment: Deployment,
) -> TradingProjectionEntities:
    run = _latest_run(deployment)
    runtime_state = (
        run.runtime_state if run is not None and isinstance(run.runtime_state, dict) else {}
    )
    scheduler = (
        runtime_state.get("scheduler")
        if isinstance(runtime_state.get("scheduler"), dict)
        else {}
    )
    positions = (
        await db.scalars(
            select(Position)
            .where(Position.deployment_id == deployment.id)
            .order_by(Position.updated_at.desc())
        )
    ).all()
    orders = (
        await db.scalars(
            select(Order)
            .where(Order.deployment_id == deployment.id)
            .order_by(Order.submitted_at.desc(), Order.id.desc())
            .limit(200)
        )
    ).all()
    fills = (
        await db.scalars(
            select(Fill)
            .join(Order, Order.id == Fill.order_id)
            .where(Order.deployment_id == deployment.id)
            .order_by(Fill.filled_at.desc(), Fill.id.desc())
            .limit(200)
        )
    ).all()
    approvals = (
        await db.scalars(
            select(TradeApprovalRequest)
            .where(TradeApprovalRequest.deployment_id == deployment.id)
            .order_by(TradeApprovalRequest.requested_at.desc(), TradeApprovalRequest.id.desc())
            .limit(50)
        )
    ).all()
    manual_actions = (
        await db.scalars(
            select(ManualTradeAction)
            .where(ManualTradeAction.deployment_id == deployment.id)
            .order_by(
                ManualTradeAction.updated_at.desc(),
                ManualTradeAction.created_at.desc(),
                ManualTradeAction.id.desc(),
            )
            .limit(50)
        )
    ).all()
    return TradingProjectionEntities(
        run=run,
        runtime_state=runtime_state,
        scheduler=scheduler,
        positions=list(positions),
        orders=list(orders),
        fills=list(fills),
        approvals=list(approvals),
        manual_actions=list(manual_actions),
    )


def build_projection_payload(
    *,
    deployment: Deployment,
    entities: TradingProjectionEntities,
    snapshot: PortfolioSnapshot,
    pnl_source: str = "platform_estimate",
    broker_account: dict[str, object] | None = None,
) -> dict[str, object]:
    run = entities.run
    return {
        "deployment_id": str(deployment.id),
        "status": deployment.status,
        "run": (
            {
                "deployment_run_id": str(run.id),
                "status": run.status,
                "last_bar_time": run.last_bar_time.isoformat()
                if run.last_bar_time is not None
                else None,
                "runtime_state": entities.runtime_state,
                "timeframe_seconds": _as_optional_int(
                    entities.scheduler.get("timeframe_seconds")
                ),
                "last_trigger_bucket": _as_optional_int(
                    entities.scheduler.get("last_trigger_bucket")
                ),
                "last_enqueued_at": _as_iso_or_none(
                    entities.scheduler.get("last_enqueued_at")
                ),
            }
            if run is not None
            else None
        ),
        "pnl": {
            "equity": float(snapshot.equity),
            "cash": float(snapshot.cash),
            "margin_used": float(snapshot.margin_used),
            "unrealized_pnl": float(snapshot.unrealized_pnl),
            "realized_pnl": float(snapshot.realized_pnl),
            "snapshot_time": snapshot.snapshot_time.isoformat(),
        },
        "pnl_source": pnl_source,
        "broker_account": broker_account,
        "positions": [
            {
                "symbol": position.symbol,
                "side": position.side,
                "qty": float(position.qty),
                "avg_entry_price": float(position.avg_entry_price),
                "mark_price": float(position.mark_price),
                "unrealized_pnl": float(position.unrealized_pnl),
                "realized_pnl": float(position.realized_pnl),
            }
            for position in entities.positions
        ],
        "orders": [_serialize_order(order) for order in entities.orders],
        "fills": [
            {
                "fill_id": str(fill.id),
                "order_id": str(fill.order_id),
                "provider_fill_id": fill.provider_fill_id,
                "fill_price": float(fill.fill_price),
                "fill_qty": float(fill.fill_qty),
                "fee": float(fill.fee),
                "filled_at": fill.filled_at.isoformat(),
            }
            for fill in entities.fills
        ],
        "approvals": [
            {
                "trade_approval_request_id": str(approval.id),
                "deployment_id": str(approval.deployment_id),
                "signal": approval.signal,
                "side": approval.side,
                "symbol": approval.symbol,
                "qty": float(approval.qty),
                "mark_price": float(approval.mark_price),
                "reason": approval.reason,
                "timeframe": approval.timeframe,
                "status": approval.status,
                "requested_at": approval.requested_at.isoformat(),
                "expires_at": approval.expires_at.isoformat(),
                "approved_at": approval.approved_at.isoformat()
                if approval.approved_at
                else None,
                "rejected_at": approval.rejected_at.isoformat()
                if approval.rejected_at
                else None,
                "expired_at": approval.expired_at.isoformat()
                if approval.expired_at
                else None,
                "executed_at": approval.executed_at.isoformat()
                if approval.executed_at
                else None,
                "approved_via": approval.approved_via,
                "decision_actor": approval.decision_actor,
                "execution_error": approval.execution_error,
            }
            for approval in entities.approvals
        ],
        "manual_actions": [
            {
                "manual_trade_action_id": str(action.id),
                "deployment_id": str(action.deployment_id),
                "action": action.action,
                "status": action.status,
                "payload": action.payload if isinstance(action.payload, dict) else {},
                "created_at": action.created_at.isoformat(),
                "updated_at": action.updated_at.isoformat(),
            }
            for action in entities.manual_actions
        ],
    }
