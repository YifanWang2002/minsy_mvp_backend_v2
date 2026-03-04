"""Persistent trading-event snapshot emission for deployment streams."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from packages.domain.trading.pnl.service import PnlService, PortfolioSnapshot
from packages.infra.db.models.deployment import Deployment
from packages.infra.db.models.deployment_run import DeploymentRun
from packages.infra.db.models.fill import Fill
from packages.infra.db.models.order import Order
from packages.infra.db.models.pnl_snapshot import PnlSnapshot
from packages.infra.db.models.position import Position
from packages.infra.db.models.trade_approval_request import TradeApprovalRequest
from packages.infra.db.models.trading_event_outbox import TradingEventOutbox

_EVENT_TYPES: tuple[str, ...] = (
    "deployment_status",
    "order_update",
    "fill_update",
    "position_update",
    "pnl_update",
    "trade_approval_update",
)
_OUTBOX_RETENTION_PER_DEPLOYMENT = 2000


@dataclass(frozen=True, slots=True)
class TradingEventSnapshot:
    """Serializable snapshot backing deployment stream outbox rows."""

    payload: dict[str, object]
    pnl_snapshot: PortfolioSnapshot
    pnl_source: str
    broker_account: dict[str, object] | None


def _as_optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


def _extract_broker_account_payload(
    runtime_state: dict[str, object],
) -> dict[str, object] | None:
    raw = runtime_state.get("broker_account")
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
    try:
        positions_count = (
            int(raw.get("positions_count"))
            if raw.get("positions_count") is not None
            else None
        )
    except (TypeError, ValueError):
        positions_count = None

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
        "positions_count": positions_count,
        "symbols": symbols,
        "error": str(raw.get("error"))[:500] if raw.get("error") is not None else None,
        "updated_at": _as_iso_or_none(raw.get("updated_at")),
    }


def _latest_run(deployment: Deployment) -> DeploymentRun | None:
    if not deployment.deployment_runs:
        return None
    return sorted(
        deployment.deployment_runs, key=lambda item: item.created_at, reverse=True
    )[0]


def _broker_metrics_or_snapshot(
    *,
    snapshot: PortfolioSnapshot,
    broker_account: dict[str, object] | None,
) -> tuple[dict[str, float | str], str]:
    return (
        {
            "equity": float(snapshot.equity),
            "cash": float(snapshot.cash),
            "margin_used": float(snapshot.margin_used),
            "unrealized_pnl": float(snapshot.unrealized_pnl),
            "realized_pnl": float(snapshot.realized_pnl),
            "snapshot_time": snapshot.snapshot_time.isoformat(),
        },
        "platform_estimate",
    )


async def _load_deployment_for_snapshot(
    db: AsyncSession,
    *,
    deployment_id: object,
) -> Deployment | None:
    return await db.scalar(
        select(Deployment)
        .options(
            selectinload(Deployment.strategy),
            selectinload(Deployment.deployment_runs),
            selectinload(Deployment.positions),
        )
        .where(Deployment.id == deployment_id)
    )


async def build_trading_event_snapshot(
    db: AsyncSession,
    *,
    deployment: Deployment,
) -> TradingEventSnapshot:
    latest_run = _latest_run(deployment)
    runtime_state = (
        latest_run.runtime_state
        if latest_run is not None and isinstance(latest_run.runtime_state, dict)
        else {}
    )
    scheduler = (
        runtime_state.get("scheduler")
        if isinstance(runtime_state.get("scheduler"), dict)
        else {}
    )
    broker_account = _extract_broker_account_payload(runtime_state)

    latest_pnl_snapshot = await db.scalar(
        select(PnlSnapshot)
        .where(PnlSnapshot.deployment_id == deployment.id)
        .order_by(PnlSnapshot.snapshot_time.desc())
        .limit(1)
    )
    if latest_pnl_snapshot is None:
        snapshot = await PnlService().build_snapshot(db, deployment_id=deployment.id)
    else:
        snapshot = PortfolioSnapshot(
            deployment_id=latest_pnl_snapshot.deployment_id,
            equity=Decimal(str(latest_pnl_snapshot.equity)),
            cash=Decimal(str(latest_pnl_snapshot.cash)),
            margin_used=Decimal(str(latest_pnl_snapshot.margin_used)),
            unrealized_pnl=Decimal(str(latest_pnl_snapshot.unrealized_pnl)),
            realized_pnl=Decimal(str(latest_pnl_snapshot.realized_pnl)),
            snapshot_time=latest_pnl_snapshot.snapshot_time,
        )

    pnl_payload, pnl_source = _broker_metrics_or_snapshot(
        snapshot=snapshot,
        broker_account=broker_account,
    )

    positions = sorted(
        list(deployment.positions),
        key=lambda item: item.updated_at,
        reverse=True,
    )
    orders = (
        await db.scalars(
            select(Order)
            .where(Order.deployment_id == deployment.id)
            .order_by(Order.submitted_at.desc())
            .limit(50)
        )
    ).all()
    fills = (
        await db.scalars(
            select(Fill)
            .join(Order, Order.id == Fill.order_id)
            .where(Order.deployment_id == deployment.id)
            .order_by(Fill.filled_at.desc())
            .limit(100)
        )
    ).all()
    approvals = (
        await db.scalars(
            select(TradeApprovalRequest)
            .where(TradeApprovalRequest.deployment_id == deployment.id)
            .order_by(TradeApprovalRequest.requested_at.desc())
            .limit(20)
        )
    ).all()

    payload = {
        "deployment_id": str(deployment.id),
        "status": deployment.status,
        "run": (
            {
                "deployment_run_id": str(latest_run.id),
                "status": latest_run.status,
                "last_bar_time": latest_run.last_bar_time.isoformat()
                if latest_run.last_bar_time is not None
                else None,
                "runtime_state": runtime_state,
                "timeframe_seconds": _as_optional_int(
                    scheduler.get("timeframe_seconds")
                ),
                "last_trigger_bucket": _as_optional_int(
                    scheduler.get("last_trigger_bucket")
                ),
                "last_enqueued_at": _as_iso_or_none(scheduler.get("last_enqueued_at")),
            }
            if latest_run is not None
            else None
        ),
        "pnl": pnl_payload,
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
            for position in positions
        ],
        "orders": [
            {
                "order_id": str(order.id),
                "symbol": order.symbol,
                "side": order.side,
                "type": order.type,
                "qty": float(order.qty),
                "price": float(order.price) if order.price is not None else None,
                "status": order.status,
                "provider_status": (
                    str(order.metadata_.get("provider_status"))
                    if isinstance(order.metadata_, dict)
                    and order.metadata_.get("provider_status") is not None
                    else order.status
                ),
                "reject_reason": order.reject_reason,
                "last_sync_at": order.last_sync_at.isoformat()
                if order.last_sync_at
                else None,
                "submitted_at": order.submitted_at.isoformat(),
            }
            for order in orders
        ],
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
            for fill in fills
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
            for approval in approvals
        ],
    }

    return TradingEventSnapshot(
        payload=payload,
        pnl_snapshot=snapshot,
        pnl_source=pnl_source,
        broker_account=broker_account,
    )


async def append_trading_event_snapshot(
    db: AsyncSession,
    *,
    deployment: Deployment | None = None,
    deployment_id: object | None = None,
) -> dict[str, object]:
    current = deployment
    if current is None:
        if deployment_id is None:
            raise ValueError("deployment or deployment_id is required")
        current = await _load_deployment_for_snapshot(db, deployment_id=deployment_id)
        if current is None:
            return {"status": "missing"}

    snapshot = await build_trading_event_snapshot(db, deployment=current)
    payload = snapshot.payload
    now = datetime.now(UTC).isoformat()
    raw_pnl = payload.get("pnl")
    pnl_payload = dict(raw_pnl) if isinstance(raw_pnl, dict) else {}
    pnl_payload.pop("snapshot_time", None)
    event_payloads: dict[str, dict[str, object]] = {
        "deployment_status": {
            "deployment_id": payload["deployment_id"],
            "status": payload["status"],
            "run": payload.get("run"),
            "updated_at": (
                current.updated_at.astimezone(UTC).isoformat()
                if current.updated_at is not None
                else now
            ),
        },
        "order_update": {
            "deployment_id": payload["deployment_id"],
            "orders": payload["orders"],
        },
        "fill_update": {
            "deployment_id": payload["deployment_id"],
            "fills": payload["fills"],
        },
        "position_update": {
            "deployment_id": payload["deployment_id"],
            "positions": payload["positions"],
        },
        "pnl_update": {
            "deployment_id": payload["deployment_id"],
            "pnl": pnl_payload,
            "pnl_source": payload.get("pnl_source"),
            "broker_account": payload.get("broker_account"),
        },
        "trade_approval_update": {
            "deployment_id": payload["deployment_id"],
            "approvals": payload["approvals"],
            "open_approval_count": sum(
                1
                for item in (
                    payload["approvals"]
                    if isinstance(payload["approvals"], list)
                    else []
                )
                if isinstance(item, dict) and str(item.get("status")) == "pending"
            ),
        },
    }

    latest_rows = (
        await db.scalars(
            select(TradingEventOutbox)
            .where(TradingEventOutbox.deployment_id == current.id)
            .order_by(TradingEventOutbox.event_seq.desc())
            .limit(64)
        )
    ).all()
    latest_payload_by_type: dict[str, dict[str, object]] = {}
    for row in latest_rows:
        if row.event_type in latest_payload_by_type:
            continue
        if isinstance(row.payload, dict):
            latest_payload_by_type[row.event_type] = dict(row.payload)

    inserted = 0
    for event_type in _EVENT_TYPES:
        next_payload = event_payloads[event_type]
        if latest_payload_by_type.get(event_type) == next_payload:
            continue
        db.add(
            TradingEventOutbox(
                deployment_id=current.id,
                event_type=event_type,
                payload=next_payload,
                occurred_at=datetime.now(UTC),
            )
        )
        inserted += 1

    if inserted > 0:
        cutoff_event_seq = await db.scalar(
            select(TradingEventOutbox.event_seq)
            .where(TradingEventOutbox.deployment_id == current.id)
            .order_by(TradingEventOutbox.event_seq.desc())
            .offset(_OUTBOX_RETENTION_PER_DEPLOYMENT)
            .limit(1)
        )
        if cutoff_event_seq is not None:
            await db.execute(
                delete(TradingEventOutbox).where(
                    TradingEventOutbox.deployment_id == current.id,
                    TradingEventOutbox.event_seq < int(cutoff_event_seq),
                )
            )
        await db.commit()

    return {
        "status": "ok",
        "deployment_id": str(current.id),
        "inserted": inserted,
    }
