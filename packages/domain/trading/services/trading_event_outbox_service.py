"""Persistent trading-event snapshot emission for deployment streams."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from packages.domain.trading.pnl.service import PnlService, PortfolioSnapshot
from packages.domain.trading.services.trading_event_pubsub import (
    TradingRealtimeEvent,
    publish_realtime_event,
)
from packages.domain.trading.services.trading_projection_builder import (
    build_projection_payload,
    extract_broker_account_payload,
    load_projection_entities,
)
from packages.infra.db.models.deployment import Deployment
from packages.infra.db.models.pnl_snapshot import PnlSnapshot
from packages.infra.db.models.trading_event_outbox import TradingEventOutbox

_EVENT_TYPES: tuple[str, ...] = (
    "deployment_status",
    "order_update",
    "fill_update",
    "position_update",
    "pnl_update",
    "manual_action_update",
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
    entities = await load_projection_entities(db, deployment=deployment)
    broker_account = extract_broker_account_payload(runtime_state=entities.runtime_state)

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

    payload = build_projection_payload(
        deployment=deployment,
        entities=entities,
        snapshot=snapshot,
        pnl_source=pnl_source,
        broker_account=broker_account,
    )
    payload["pnl"] = pnl_payload

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
        "manual_action_update": {
            "deployment_id": payload["deployment_id"],
            "manual_actions": payload.get("manual_actions", []),
            "latest_manual_action_id": (
                payload["manual_actions"][0]["manual_trade_action_id"]
                if isinstance(payload.get("manual_actions"), list)
                and payload["manual_actions"]
                and isinstance(payload["manual_actions"][0], dict)
                and payload["manual_actions"][0].get("manual_trade_action_id")
                is not None
                else None
            ),
            "updated_at": (
                payload["manual_actions"][0]["updated_at"]
                if isinstance(payload.get("manual_actions"), list)
                and payload["manual_actions"]
                and isinstance(payload["manual_actions"][0], dict)
                and payload["manual_actions"][0].get("updated_at") is not None
                else now
            ),
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
    inserted_rows: list[TradingEventOutbox] = []
    for event_type in _EVENT_TYPES:
        next_payload = event_payloads[event_type]
        if latest_payload_by_type.get(event_type) == next_payload:
            continue
        row = TradingEventOutbox(
            deployment_id=current.id,
            event_type=event_type,
            payload=next_payload,
            occurred_at=datetime.now(UTC),
        )
        db.add(row)
        inserted_rows.append(row)
        inserted += 1

    if inserted > 0:
        await db.flush()
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
        for row in inserted_rows:
            payload = row.payload if isinstance(row.payload, dict) else {}
            await publish_realtime_event(
                TradingRealtimeEvent(
                    deployment_id=str(current.id),
                    event_type=str(row.event_type),
                    event_seq=int(row.event_seq),
                    payload=dict(payload),
                )
            )

    return {
        "status": "ok",
        "deployment_id": str(current.id),
        "inserted": inserted,
    }
