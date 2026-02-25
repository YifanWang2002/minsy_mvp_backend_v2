"""Streaming endpoints for deployment runtime updates."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.middleware.auth import get_current_user
from src.dependencies import get_db
from src.engine.pnl.service import PnlService
from src.models.deployment import Deployment
from src.models.deployment_run import DeploymentRun
from src.models.fill import Fill
from src.models.order import Order
from src.models.position import Position
from src.models.trade_approval_request import TradeApprovalRequest
from src.models.trading_event_outbox import TradingEventOutbox
from src.models.user import User

router = APIRouter(prefix="/stream", tags=["trading-stream"])

_EVENT_TYPES = (
    "deployment_status",
    "order_update",
    "fill_update",
    "position_update",
    "pnl_update",
    "trade_approval_update",
)
_OUTBOX_RETENTION_PER_DEPLOYMENT = 2000


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


def _extract_broker_account_payload(runtime_state: dict[str, object]) -> dict[str, object] | None:
    raw = runtime_state.get("broker_account")
    if not isinstance(raw, dict):
        return None
    provider = str(raw.get("provider") or "").strip().lower()
    source = str(raw.get("source") or "").strip()
    sync_status = str(raw.get("sync_status") or "").strip()
    if not provider or not source or not sync_status:
        return None
    symbols_raw = raw.get("symbols")
    symbols = [str(item).upper() for item in symbols_raw if isinstance(item, str)] if isinstance(symbols_raw, list) else []
    positions_count = raw.get("positions_count")
    try:
        parsed_positions_count = int(positions_count) if positions_count is not None else None
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


async def _load_owned_deployment(
    db: AsyncSession,
    *,
    deployment_id: str,
    user_id: str,
) -> Deployment:
    deployment = await db.scalar(
        select(Deployment).where(Deployment.id == deployment_id, Deployment.user_id == user_id)
    )
    if deployment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "DEPLOYMENT_NOT_FOUND", "message": "Deployment not found."},
        )
    return deployment


async def _build_snapshot_payload(
    db: AsyncSession,
    *,
    deployment: Deployment,
) -> dict[str, object]:
    pnl_service = PnlService()
    snapshot = await pnl_service.build_snapshot(db, deployment_id=deployment.id)
    positions = (
        await db.scalars(select(Position).where(Position.deployment_id == deployment.id))
    ).all()
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
    run = await db.scalar(
        select(DeploymentRun)
        .where(DeploymentRun.deployment_id == deployment.id)
        .order_by(DeploymentRun.created_at.desc())
        .limit(1)
    )
    runtime_state = run.runtime_state if run is not None and isinstance(run.runtime_state, dict) else {}
    scheduler = runtime_state.get("scheduler") if isinstance(runtime_state.get("scheduler"), dict) else {}
    broker_account = _extract_broker_account_payload(runtime_state)
    return {
        "deployment_id": str(deployment.id),
        "status": deployment.status,
        "run": (
            {
                "deployment_run_id": str(run.id),
                "status": run.status,
                "last_bar_time": run.last_bar_time.isoformat() if run.last_bar_time is not None else None,
                "runtime_state": runtime_state,
                "timeframe_seconds": _as_optional_int(scheduler.get("timeframe_seconds")),
                "last_trigger_bucket": _as_optional_int(scheduler.get("last_trigger_bucket")),
                "last_enqueued_at": _as_iso_or_none(scheduler.get("last_enqueued_at")),
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
        "pnl_source": "platform_estimate",
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
                    if isinstance(order.metadata_, dict) and order.metadata_.get("provider_status") is not None
                    else order.status
                ),
                "reject_reason": order.reject_reason,
                "last_sync_at": order.last_sync_at.isoformat() if order.last_sync_at else None,
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
                "approved_at": approval.approved_at.isoformat() if approval.approved_at else None,
                "rejected_at": approval.rejected_at.isoformat() if approval.rejected_at else None,
                "expired_at": approval.expired_at.isoformat() if approval.expired_at else None,
                "executed_at": approval.executed_at.isoformat() if approval.executed_at else None,
                "approved_via": approval.approved_via,
                "decision_actor": approval.decision_actor,
                "execution_error": approval.execution_error,
            }
            for approval in approvals
        ],
    }


async def _append_outbox_snapshot(
    db: AsyncSession,
    *,
    deployment: Deployment,
) -> dict[str, object]:
    snapshot = await _build_snapshot_payload(db, deployment=deployment)
    now = datetime.now(UTC).isoformat()
    raw_pnl = snapshot.get("pnl")
    pnl_payload = dict(raw_pnl) if isinstance(raw_pnl, dict) else {}
    # Prevent synthetic churn from time-only changes in computed snapshots.
    pnl_payload.pop("snapshot_time", None)
    event_payloads: dict[str, dict[str, object]] = {
        "deployment_status": {
            "deployment_id": snapshot["deployment_id"],
            "status": snapshot["status"],
            "run": snapshot.get("run"),
            "updated_at": (
                deployment.updated_at.astimezone(UTC).isoformat()
                if deployment.updated_at is not None
                else now
            ),
        },
        "order_update": {
            "deployment_id": snapshot["deployment_id"],
            "orders": snapshot["orders"],
        },
        "fill_update": {
            "deployment_id": snapshot["deployment_id"],
            "fills": snapshot["fills"],
        },
        "position_update": {
            "deployment_id": snapshot["deployment_id"],
            "positions": snapshot["positions"],
        },
        "pnl_update": {
            "deployment_id": snapshot["deployment_id"],
            "pnl": pnl_payload,
            "pnl_source": snapshot.get("pnl_source"),
            "broker_account": snapshot.get("broker_account"),
        },
        "trade_approval_update": {
            "deployment_id": snapshot["deployment_id"],
            "approvals": snapshot["approvals"],
            "open_approval_count": sum(
                1
                for item in (snapshot["approvals"] if isinstance(snapshot["approvals"], list) else [])
                if isinstance(item, dict) and str(item.get("status")) == "pending"
            ),
        },
    }

    latest_rows = (
        await db.scalars(
            select(TradingEventOutbox)
            .where(TradingEventOutbox.deployment_id == deployment.id)
            .order_by(TradingEventOutbox.event_seq.desc())
            .limit(64),
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
        payload = event_payloads[event_type]
        last_payload = latest_payload_by_type.get(event_type)
        if last_payload == payload:
            continue
        db.add(
            TradingEventOutbox(
                deployment_id=deployment.id,
                event_type=event_type,
                payload=payload,
                occurred_at=datetime.now(UTC),
            )
        )
        inserted += 1

    if inserted > 0:
        cutoff_event_seq = await db.scalar(
            select(TradingEventOutbox.event_seq)
            .where(TradingEventOutbox.deployment_id == deployment.id)
            .order_by(TradingEventOutbox.event_seq.desc())
            .offset(_OUTBOX_RETENTION_PER_DEPLOYMENT)
            .limit(1)
        )
        if cutoff_event_seq is not None:
            await db.execute(
                delete(TradingEventOutbox).where(
                    TradingEventOutbox.deployment_id == deployment.id,
                    TradingEventOutbox.event_seq < int(cutoff_event_seq),
                )
            )
        await db.commit()
    return {
        "deployment_id": str(deployment.id),
        "status": deployment.status,
        "updated_at": now,
    }


async def _poll_outbox_events(
    db: AsyncSession,
    *,
    deployment_id: str,
    cursor: int,
    limit: int = 50,
) -> list[TradingEventOutbox]:
    rows = (
        await db.scalars(
            select(TradingEventOutbox)
            .where(
                TradingEventOutbox.deployment_id == deployment_id,
                TradingEventOutbox.event_seq > cursor,
            )
            .order_by(TradingEventOutbox.event_seq.asc())
            .limit(limit),
        )
    ).all()
    return list(rows)


async def _append_heartbeat_outbox_event(
    db: AsyncSession,
    *,
    deployment: Deployment,
    cursor: int,
    updated_at: str,
) -> TradingEventOutbox:
    row = TradingEventOutbox(
        deployment_id=deployment.id,
        event_type="heartbeat",
        payload={
            "deployment_id": str(deployment.id),
            "status": deployment.status,
            "cursor": cursor,
            "updated_at": updated_at,
        },
        occurred_at=datetime.now(UTC),
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return row


@router.get("/deployments/{deployment_id}")
async def stream_deployment(
    request: Request,
    deployment_id: str,
    cursor: int | None = Query(default=None, ge=0),
    poll_seconds: float = Query(default=1.0, ge=0.2, le=10.0),
    heartbeat_seconds: float = Query(default=1.0, ge=0.2, le=60.0),
    max_events: int | None = Query(default=None, ge=1, le=5000),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    user_id = str(user.id)
    await _load_owned_deployment(db, deployment_id=deployment_id, user_id=user_id)

    async def _event_stream() -> AsyncIterator[str]:
        if cursor is None:
            latest = await db.scalar(
                select(TradingEventOutbox.event_seq)
                .where(TradingEventOutbox.deployment_id == deployment_id)
                .order_by(TradingEventOutbox.event_seq.desc())
                .limit(1)
            )
            next_cursor = int(latest or 0)
        else:
            next_cursor = int(cursor)
        emitted = 0
        last_heartbeat_at = time.monotonic()

        while True:
            if await request.is_disconnected():
                break

            db.expire_all()
            deployment = await _load_owned_deployment(
                db,
                deployment_id=deployment_id,
                user_id=user_id,
            )
            heartbeat_payload = await _append_outbox_snapshot(db, deployment=deployment)
            rows = await _poll_outbox_events(
                db,
                deployment_id=deployment_id,
                cursor=next_cursor,
            )
            serialized_rows: list[tuple[int, str, dict[str, object]]] = []
            for row in rows:
                payload = row.payload if isinstance(row.payload, dict) else {}
                payload_dict = dict(payload)
                payload_dict.setdefault("deployment_id", deployment_id)
                payload_dict.setdefault("event_seq", row.event_seq)
                serialized_rows.append((int(row.event_seq), row.event_type, payload_dict))

            # Do not keep a long-lived transaction open while waiting on SSE I/O.
            # Leaving sessions "idle in transaction" can amplify lock contention
            # and stall unrelated orders/fills reads.
            await db.rollback()

            if serialized_rows:
                for event_seq, event_type, payload in serialized_rows:
                    yield (
                        f"id: {event_seq}\n"
                        f"event: {event_type}\n"
                        f"data: {json.dumps(payload, ensure_ascii=True)}\n\n"
                    )
                    next_cursor = max(next_cursor, event_seq)
                    emitted += 1
                    if max_events is not None and emitted >= max_events:
                        break

            if max_events is not None and emitted >= max_events:
                break

            if not serialized_rows:
                now_monotonic = time.monotonic()
                if now_monotonic - last_heartbeat_at >= heartbeat_seconds:
                    heartbeat = dict(heartbeat_payload)
                    heartbeat["cursor"] = next_cursor
                    if max_events is None:
                        yield f"event: heartbeat\ndata: {json.dumps(heartbeat, ensure_ascii=True)}\n\n"
                    else:
                        heartbeat_row = await _append_heartbeat_outbox_event(
                            db,
                            deployment=deployment,
                            cursor=next_cursor,
                            updated_at=str(heartbeat.get("updated_at", datetime.now(UTC).isoformat())),
                        )
                        await db.rollback()
                        heartbeat["event_seq"] = heartbeat_row.event_seq
                        yield (
                            f"id: {heartbeat_row.event_seq}\n"
                            "event: heartbeat\n"
                            f"data: {json.dumps(heartbeat, ensure_ascii=True)}\n\n"
                        )
                        next_cursor = max(next_cursor, int(heartbeat_row.event_seq))
                    last_heartbeat_at = now_monotonic
                    emitted += 1
                    if max_events is not None and emitted >= max_events:
                        break
                await asyncio.sleep(poll_seconds)
                continue

            await asyncio.sleep(0)

        if max_events is not None:
            yield f"event: stream_end\ndata: {json.dumps({'cursor': next_cursor}, ensure_ascii=True)}\n\n"

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
