"""Streaming endpoints for deployment runtime updates."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.dependencies import get_db
from apps.api.middleware.auth import get_current_user
from apps.api.services.trading_stream_replay_service import (
    load_owned_deployment as load_owned_deployment_service,
)
from apps.api.services.trading_stream_replay_service import (
    poll_outbox_events as poll_outbox_events_service,
)
from apps.api.services.trading_stream_replay_service import (
    resolve_replay_cursor,
)
from packages.domain.trading.runtime.runtime_service import (
    refresh_portfolio_snapshot_for_poll,
)
from packages.domain.trading.services.trading_projection_builder import (
    build_projection_payload,
    extract_broker_account_payload,
    load_projection_entities,
)
from packages.infra.db.models.deployment import Deployment
from packages.infra.db.models.trading_event_outbox import TradingEventOutbox
from packages.infra.db.models.user import User

router = APIRouter(prefix="/stream", tags=["trading-stream"])

_EVENT_TYPES = (
    "deployment_status",
    "order_update",
    "fill_update",
    "position_update",
    "pnl_update",
    "manual_action_update",
    "trade_approval_update",
)
_OUTBOX_RETENTION_PER_DEPLOYMENT = 2000


async def _load_owned_deployment(
    db: AsyncSession,
    *,
    deployment_id: str,
    user_id: str,
) -> Deployment:
    return await load_owned_deployment_service(
        db,
        deployment_id=deployment_id,
        user_id=user_id,
    )


async def _build_snapshot_payload(
    db: AsyncSession,
    *,
    deployment: Deployment,
) -> dict[str, object]:
    snapshot, broker_payload = await refresh_portfolio_snapshot_for_poll(
        db,
        deployment=deployment,
    )
    entities = await load_projection_entities(db, deployment=deployment)
    broker_account = extract_broker_account_payload(
        runtime_state=entities.runtime_state,
        broker_payload=broker_payload if isinstance(broker_payload, dict) else None,
    )
    return build_projection_payload(
        deployment=deployment,
        entities=entities,
        snapshot=snapshot,
        pnl_source="platform_estimate",
        broker_account=broker_account,
    )


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
        "manual_action_update": {
            "deployment_id": snapshot["deployment_id"],
            "manual_actions": snapshot.get("manual_actions", []),
            "latest_manual_action_id": (
                snapshot["manual_actions"][0]["manual_trade_action_id"]
                if isinstance(snapshot.get("manual_actions"), list)
                and snapshot["manual_actions"]
                and isinstance(snapshot["manual_actions"][0], dict)
                and snapshot["manual_actions"][0].get("manual_trade_action_id") is not None
                else None
            ),
            "updated_at": (
                snapshot["manual_actions"][0]["updated_at"]
                if isinstance(snapshot.get("manual_actions"), list)
                and snapshot["manual_actions"]
                and isinstance(snapshot["manual_actions"][0], dict)
                and snapshot["manual_actions"][0].get("updated_at") is not None
                else now
            ),
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
    limit: int = 300,
) -> list[TradingEventOutbox]:
    return await poll_outbox_events_service(
        db,
        deployment_id=deployment_id,
        cursor=cursor,
        limit=limit,
    )


async def _append_heartbeat_outbox_event(
    db: AsyncSession,
    *,
    deployment_id: object,
    deployment_status: str,
    cursor: int,
    updated_at: str,
) -> TradingEventOutbox:
    row = TradingEventOutbox(
        deployment_id=deployment_id,
        event_type="heartbeat",
        payload={
            "deployment_id": str(deployment_id),
            "status": deployment_status,
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
        resolved = await resolve_replay_cursor(
            db,
            deployment_id=deployment_id,
            requested_cursor=cursor,
        )
        next_cursor = int(resolved.cursor)
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
            deployment_status = deployment.status
            deployment_updated_at = (
                deployment.updated_at.astimezone(UTC).isoformat()
                if deployment.updated_at is not None
                else datetime.now(UTC).isoformat()
            )
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
                    heartbeat = {
                        "deployment_id": deployment_id,
                        "status": deployment_status,
                        "cursor": next_cursor,
                        "updated_at": deployment_updated_at,
                    }
                    yield f"event: heartbeat\ndata: {json.dumps(heartbeat, ensure_ascii=True)}\n\n"
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
