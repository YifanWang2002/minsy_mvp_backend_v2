"""Shared replay helpers for trading stream and WebSocket routes."""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from packages.infra.db.models.deployment import Deployment
from packages.infra.db.models.trading_event_outbox import TradingEventOutbox


@dataclass(frozen=True, slots=True)
class ReplayCursorResolution:
    """Resolved replay cursor for one deployment subscription."""

    deployment_id: str
    cursor: int


async def load_owned_deployment(
    db: AsyncSession,
    *,
    deployment_id: str,
    user_id: str,
) -> Deployment:
    """Load a deployment and verify ownership for the authenticated user."""
    deployment = await db.scalar(
        select(Deployment)
        .options(
            selectinload(Deployment.strategy),
            selectinload(Deployment.deployment_runs),
            selectinload(Deployment.positions),
        )
        .where(Deployment.id == deployment_id, Deployment.user_id == user_id)
    )
    if deployment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "DEPLOYMENT_NOT_FOUND", "message": "Deployment not found."},
        )
    return deployment


async def resolve_replay_cursor(
    db: AsyncSession,
    *,
    deployment_id: str,
    requested_cursor: int | None,
) -> ReplayCursorResolution:
    """Resolve replay cursor, defaulting to the latest outbox sequence."""
    if requested_cursor is not None and requested_cursor > 0:
        return ReplayCursorResolution(deployment_id=deployment_id, cursor=requested_cursor)

    latest = await db.scalar(
        select(TradingEventOutbox.event_seq)
        .where(TradingEventOutbox.deployment_id == deployment_id)
        .order_by(TradingEventOutbox.event_seq.desc())
        .limit(1)
    )
    return ReplayCursorResolution(
        deployment_id=deployment_id,
        cursor=_coerce_int_or_zero(latest),
    )


async def poll_outbox_events(
    db: AsyncSession,
    *,
    deployment_id: str,
    cursor: int,
    limit: int = 300,
) -> list[TradingEventOutbox]:
    """Fetch ordered outbox rows newer than cursor."""
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


def _coerce_int_or_zero(value: object) -> int:
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return 0
    return 0
