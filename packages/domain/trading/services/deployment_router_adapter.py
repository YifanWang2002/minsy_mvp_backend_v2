"""Compatibility adapter exposing deployment lifecycle operations."""

from __future__ import annotations

from decimal import Decimal
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from packages.domain.trading import deployment_ops
from packages.infra.db.models.deployment import Deployment
from packages.infra.db.models.order import Order
from packages.infra.db.models.position import Position


def _as_uuid(value: Any) -> UUID:
    return value if isinstance(value, UUID) else UUID(str(value))


def _as_decimal(value: Any) -> Decimal:
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


async def create_deployment(
    *,
    payload: Any,
    user: Any,
    db: AsyncSession,
) -> dict[str, Any]:
    """Compatibility shape for legacy callers expecting payload/user objects."""

    deployment = await deployment_ops.create_deployment(
        db,
        strategy_id=_as_uuid(payload.strategy_id),
        broker_account_id=_as_uuid(payload.broker_account_id),
        user_id=_as_uuid(user.id),
        mode=str(payload.mode),
        capital_allocated=_as_decimal(payload.capital_allocated),
        risk_limits=dict(payload.risk_limits or {}),
        runtime_state=dict(payload.runtime_state or {}),
    )
    return deployment_ops.serialize_deployment(deployment)


async def load_owned_deployment(
    db: AsyncSession,
    *,
    deployment_id: UUID,
    user_id: UUID,
) -> Deployment:
    return await deployment_ops.load_owned_deployment(
        db,
        deployment_id=deployment_id,
        user_id=user_id,
    )


async def apply_status_transition(
    db: AsyncSession,
    *,
    deployment: Deployment,
    target_status: str,
) -> Deployment:
    return await deployment_ops.apply_status_transition(
        db,
        deployment=deployment,
        target_status=target_status,
    )


def serialize_deployment(deployment: Deployment) -> dict[str, Any]:
    return deployment_ops.serialize_deployment(deployment)


def serialize_position(position: Position) -> dict[str, Any]:
    return deployment_ops.serialize_position(position)


def serialize_order(order: Order) -> dict[str, Any]:
    return deployment_ops.serialize_order(order)
