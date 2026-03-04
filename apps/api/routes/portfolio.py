"""Portfolio and fill endpoints for deployment runtime."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from apps.api.middleware.auth import get_current_user
from apps.api.schemas.events import (
    BrokerAccountSnapshotResponse,
    FillResponse,
    PortfolioResponse,
    PositionResponse,
)
from apps.api.dependencies import get_db
from packages.domain.trading.pnl.service import PnlService, PortfolioSnapshot
from packages.domain.trading.runtime.runtime_service import (
    refresh_portfolio_snapshot_for_poll,
    sync_pending_orders_for_poll,
)
from packages.infra.db.models.deployment import Deployment
from packages.infra.db.models.fill import Fill
from packages.infra.db.models.order import Order
from packages.infra.db.models.pnl_snapshot import PnlSnapshot
from packages.infra.db.models.position import Position
from packages.infra.db.models.user import User

router = APIRouter(prefix="/deployments", tags=["portfolio"])

_PORTFOLIO_POLL_REFRESH_MIN_INTERVAL_SECONDS = 12.0


async def _load_owned_deployment(
    db: AsyncSession,
    *,
    deployment_id: str,
    user_id: str,
) -> Deployment:
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


def _serialize_position(position: Position) -> PositionResponse:
    return PositionResponse(
        position_id=position.id,
        deployment_id=position.deployment_id,
        symbol=position.symbol,
        side=position.side,
        qty=float(position.qty),
        avg_entry_price=float(position.avg_entry_price),
        mark_price=float(position.mark_price),
        unrealized_pnl=float(position.unrealized_pnl),
        realized_pnl=float(position.realized_pnl),
        created_at=position.created_at,
        updated_at=position.updated_at,
    )


def _as_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_optional_datetime(value: Any) -> datetime | None:
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


def _serialize_broker_account_snapshot(
    raw: Any,
) -> BrokerAccountSnapshotResponse | None:
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
    return BrokerAccountSnapshotResponse(
        provider=provider,
        source=source,
        sync_status=sync_status,
        fetched_at=_as_optional_datetime(raw.get("fetched_at")),
        equity=_as_optional_float(raw.get("equity")),
        cash=_as_optional_float(raw.get("cash")),
        buying_power=_as_optional_float(raw.get("buying_power")),
        margin_used=_as_optional_float(raw.get("margin_used")),
        unrealized_pnl=_as_optional_float(raw.get("unrealized_pnl")),
        realized_pnl=_as_optional_float(raw.get("realized_pnl")),
        positions_count=parsed_positions_count,
        symbols=symbols,
        error=str(raw.get("error"))[:500] if raw.get("error") is not None else None,
        updated_at=_as_optional_datetime(raw.get("updated_at")),
    )


def _runtime_broker_payload(deployment: Deployment) -> dict[str, Any] | None:
    run = deployment.deployment_runs[0] if deployment.deployment_runs else None
    if run is None or not isinstance(run.runtime_state, dict):
        return None
    payload = run.runtime_state.get("broker_account")
    if not isinstance(payload, dict):
        return None
    return payload


async def _load_cached_portfolio_snapshot(
    db: AsyncSession,
    *,
    deployment_id: str,
    max_age_seconds: float | None,
) -> PortfolioSnapshot | None:
    row = await db.scalar(
        select(PnlSnapshot)
        .where(PnlSnapshot.deployment_id == deployment_id)
        .order_by(PnlSnapshot.snapshot_time.desc())
        .limit(1)
    )
    if row is None:
        return None
    if max_age_seconds is not None:
        age_seconds = (datetime.now(UTC) - row.snapshot_time).total_seconds()
        if age_seconds > max_age_seconds:
            return None
    return PortfolioSnapshot(
        deployment_id=row.deployment_id,
        equity=row.equity,
        cash=row.cash,
        margin_used=row.margin_used,
        unrealized_pnl=row.unrealized_pnl,
        realized_pnl=row.realized_pnl,
        snapshot_time=row.snapshot_time,
    )


def _resolve_portfolio_metrics(
    *,
    latest_snapshot: Any,
    broker_account: BrokerAccountSnapshotResponse | None,
) -> tuple[str, float, float, float, float, float]:
    return (
        "platform_estimate",
        float(latest_snapshot.equity),
        float(latest_snapshot.cash),
        float(latest_snapshot.margin_used),
        float(latest_snapshot.unrealized_pnl),
        float(latest_snapshot.realized_pnl),
    )


@router.get("/{deployment_id}/portfolio", response_model=PortfolioResponse)
async def get_deployment_portfolio(
    deployment_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> PortfolioResponse:
    deployment = await _load_owned_deployment(
        db, deployment_id=deployment_id, user_id=str(user.id)
    )
    broker_payload = _runtime_broker_payload(deployment)
    if deployment.status != "active":
        latest_snapshot = await _load_cached_portfolio_snapshot(
            db,
            deployment_id=deployment.id,
            max_age_seconds=None,
        )
        if latest_snapshot is None:
            latest_snapshot = await PnlService().build_snapshot(
                db,
                deployment_id=deployment.id,
            )
        pending_sync = await sync_pending_orders_for_poll(
            db,
            deployment=deployment,
        )
        if int(pending_sync.get("pending_order_fill_updates", 0) or 0) > 0:
            refreshed_snapshot = await _load_cached_portfolio_snapshot(
                db,
                deployment_id=deployment.id,
                max_age_seconds=None,
            )
            if refreshed_snapshot is not None:
                latest_snapshot = refreshed_snapshot
    else:
        latest_snapshot = await _load_cached_portfolio_snapshot(
            db,
            deployment_id=deployment.id,
            max_age_seconds=_PORTFOLIO_POLL_REFRESH_MIN_INTERVAL_SECONDS,
        )
        if latest_snapshot is None:
            latest_snapshot, broker_payload = await refresh_portfolio_snapshot_for_poll(
                db,
                deployment=deployment,
            )
            await PnlService().persist_snapshot(
                db,
                snapshot=latest_snapshot,
            )
        else:
            pending_sync = await sync_pending_orders_for_poll(
                db,
                deployment=deployment,
            )
            if int(pending_sync.get("pending_order_fill_updates", 0) or 0) > 0:
                refreshed_snapshot = await _load_cached_portfolio_snapshot(
                    db,
                    deployment_id=deployment.id,
                    max_age_seconds=None,
                )
                if refreshed_snapshot is not None:
                    latest_snapshot = refreshed_snapshot
    positions = (
        await db.scalars(
            select(Position)
            .where(Position.deployment_id == deployment.id)
            .order_by(Position.updated_at.desc())
        )
    ).all()
    broker_account = _serialize_broker_account_snapshot(broker_payload)
    (
        metrics_source,
        equity,
        cash,
        margin_used,
        unrealized_pnl,
        realized_pnl,
    ) = _resolve_portfolio_metrics(
        latest_snapshot=latest_snapshot,
        broker_account=broker_account,
    )

    return PortfolioResponse(
        deployment_id=deployment.id,
        metrics_source=metrics_source,
        equity=equity,
        cash=cash,
        margin_used=margin_used,
        unrealized_pnl=unrealized_pnl,
        realized_pnl=realized_pnl,
        snapshot_time=latest_snapshot.snapshot_time,
        broker_account=broker_account,
        positions=[_serialize_position(position) for position in positions],
    )


@router.get("/{deployment_id}/fills", response_model=list[FillResponse])
async def list_deployment_fills(
    deployment_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[FillResponse]:
    deployment = await _load_owned_deployment(
        db, deployment_id=deployment_id, user_id=str(user.id)
    )
    rows = (
        await db.scalars(
            select(Fill)
            .join(Order, Order.id == Fill.order_id)
            .where(Order.deployment_id == deployment.id)
            .order_by(Fill.filled_at.desc()),
        )
    ).all()
    return [
        FillResponse(
            fill_id=row.id,
            order_id=row.order_id,
            provider_fill_id=row.provider_fill_id,
            fill_price=float(row.fill_price),
            fill_qty=float(row.fill_qty),
            fee=float(row.fee),
            filled_at=row.filled_at,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )
        for row in rows
    ]
