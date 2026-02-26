"""PnL and portfolio snapshot service."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from packages.infra.db.models.deployment import Deployment
from packages.infra.db.models.pnl_snapshot import PnlSnapshot
from packages.infra.db.models.position import Position


def _to_decimal(value: Decimal | float | int) -> Decimal:
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


@dataclass(frozen=True, slots=True)
class PortfolioSnapshot:
    deployment_id: UUID
    equity: Decimal
    cash: Decimal
    margin_used: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    snapshot_time: datetime


class PnlService:
    """Computes and persists deployment-level PnL snapshots."""

    def compute_unrealized(
        self,
        *,
        side: str,
        qty: Decimal,
        avg_entry_price: Decimal,
        mark_price: Decimal,
    ) -> Decimal:
        if qty <= 0:
            return Decimal("0")
        if side == "short":
            return (avg_entry_price - mark_price) * qty
        return (mark_price - avg_entry_price) * qty

    async def build_snapshot(
        self,
        db: AsyncSession,
        *,
        deployment_id: UUID,
    ) -> PortfolioSnapshot:
        deployment = await db.scalar(select(Deployment).where(Deployment.id == deployment_id))
        if deployment is None:
            raise ValueError("Deployment not found.")

        positions = (
            await db.scalars(select(Position).where(Position.deployment_id == deployment_id))
        ).all()
        capital = _to_decimal(deployment.capital_allocated)

        realized = sum((_to_decimal(position.realized_pnl) for position in positions), Decimal("0"))
        unrealized = Decimal("0")
        margin_used = Decimal("0")
        for position in positions:
            qty = _to_decimal(position.qty)
            mark_price = _to_decimal(position.mark_price)
            avg_entry = _to_decimal(position.avg_entry_price)
            side = str(position.side).strip().lower()
            position_unrealized = self.compute_unrealized(
                side=side,
                qty=qty,
                avg_entry_price=avg_entry,
                mark_price=mark_price,
            )
            position.unrealized_pnl = position_unrealized
            unrealized += position_unrealized
            margin_used += qty * mark_price
        await db.flush()
        cash = capital + realized - margin_used
        if cash < 0:
            cash = Decimal("0")
        equity = cash + margin_used + unrealized
        snapshot_time = datetime.now(UTC)
        return PortfolioSnapshot(
            deployment_id=deployment_id,
            equity=equity,
            cash=cash,
            margin_used=margin_used,
            unrealized_pnl=unrealized,
            realized_pnl=realized,
            snapshot_time=snapshot_time,
        )

    async def persist_snapshot(
        self,
        db: AsyncSession,
        *,
        snapshot: PortfolioSnapshot,
    ) -> PnlSnapshot:
        row = PnlSnapshot(
            deployment_id=snapshot.deployment_id,
            equity=snapshot.equity,
            cash=snapshot.cash,
            margin_used=snapshot.margin_used,
            unrealized_pnl=snapshot.unrealized_pnl,
            realized_pnl=snapshot.realized_pnl,
            snapshot_time=snapshot.snapshot_time,
        )
        db.add(row)
        await db.commit()
        await db.refresh(row)
        return row

    @staticmethod
    def snapshot_to_payload(snapshot: PortfolioSnapshot) -> dict[str, float | str]:
        return {
            "deployment_id": str(snapshot.deployment_id),
            "equity": float(snapshot.equity),
            "cash": float(snapshot.cash),
            "margin_used": float(snapshot.margin_used),
            "unrealized_pnl": float(snapshot.unrealized_pnl),
            "realized_pnl": float(snapshot.realized_pnl),
            "snapshot_time": snapshot.snapshot_time.isoformat(),
        }
