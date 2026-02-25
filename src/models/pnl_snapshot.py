"""PnL snapshot model."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Index, Numeric
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base

if TYPE_CHECKING:
    from src.models.deployment import Deployment


class PnlSnapshot(Base):
    """Time-series equity/cash snapshot for one deployment."""

    __tablename__ = "pnl_snapshots"
    __table_args__ = (
        CheckConstraint("equity >= 0", name="ck_pnl_snapshots_equity_non_negative"),
        CheckConstraint("cash >= 0", name="ck_pnl_snapshots_cash_non_negative"),
        CheckConstraint("margin_used >= 0", name="ck_pnl_snapshots_margin_used_non_negative"),
        Index("ix_pnl_snapshots_deployment_time", "deployment_id", "snapshot_time"),
    )

    deployment_id: Mapped[UUID] = mapped_column(
        ForeignKey("deployments.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    equity: Mapped[Decimal] = mapped_column(
        Numeric(20, 8),
        nullable=False,
        default=Decimal("0"),
        server_default="0",
    )
    cash: Mapped[Decimal] = mapped_column(
        Numeric(20, 8),
        nullable=False,
        default=Decimal("0"),
        server_default="0",
    )
    margin_used: Mapped[Decimal] = mapped_column(
        Numeric(20, 8),
        nullable=False,
        default=Decimal("0"),
        server_default="0",
    )
    unrealized_pnl: Mapped[Decimal] = mapped_column(
        Numeric(20, 8),
        nullable=False,
        default=Decimal("0"),
        server_default="0",
    )
    realized_pnl: Mapped[Decimal] = mapped_column(
        Numeric(20, 8),
        nullable=False,
        default=Decimal("0"),
        server_default="0",
    )
    snapshot_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)

    deployment: Mapped[Deployment] = relationship(back_populates="pnl_snapshots")
