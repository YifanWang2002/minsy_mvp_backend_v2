"""Position model for deployment runtime holdings."""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import CheckConstraint, ForeignKey, Index, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from packages.infra.db.models.base import Base

if TYPE_CHECKING:
    from packages.infra.db.models.deployment import Deployment


class Position(Base):
    """Current position per deployment and symbol."""

    __tablename__ = "positions"
    __table_args__ = (
        CheckConstraint("side IN ('long', 'short', 'flat')", name="ck_positions_side"),
        CheckConstraint("qty >= 0", name="ck_positions_qty_non_negative"),
        CheckConstraint("avg_entry_price >= 0", name="ck_positions_avg_entry_price_non_negative"),
        CheckConstraint("mark_price >= 0", name="ck_positions_mark_price_non_negative"),
        Index("ix_positions_deployment_symbol", "deployment_id", "symbol", unique=True),
    )

    deployment_id: Mapped[UUID] = mapped_column(
        ForeignKey("deployments.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    symbol: Mapped[str] = mapped_column(String(80), nullable=False)
    side: Mapped[str] = mapped_column(String(10), nullable=False, default="flat", server_default="flat")
    qty: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False, default=Decimal("0"), server_default="0")
    avg_entry_price: Mapped[Decimal] = mapped_column(
        Numeric(20, 8),
        nullable=False,
        default=Decimal("0"),
        server_default="0",
    )
    mark_price: Mapped[Decimal] = mapped_column(
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

    deployment: Mapped[Deployment] = relationship(back_populates="positions")
