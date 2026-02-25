"""Fill model for executed order fragments."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base

if TYPE_CHECKING:
    from src.models.order import Order


class Fill(Base):
    """Broker fill records for one order."""

    __tablename__ = "fills"
    __table_args__ = (
        CheckConstraint("fill_price >= 0", name="ck_fills_fill_price_non_negative"),
        CheckConstraint("fill_qty > 0", name="ck_fills_fill_qty_positive"),
        CheckConstraint("fee >= 0", name="ck_fills_fee_non_negative"),
    )

    order_id: Mapped[UUID] = mapped_column(
        ForeignKey("orders.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    provider_fill_id: Mapped[str | None] = mapped_column(String(120), nullable=True, index=True)
    fill_price: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    fill_qty: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    fee: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False, default=Decimal("0"), server_default="0")
    filled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    order: Mapped[Order] = relationship(back_populates="fills")
