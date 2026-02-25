"""Audit rows for broker order status transitions."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Index, String, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base

if TYPE_CHECKING:
    from src.models.order import Order


class OrderStateTransition(Base):
    """Persisted state-machine transition row for one order."""

    __tablename__ = "order_state_transitions"
    __table_args__ = (
        CheckConstraint(
            "from_status IN ('new', 'accepted', 'pending_new', 'partially_filled', "
            "'filled', 'canceled', 'rejected', 'expired')",
            name="ck_order_state_transitions_from_status",
        ),
        CheckConstraint(
            "to_status IN ('new', 'accepted', 'pending_new', 'partially_filled', "
            "'filled', 'canceled', 'rejected', 'expired')",
            name="ck_order_state_transitions_to_status",
        ),
        Index(
            "ix_order_state_transitions_order_transitioned_at",
            "order_id",
            "transitioned_at",
        ),
    )

    order_id: Mapped[UUID] = mapped_column(
        ForeignKey("orders.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    from_status: Mapped[str] = mapped_column(String(30), nullable=False)
    to_status: Mapped[str] = mapped_column(String(30), nullable=False)
    reason: Mapped[str] = mapped_column(String(120), nullable=False)
    metadata_: Mapped[dict] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )
    transitioned_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("now()"),
    )

    order: Mapped[Order] = relationship(back_populates="state_transitions")
