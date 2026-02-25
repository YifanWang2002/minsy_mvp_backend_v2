"""Order model for broker execution state."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Numeric,
    String,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base

if TYPE_CHECKING:
    from src.models.deployment import Deployment
    from src.models.fill import Fill
    from src.models.order_state_transition import OrderStateTransition


class Order(Base):
    """Broker order lifecycle record."""

    __tablename__ = "orders"
    __table_args__ = (
        CheckConstraint("side IN ('buy', 'sell')", name="ck_orders_side"),
        CheckConstraint(
            "type IN ('market', 'limit', 'stop', 'stop_limit')",
            name="ck_orders_type",
        ),
        CheckConstraint(
            "status IN "
            "('new', 'accepted', 'pending_new', 'partially_filled', "
            "'filled', 'canceled', 'rejected', 'expired')",
            name="ck_orders_status",
        ),
        CheckConstraint("qty > 0", name="ck_orders_qty_positive"),
        CheckConstraint("price IS NULL OR price >= 0", name="ck_orders_price_non_negative"),
        Index("ix_orders_deployment_status", "deployment_id", "status"),
        Index("ix_orders_client_order_id", "client_order_id", unique=True),
    )

    deployment_id: Mapped[UUID] = mapped_column(
        ForeignKey("deployments.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    provider_order_id: Mapped[str | None] = mapped_column(String(120), nullable=True, index=True)
    client_order_id: Mapped[str] = mapped_column(String(120), nullable=False)
    symbol: Mapped[str] = mapped_column(String(80), nullable=False, index=True)
    side: Mapped[str] = mapped_column(String(10), nullable=False)
    type: Mapped[str] = mapped_column(String(20), nullable=False, default="market", server_default="market")
    qty: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    price: Mapped[Decimal | None] = mapped_column(Numeric(20, 8), nullable=True)
    status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default="new",
        server_default="new",
    )
    reject_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    provider_updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_sync_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    submitted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("now()"),
    )
    metadata_: Mapped[dict] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )

    deployment: Mapped[Deployment] = relationship(back_populates="orders")
    fills: Mapped[list[Fill]] = relationship(back_populates="order", cascade="all, delete-orphan")
    state_transitions: Mapped[list[OrderStateTransition]] = relationship(
        back_populates="order",
        cascade="all, delete-orphan",
    )
