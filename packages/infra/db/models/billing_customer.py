"""Stripe customer mapping per user."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import DateTime, ForeignKey, String, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from packages.infra.db.models.base import Base

if TYPE_CHECKING:
    from packages.infra.db.models.billing_subscription import BillingSubscription
    from packages.infra.db.models.user import User


class BillingCustomer(Base):
    """Persistent mapping: user -> Stripe customer."""

    __tablename__ = "billing_customers"
    __table_args__ = (
        UniqueConstraint("user_id", name="uq_billing_customers_user_id"),
        UniqueConstraint("stripe_customer_id", name="uq_billing_customers_stripe_customer_id"),
    )

    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    stripe_customer_id: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        index=True,
    )
    email: Mapped[str | None] = mapped_column(String(320), nullable=True)
    metadata_: Mapped[dict] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )
    synced_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("now()"),
    )

    user: Mapped[User] = relationship(back_populates="billing_customer")
    subscriptions: Mapped[list[BillingSubscription]] = relationship(
        back_populates="customer",
    )
