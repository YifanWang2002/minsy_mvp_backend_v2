"""Stripe subscription state synchronized to local DB."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    String,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from packages.infra.db.models.base import Base

if TYPE_CHECKING:
    from packages.infra.db.models.billing_customer import BillingCustomer
    from packages.infra.db.models.user import User


class BillingSubscription(Base):
    """One Stripe subscription record attached to a user."""

    __tablename__ = "billing_subscriptions"
    __table_args__ = (
        UniqueConstraint(
            "stripe_subscription_id",
            name="uq_billing_subscriptions_stripe_subscription_id",
        ),
        CheckConstraint(
            "tier IN ('free', 'go', 'plus', 'pro')",
            name="ck_billing_subscriptions_tier",
        ),
        CheckConstraint(
            "pending_tier IS NULL OR pending_tier IN ('free', 'go', 'plus', 'pro')",
            name="ck_billing_subscriptions_pending_tier",
        ),
        Index(
            "ix_billing_subscriptions_user_status",
            "user_id",
            "status",
        ),
        Index(
            "ix_billing_subscriptions_customer",
            "stripe_customer_id",
        ),
    )

    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    customer_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("billing_customers.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    stripe_customer_id: Mapped[str] = mapped_column(String(128), nullable=False)
    stripe_subscription_id: Mapped[str] = mapped_column(String(128), nullable=False)
    stripe_price_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    tier: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="free",
        server_default="free",
    )
    status: Mapped[str] = mapped_column(
        String(40),
        nullable=False,
        default="inactive",
        server_default="inactive",
    )
    cancel_at_period_end: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=text("false"),
    )
    current_period_start: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    current_period_end: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    trial_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    trial_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    canceled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    pending_price_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    pending_tier: Mapped[str | None] = mapped_column(String(20), nullable=True)
    latest_invoice_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    latest_event_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    raw_payload: Mapped[dict] = mapped_column(
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

    user: Mapped[User] = relationship(back_populates="billing_subscriptions")
    customer: Mapped[BillingCustomer | None] = relationship(back_populates="subscriptions")
