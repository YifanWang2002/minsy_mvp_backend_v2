"""Monthly usage aggregates for quota checks."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import Date, ForeignKey, Integer, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from packages.infra.db.models.base import Base

if TYPE_CHECKING:
    from packages.infra.db.models.user import User


class BillingUsageMonthly(Base):
    """Per-user per-month aggregate counters for high-frequency quota checks."""

    __tablename__ = "billing_usage_monthly"
    __table_args__ = (
        UniqueConstraint("user_id", "window_month", name="uq_billing_usage_monthly_user_month"),
    )

    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    window_month: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    ai_input_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    ai_reasoning_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    ai_output_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    ai_total_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    cpu_jobs_total: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")

    user: Mapped[User] = relationship(back_populates="billing_usage_monthly")
