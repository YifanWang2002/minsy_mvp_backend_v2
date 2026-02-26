"""Manual trade action model for user-initiated operations."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, String, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from packages.infra.db.models.base import Base

if TYPE_CHECKING:
    from packages.infra.db.models.deployment import Deployment
    from packages.infra.db.models.user import User


class ManualTradeAction(Base):
    """Queued or executed user manual trade action."""

    __tablename__ = "manual_trade_actions"
    __table_args__ = (
        CheckConstraint(
            "action IN ('open', 'close', 'reduce', 'stop')",
            name="ck_manual_trade_actions_action",
        ),
        CheckConstraint(
            "status IN ('pending', 'accepted', 'executing', 'completed', 'rejected', 'failed')",
            name="ck_manual_trade_actions_status",
        ),
    )

    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    deployment_id: Mapped[UUID] = mapped_column(
        ForeignKey("deployments.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    action: Mapped[str] = mapped_column(String(20), nullable=False)
    payload: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="pending",
        server_default="pending",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("now()"),
    )

    user: Mapped[User] = relationship(back_populates="manual_trade_actions")
    deployment: Mapped[Deployment] = relationship(back_populates="manual_trade_actions")
