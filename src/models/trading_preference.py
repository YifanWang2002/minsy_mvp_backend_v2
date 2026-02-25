"""User trading execution preference model."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import CheckConstraint, ForeignKey, Integer, String, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base

if TYPE_CHECKING:
    from src.models.user import User


class TradingPreference(Base):
    """Execution-mode preference row for one user."""

    __tablename__ = "trading_preferences"
    __table_args__ = (
        CheckConstraint(
            "execution_mode IN ('auto_execute', 'approval_required')",
            name="ck_trading_preferences_execution_mode",
        ),
        CheckConstraint(
            "approval_channel IN ('telegram', 'discord', 'slack', 'whatsapp')",
            name="ck_trading_preferences_approval_channel",
        ),
        CheckConstraint(
            "approval_timeout_seconds > 0",
            name="ck_trading_preferences_timeout_positive",
        ),
        CheckConstraint(
            "approval_scope IN ('open_only', 'open_and_close')",
            name="ck_trading_preferences_approval_scope",
        ),
    )

    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    execution_mode: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="auto_execute",
        server_default="auto_execute",
    )
    approval_channel: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="telegram",
        server_default="telegram",
    )
    approval_timeout_seconds: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=120,
        server_default="120",
    )
    approval_scope: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="open_only",
        server_default="open_only",
    )

    user: Mapped[User] = relationship(back_populates="trading_preference")
