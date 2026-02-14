"""Session and message models for multi-phase conversation state."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    String,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base

if TYPE_CHECKING:
    from src.models.backtest import BacktestJob
    from src.models.phase_transition import PhaseTransition
    from src.models.strategy import Strategy
    from src.models.user import User


class Session(Base):
    """User workflow session that spans kyc -> pre_strategy -> strategy -> stress_test -> deployment."""

    __tablename__ = "sessions"
    __table_args__ = (
        CheckConstraint(
            "current_phase IN "
            "('kyc', 'pre_strategy', 'strategy', 'stress_test', 'deployment', 'completed', 'error')",
            name="ck_sessions_current_phase",
        ),
        CheckConstraint(
            "status IN ('active', 'paused', 'completed', 'error')",
            name="ck_sessions_status",
        ),
        Index("ix_sessions_user_status", "user_id", "status"),
        Index("ix_sessions_user_archived_updated", "user_id", "archived_at", "updated_at"),
    )

    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    parent_session_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("sessions.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    current_phase: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="kyc",
        server_default="kyc",
    )
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="active",
        server_default="active",
    )
    archived_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    previous_response_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    artifacts: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )
    metadata_: Mapped[dict] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )
    last_activity_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        server_default=func.now(),
    )

    user: Mapped[User] = relationship(back_populates="sessions")
    parent_session: Mapped[Session | None] = relationship(
        "Session",
        remote_side="Session.id",
        back_populates="child_sessions",
    )
    child_sessions: Mapped[list[Session]] = relationship(back_populates="parent_session")
    messages: Mapped[list[Message]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
    )
    strategies: Mapped[list[Strategy]] = relationship(back_populates="session")
    backtest_jobs: Mapped[list[BacktestJob]] = relationship(back_populates="session")
    phase_transitions: Mapped[list[PhaseTransition]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
    )


class Message(Base):
    """Conversation message persisted by session."""

    __tablename__ = "messages"
    __table_args__ = (
        CheckConstraint(
            "role IN ('user', 'assistant', 'system', 'tool')",
            name="ck_messages_role",
        ),
        Index("ix_messages_session_created", "session_id", "created_at"),
    )

    session_id: Mapped[UUID] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    phase: Mapped[str] = mapped_column(String(20), nullable=False)
    response_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    tool_calls: Mapped[list[dict] | None] = mapped_column(JSONB, nullable=True)
    token_usage: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    session: Mapped[Session] = relationship(back_populates="messages")
