"""Backtest job model."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base

if TYPE_CHECKING:
    from src.models.deployment import Deployment
    from src.models.session import Session
    from src.models.stress_job import StressJob
    from src.models.strategy import Strategy
    from src.models.user import User


class BacktestJob(Base):
    """Asynchronous backtest execution unit."""

    __tablename__ = "backtest_jobs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('queued', 'running', 'completed', 'failed', 'cancelled')",
            name="ck_backtest_jobs_status",
        ),
        CheckConstraint("progress >= 0 AND progress <= 100", name="ck_backtest_jobs_progress"),
        Index("ix_backtest_jobs_user_status", "user_id", "status"),
    )

    strategy_id: Mapped[UUID] = mapped_column(
        ForeignKey("strategies.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    session_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("sessions.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="queued",
        server_default="queued",
    )
    progress: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    current_step: Mapped[str | None] = mapped_column(String(40), nullable=True)
    config: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )
    results: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    submitted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    strategy: Mapped[Strategy] = relationship(back_populates="backtest_jobs")
    user: Mapped[User] = relationship(back_populates="backtest_jobs")
    session: Mapped[Session | None] = relationship(back_populates="backtest_jobs")
    deployments: Mapped[list[Deployment]] = relationship(back_populates="backtest_job")
    stress_jobs: Mapped[list[StressJob]] = relationship(back_populates="base_backtest_job")
