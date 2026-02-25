"""Stress testing and optimization job model."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Index, Integer, String, Text, func, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base

if TYPE_CHECKING:
    from src.models.backtest import BacktestJob
    from src.models.optimization_trial import OptimizationTrial
    from src.models.strategy import Strategy
    from src.models.stress_job_item import StressJobItem
    from src.models.user import User


class StressJob(Base):
    """Asynchronous stress/optimization computation unit."""

    __tablename__ = "stress_jobs"
    __table_args__ = (
        CheckConstraint(
            "job_type IN ('black_swan', 'monte_carlo', 'param_scan', 'optimization')",
            name="ck_stress_jobs_job_type",
        ),
        CheckConstraint(
            "status IN ('queued', 'running', 'completed', 'failed', 'cancelled')",
            name="ck_stress_jobs_status",
        ),
        CheckConstraint("progress >= 0 AND progress <= 100", name="ck_stress_jobs_progress"),
        Index("ix_stress_jobs_user_status", "user_id", "status"),
        Index("ix_stress_jobs_strategy_type", "strategy_id", "job_type"),
    )

    user_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    strategy_id: Mapped[UUID] = mapped_column(
        ForeignKey("strategies.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    base_backtest_job_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("backtest_jobs.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    job_type: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="queued",
        server_default="queued",
    )
    progress: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    current_step: Mapped[str | None] = mapped_column(String(64), nullable=True)
    config: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )
    summary: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    submitted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    user: Mapped[User | None] = relationship(back_populates="stress_jobs")
    strategy: Mapped[Strategy] = relationship(back_populates="stress_jobs")
    base_backtest_job: Mapped[BacktestJob | None] = relationship(back_populates="stress_jobs")
    items: Mapped[list[StressJobItem]] = relationship(
        back_populates="job",
        cascade="all, delete-orphan",
    )
    optimization_trials: Mapped[list[OptimizationTrial]] = relationship(
        back_populates="job",
        cascade="all, delete-orphan",
    )
