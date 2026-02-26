"""Optimization trial persistence for stress optimization jobs."""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import Boolean, ForeignKey, Index, Integer, Numeric, String, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from packages.infra.db.models.base import Base

if TYPE_CHECKING:
    from packages.infra.db.models.stress_job import StressJob


class OptimizationTrial(Base):
    """One sampled trial from optimization search."""

    __tablename__ = "optimization_trials"
    __table_args__ = (
        Index("ix_optimization_trials_job_trialno", "job_id", "trial_no"),
        Index("ix_optimization_trials_job_pareto", "job_id", "is_pareto"),
    )

    job_id: Mapped[UUID] = mapped_column(
        ForeignKey("stress_jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    trial_no: Mapped[int] = mapped_column(Integer, nullable=False)
    method: Mapped[str] = mapped_column(String(16), nullable=False)
    params: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )
    metrics: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )
    stability_score: Mapped[Decimal | None] = mapped_column(Numeric(18, 8), nullable=True)
    is_pareto: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=text("false"),
    )

    job: Mapped[StressJob] = relationship(back_populates="optimization_trials")
