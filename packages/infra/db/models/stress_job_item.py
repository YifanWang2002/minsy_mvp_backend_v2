"""Stress job result item model."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import CheckConstraint, ForeignKey, Index, Integer, String, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from packages.infra.db.models.base import Base

if TYPE_CHECKING:
    from packages.infra.db.models.stress_job import StressJob


class StressJobItem(Base):
    """One result item attached to a stress job."""

    __tablename__ = "stress_job_items"
    __table_args__ = (
        CheckConstraint(
            "item_type IN ('window', 'trial', 'param_variant', 'pareto_point')",
            name="ck_stress_job_items_item_type",
        ),
        Index("ix_stress_job_items_job_type", "job_id", "item_type"),
    )

    job_id: Mapped[UUID] = mapped_column(
        ForeignKey("stress_jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    item_type: Mapped[str] = mapped_column(String(32), nullable=False)
    item_index: Mapped[int] = mapped_column(Integer, nullable=False)
    labels: Mapped[dict] = mapped_column(
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
    artifacts: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )

    job: Mapped[StressJob] = relationship(back_populates="items")
