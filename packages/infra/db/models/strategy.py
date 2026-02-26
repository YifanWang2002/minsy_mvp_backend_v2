"""Strategy model definitions."""

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
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from packages.infra.db.models.base import Base

if TYPE_CHECKING:
    from packages.infra.db.models.backtest import BacktestJob
    from packages.infra.db.models.deployment import Deployment
    from packages.infra.db.models.session import Session
    from packages.infra.db.models.stress_job import StressJob
    from packages.infra.db.models.strategy_revision import StrategyRevision
    from packages.infra.db.models.user import User


class Strategy(Base):
    """Trading strategy DSL and metadata."""

    __tablename__ = "strategies"
    __table_args__ = (
        CheckConstraint(
            "status IN ('draft', 'validated', 'backtested', 'deployed', 'archived')",
            name="ck_strategies_status",
        ),
        Index("ix_strategies_user_session", "user_id", "session_id"),
    )

    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    session_id: Mapped[UUID] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    strategy_type: Mapped[str] = mapped_column(String(80), nullable=False)
    symbols: Mapped[list[str]] = mapped_column(ARRAY(String()), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(20), nullable=False)
    parameters: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )
    entry_rules: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )
    exit_rules: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )
    risk_management: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )
    dsl_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    dsl_payload: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )
    last_validated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        server_default=func.now(),
    )
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="draft",
        server_default="draft",
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default="1")

    user: Mapped[User] = relationship(back_populates="strategies")
    session: Mapped[Session] = relationship(back_populates="strategies")
    backtest_jobs: Mapped[list[BacktestJob]] = relationship(back_populates="strategy")
    stress_jobs: Mapped[list[StressJob]] = relationship(back_populates="strategy")
    deployments: Mapped[list[Deployment]] = relationship(back_populates="strategy")
    revisions: Mapped[list[StrategyRevision]] = relationship(
        back_populates="strategy",
        cascade="all, delete-orphan",
    )
