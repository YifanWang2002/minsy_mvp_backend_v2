"""Deployment model."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Numeric,
    String,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base

if TYPE_CHECKING:
    from src.models.backtest import BacktestJob
    from src.models.deployment_run import DeploymentRun
    from src.models.manual_trade_action import ManualTradeAction
    from src.models.order import Order
    from src.models.pnl_snapshot import PnlSnapshot
    from src.models.position import Position
    from src.models.signal_event import SignalEvent
    from src.models.strategy import Strategy
    from src.models.trade_approval_request import TradeApprovalRequest
    from src.models.user import User


class Deployment(Base):
    """Deployment record for paper/live strategy runtime."""

    __tablename__ = "deployments"
    __table_args__ = (
        CheckConstraint("mode IN ('paper', 'live')", name="ck_deployments_mode"),
        CheckConstraint(
            "status IN ('pending', 'active', 'paused', 'stopped', 'error')",
            name="ck_deployments_status",
        ),
        CheckConstraint(
            "capital_allocated >= 0",
            name="ck_deployments_capital_allocated_non_negative",
        ),
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
    backtest_job_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("backtest_jobs.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    mode: Mapped[str] = mapped_column(String(20), nullable=False, default="paper", server_default="paper")
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="pending",
        server_default="pending",
    )
    risk_limits: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )
    capital_allocated: Mapped[Decimal] = mapped_column(
        Numeric(18, 2),
        nullable=False,
        default=Decimal("0"),
        server_default="0",
    )
    deployed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    stopped_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    strategy: Mapped[Strategy] = relationship(back_populates="deployments")
    user: Mapped[User] = relationship(back_populates="deployments")
    backtest_job: Mapped[BacktestJob | None] = relationship(back_populates="deployments")
    deployment_runs: Mapped[list[DeploymentRun]] = relationship(
        back_populates="deployment",
        cascade="all, delete-orphan",
    )
    orders: Mapped[list[Order]] = relationship(
        back_populates="deployment",
        cascade="all, delete-orphan",
    )
    positions: Mapped[list[Position]] = relationship(
        back_populates="deployment",
        cascade="all, delete-orphan",
    )
    pnl_snapshots: Mapped[list[PnlSnapshot]] = relationship(
        back_populates="deployment",
        cascade="all, delete-orphan",
    )
    manual_trade_actions: Mapped[list[ManualTradeAction]] = relationship(
        back_populates="deployment",
        cascade="all, delete-orphan",
    )
    signal_events: Mapped[list[SignalEvent]] = relationship(
        back_populates="deployment",
        cascade="all, delete-orphan",
    )
    trade_approval_requests: Mapped[list[TradeApprovalRequest]] = relationship(
        back_populates="deployment",
        cascade="all, delete-orphan",
    )
