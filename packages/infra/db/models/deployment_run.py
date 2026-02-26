"""Deployment runtime instance model."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, String, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from packages.infra.db.models.base import Base

if TYPE_CHECKING:
    from packages.infra.db.models.broker_account import BrokerAccount
    from packages.infra.db.models.deployment import Deployment
    from packages.infra.db.models.strategy import Strategy


class DeploymentRun(Base):
    """Runtime status for one deployment instance."""

    __tablename__ = "deployment_runs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('starting', 'running', 'paused', 'stopped', 'error')",
            name="ck_deployment_runs_status",
        ),
    )

    deployment_id: Mapped[UUID] = mapped_column(
        ForeignKey("deployments.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    strategy_id: Mapped[UUID] = mapped_column(
        ForeignKey("strategies.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    broker_account_id: Mapped[UUID] = mapped_column(
        ForeignKey("broker_accounts.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="stopped",
        server_default="stopped",
    )
    last_bar_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    runtime_state: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )

    deployment: Mapped[Deployment] = relationship(back_populates="deployment_runs")
    strategy: Mapped[Strategy] = relationship()
    broker_account: Mapped[BrokerAccount] = relationship(back_populates="deployment_runs")
