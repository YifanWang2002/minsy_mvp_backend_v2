"""Persisted deployment signal events for API history/cursor queries."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import DateTime, ForeignKey, Index, String, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from packages.infra.db.models.base import Base

if TYPE_CHECKING:
    from packages.infra.db.models.deployment import Deployment


class SignalEvent(Base):
    """One generated signal event from runtime/manual execution."""

    __tablename__ = "signal_events"
    __table_args__ = (
        Index("ix_signal_events_deployment_created_at", "deployment_id", "created_at"),
        Index("ix_signal_events_deployment_bar_time", "deployment_id", "bar_time"),
    )

    deployment_id: Mapped[UUID] = mapped_column(
        ForeignKey("deployments.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    symbol: Mapped[str] = mapped_column(String(80), nullable=False, index=True)
    timeframe: Mapped[str] = mapped_column(String(20), nullable=False)
    signal: Mapped[str] = mapped_column(String(40), nullable=False)
    reason: Mapped[str] = mapped_column(String(255), nullable=False, server_default=text("''"))
    bar_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_: Mapped[dict] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )

    deployment: Mapped[Deployment] = relationship(back_populates="signal_events")
