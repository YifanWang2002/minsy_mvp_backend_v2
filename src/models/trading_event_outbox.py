"""Persistent outbox for deployment SSE event replay/resume."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Identity,
    Index,
    String,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from src.models.base import Base


class TradingEventOutbox(Base):
    """Append-only event stream log for one deployment."""

    __tablename__ = "trading_event_outbox"
    __table_args__ = (
        CheckConstraint(
            "event_type IN ('deployment_status', 'order_update', 'fill_update', "
            "'position_update', 'pnl_update', 'trade_approval_update', 'heartbeat')",
            name="ck_trading_event_outbox_event_type",
        ),
        Index("ix_trading_event_outbox_deployment_seq", "deployment_id", "event_seq"),
    )

    deployment_id: Mapped[UUID] = mapped_column(
        ForeignKey("deployments.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    event_seq: Mapped[int] = mapped_column(
        BigInteger,
        Identity(always=False),
        nullable=False,
        index=True,
        unique=True,
    )
    event_type: Mapped[str] = mapped_column(String(40), nullable=False)
    payload: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("now()"),
    )
