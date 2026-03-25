"""Outbox for realtime chart annotation replay and pubsub."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import BigInteger, ForeignKey, Identity, Index, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from packages.infra.db.models.base import Base


class ChartAnnotationOutbox(Base):
    """Append-only event log for chart annotation updates."""

    __tablename__ = "chart_annotation_outbox"
    __table_args__ = (
        Index(
            "ix_chart_annotation_outbox_owner_seq",
            "owner_user_id",
            "event_seq",
        ),
        Index(
            "ix_chart_annotation_outbox_owner_scope_seq",
            "owner_user_id",
            "market",
            "symbol",
            "timeframe",
            "event_seq",
        ),
    )

    owner_user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    annotation_id: Mapped[UUID] = mapped_column(
        ForeignKey("chart_annotations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    event_seq: Mapped[int] = mapped_column(
        BigInteger,
        Identity(always=False),
        nullable=False,
        unique=True,
        index=True,
    )
    event_type: Mapped[str] = mapped_column(String(40), nullable=False)
    market: Mapped[str] = mapped_column(String(20), nullable=False)
    symbol: Mapped[str] = mapped_column(String(80), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(20), nullable=False)
    chart_layout_id: Mapped[str | None] = mapped_column(String(120), nullable=True, index=True)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
