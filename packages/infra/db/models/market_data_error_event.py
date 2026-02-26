"""Persisted market-data provider error events for reliability monitoring."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Index, Integer, String, Text, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from packages.infra.db.models.base import Base


class MarketDataErrorEvent(Base):
    """One market-data error record (timeout/404/429/etc)."""

    __tablename__ = "market_data_error_events"
    __table_args__ = (
        Index("ix_market_data_error_events_created_at", "created_at"),
        Index("ix_market_data_error_events_market_symbol", "market", "symbol"),
        Index("ix_market_data_error_events_type_status", "error_type", "http_status"),
    )

    market: Mapped[str] = mapped_column(String(32), nullable=False, default="stocks", server_default="stocks")
    symbol: Mapped[str] = mapped_column(String(64), nullable=False, default="", server_default="")
    error_type: Mapped[str] = mapped_column(String(64), nullable=False, default="", server_default="")
    endpoint: Mapped[str | None] = mapped_column(String(255), nullable=True)
    http_status: Mapped[int | None] = mapped_column(Integer, nullable=True)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("now()"),
    )
    metadata_: Mapped[dict] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )
