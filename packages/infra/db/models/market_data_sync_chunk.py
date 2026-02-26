"""Per-range sync chunk records for market-data jobs."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from packages.infra.db.models.base import Base

if TYPE_CHECKING:
    from packages.infra.db.models.market_data_sync_job import MarketDataSyncJob


class MarketDataSyncChunk(Base):
    """One fetched/written chunk inside a sync job."""

    __tablename__ = "market_data_sync_chunks"
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'completed', 'failed')",
            name="ck_market_data_sync_chunks_status",
        ),
        Index("ix_market_data_sync_chunks_job_chunk_index", "job_id", "chunk_index"),
    )

    job_id: Mapped[UUID] = mapped_column(
        ForeignKey("market_data_sync_jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    chunk_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    fetched_rows: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    written_rows: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="pending",
        server_default="pending",
    )
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, nullable=False, default=dict)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    job: Mapped[MarketDataSyncJob] = relationship(back_populates="chunks")
