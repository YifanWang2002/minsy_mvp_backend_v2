"""Incremental market-data import job model."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, Index, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from packages.infra.db.models.base import Base


class MarketDataIncrementalImportJob(Base):
    """One asynchronous incremental import request from local collector to remote VM."""

    __tablename__ = "market_data_incremental_import_jobs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('queued', 'running', 'completed', 'failed')",
            name="ck_market_data_incremental_import_jobs_status",
        ),
        CheckConstraint(
            "file_count >= 0 AND processed_files >= 0 AND rows_written >= 0",
            name="ck_market_data_incremental_import_jobs_counters",
        ),
        Index(
            "ix_market_data_incremental_import_jobs_status_requested",
            "status",
            "requested_at",
        ),
        Index(
            "ix_market_data_incremental_import_jobs_run_id",
            "run_id",
            unique=True,
        ),
    )

    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    bucket: Mapped[str] = mapped_column(String(255), nullable=False)
    prefix: Mapped[str] = mapped_column(String(512), nullable=False)
    manifest_object: Mapped[str] = mapped_column(String(1024), nullable=False)
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="queued",
        server_default="queued",
    )
    file_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    processed_files: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
    )
    rows_written: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    requested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
