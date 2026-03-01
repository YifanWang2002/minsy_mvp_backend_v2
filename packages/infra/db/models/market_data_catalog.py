"""Catalog metadata for local market-data parquet shards."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Index, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from packages.infra.db.models.base import Base


class MarketDataCatalog(Base):
    """One catalog row per (market, symbol, timeframe, session, year) parquet shard."""

    __tablename__ = "market_data_catalog"
    __table_args__ = (
        Index("ix_market_data_catalog_lookup", "market", "symbol", "timeframe", "year"),
        UniqueConstraint(
            "market",
            "symbol",
            "timeframe",
            "session",
            "year",
            name="uq_market_data_catalog_entry",
        ),
    )

    market: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    symbol: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    timeframe: Mapped[str] = mapped_column(String(16), nullable=False)
    session: Mapped[str] = mapped_column(String(16), nullable=False)
    year: Mapped[int] = mapped_column(Integer, nullable=False)
    start_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    end_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    row_count: Mapped[int] = mapped_column(Integer, nullable=False)
    file_path: Mapped[str] = mapped_column(String(500), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    last_accessed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
