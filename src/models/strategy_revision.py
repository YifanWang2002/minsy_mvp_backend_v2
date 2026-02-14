"""Strategy revision history model."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import (
    CheckConstraint,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base

if TYPE_CHECKING:
    from src.models.strategy import Strategy


class StrategyRevision(Base):
    """Immutable versioned snapshots for one strategy."""

    __tablename__ = "strategy_revisions"
    __table_args__ = (
        CheckConstraint(
            "version >= 1",
            name="ck_strategy_revisions_version_positive",
        ),
        CheckConstraint(
            "change_type IN ('create', 'upsert', 'patch', 'rollback', 'bootstrap')",
            name="ck_strategy_revisions_change_type",
        ),
        UniqueConstraint(
            "strategy_id",
            "version",
            name="uq_strategy_revisions_strategy_version",
        ),
        Index("ix_strategy_revisions_strategy_created", "strategy_id", "created_at"),
    )

    strategy_id: Mapped[UUID] = mapped_column(
        ForeignKey("strategies.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    session_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("sessions.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    dsl_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    dsl_payload: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )
    payload_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    change_type: Mapped[str] = mapped_column(String(20), nullable=False)
    source_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    patch_ops: Mapped[list[dict] | None] = mapped_column(JSONB, nullable=True)

    strategy: Mapped[Strategy] = relationship(back_populates="revisions")
