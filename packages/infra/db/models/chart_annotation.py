"""Persisted canonical chart annotations."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from packages.infra.db.models.base import Base

if TYPE_CHECKING:
    from packages.infra.db.models.user import User


class ChartAnnotation(Base):
    """Current persisted version of a chart annotation."""

    __tablename__ = "chart_annotations"
    __table_args__ = (
        Index(
            "ix_chart_annotations_owner_scope",
            "owner_user_id",
            "market",
            "symbol",
            "timeframe",
        ),
        Index(
            "ix_chart_annotations_owner_time_window",
            "owner_user_id",
            "time_start",
            "time_end",
        ),
        Index(
            "ix_chart_annotations_deployment_scope",
            "deployment_id",
            "market",
            "symbol",
            "timeframe",
        ),
    )

    owner_user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    actor_user_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
        server_default=text("1"),
    )
    source_type: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    source_id: Mapped[str | None] = mapped_column(String(120), nullable=True, index=True)
    market: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    symbol: Mapped[str] = mapped_column(String(80), nullable=False, index=True)
    timeframe: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    chart_layout_id: Mapped[str | None] = mapped_column(String(120), nullable=True, index=True)
    deployment_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("deployments.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    strategy_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("strategies.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    backtest_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("backtest_jobs.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    anchor_space: Mapped[str] = mapped_column(String(30), nullable=False, default="time_price")
    semantic_kind: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    semantic_role: Mapped[str] = mapped_column(String(30), nullable=False)
    semantic_intent: Mapped[str | None] = mapped_column(String(40), nullable=True)
    semantic_direction: Mapped[str | None] = mapped_column(String(20), nullable=True)
    semantic_status: Mapped[str | None] = mapped_column(String(30), nullable=True)
    tool_family: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    tool_vendor: Mapped[str] = mapped_column(String(40), nullable=False, default="tradingview")
    tool_vendor_type: Mapped[str | None] = mapped_column(String(120), nullable=True, index=True)
    geometry_type: Mapped[str] = mapped_column(String(30), nullable=False)
    group_id: Mapped[str | None] = mapped_column(String(120), nullable=True, index=True)
    parent_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("chart_annotations.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    time_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    time_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    is_deleted: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=text("false"),
        index=True,
    )
    is_editable: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default=text("true"),
    )
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    data: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )
