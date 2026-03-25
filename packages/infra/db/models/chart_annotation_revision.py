"""Revision history for chart annotations."""

from __future__ import annotations

from sqlalchemy import ForeignKey, Index, Integer, text
from sqlalchemy.dialects.postgresql import JSONB
from uuid import UUID
from sqlalchemy.orm import Mapped, mapped_column

from packages.infra.db.models.base import Base


class ChartAnnotationRevision(Base):
    """Immutable snapshot of a chart annotation version."""

    __tablename__ = "chart_annotation_revisions"
    __table_args__ = (
        Index(
            "ix_chart_annotation_revisions_annotation_version",
            "annotation_id",
            "version",
            unique=True,
        ),
    )

    annotation_id: Mapped[UUID] = mapped_column(
        ForeignKey("chart_annotations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
        server_default=text("1"),
    )
    snapshot: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )
