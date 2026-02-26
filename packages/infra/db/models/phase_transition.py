"""Phase transition audit model."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import CheckConstraint, ForeignKey, Index, String, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from packages.infra.db.models.base import Base

if TYPE_CHECKING:
    from packages.infra.db.models.session import Session


class PhaseTransition(Base):
    """Track phase transitions for session orchestration."""

    __tablename__ = "phase_transitions"
    __table_args__ = (
        CheckConstraint(
            "trigger IN ('ai_output', 'user_action', 'system')",
            name="ck_phase_transitions_trigger",
        ),
        Index("ix_phase_transitions_session_created", "session_id", "created_at"),
    )

    session_id: Mapped[UUID] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    from_phase: Mapped[str] = mapped_column(String(20), nullable=False)
    to_phase: Mapped[str] = mapped_column(String(20), nullable=False)
    trigger: Mapped[str] = mapped_column(String(20), nullable=False)
    metadata_: Mapped[dict] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )

    session: Mapped[Session] = relationship(back_populates="phase_transitions")
