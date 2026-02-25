"""Audit log model for broker account lifecycle changes."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import CheckConstraint, ForeignKey, String, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base

if TYPE_CHECKING:
    from src.models.broker_account import BrokerAccount


class BrokerAccountAuditLog(Base):
    """Audit entries for broker account create/update/validate/deactivate actions."""

    __tablename__ = "broker_account_audit_logs"
    __table_args__ = (
        CheckConstraint(
            "action IN ('create', 'update', 'validate', 'deactivate')",
            name="ck_broker_account_audit_logs_action",
        ),
        CheckConstraint(
            "source IN ('api', 'manual', 'system')",
            name="ck_broker_account_audit_logs_source",
        ),
    )

    broker_account_id: Mapped[UUID] = mapped_column(
        ForeignKey("broker_accounts.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    action: Mapped[str] = mapped_column(String(20), nullable=False)
    source: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="api",
        server_default="api",
    )
    metadata_: Mapped[dict] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )

    broker_account: Mapped[BrokerAccount] = relationship(back_populates="audit_logs")
