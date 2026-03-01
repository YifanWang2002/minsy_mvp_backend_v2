"""Broker account model for execution providers."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    String,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from packages.infra.db.models.base import Base

if TYPE_CHECKING:
    from packages.infra.db.models.broker_account_audit_log import BrokerAccountAuditLog
    from packages.infra.db.models.deployment_run import DeploymentRun
    from packages.infra.db.models.user import User


class BrokerAccount(Base):
    """User-owned broker credentials and provider metadata."""

    __tablename__ = "broker_accounts"
    __table_args__ = (
        CheckConstraint("provider IN ('alpaca', 'ccxt', 'sandbox')", name="ck_broker_accounts_provider"),
        CheckConstraint("mode = 'paper'", name="ck_broker_accounts_mode_paper_only"),
        CheckConstraint(
            "status IN ('active', 'inactive', 'error')",
            name="ck_broker_accounts_status",
        ),
        CheckConstraint(
            "updated_source IN ('api', 'manual', 'system')",
            name="ck_broker_accounts_updated_source",
        ),
        # Keep one active default broker per user+mode (paper today).
        Index(
            "uq_broker_accounts_user_mode_default_active",
            "user_id",
            "mode",
            unique=True,
            postgresql_where=text("is_default = true AND status = 'active'"),
        ),
        # One active account identity per user/provider/exchange/account_uid.
        Index(
            "uq_broker_accounts_user_provider_exchange_account_uid_active",
            "user_id",
            "provider",
            "exchange_id",
            "account_uid",
            unique=True,
            postgresql_where=text("status = 'active'"),
        ),
    )

    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    provider: Mapped[str] = mapped_column(String(20), nullable=False)
    exchange_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default="",
        server_default="",
    )
    account_uid: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        default="",
        server_default="",
    )
    mode: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="paper",
        server_default="paper",
    )
    encrypted_credentials: Mapped[str] = mapped_column(Text, nullable=False)
    key_fingerprint: Mapped[str | None] = mapped_column(String(128), nullable=True)
    encryption_version: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="fernet_v1",
        server_default="fernet_v1",
    )
    updated_source: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="api",
        server_default="api",
    )
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="active",
        server_default="active",
    )
    is_default: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=text("false"),
    )
    is_sandbox: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=text("false"),
    )
    last_validated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    last_validated_status: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )
    last_validation_error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    last_error: Mapped[str | None] = mapped_column(String(500), nullable=True)
    capabilities: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )
    validation_metadata: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )
    metadata_: Mapped[dict] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )

    user: Mapped[User] = relationship(back_populates="broker_accounts")
    deployment_runs: Mapped[list[DeploymentRun]] = relationship(
        back_populates="broker_account",
    )
    audit_logs: Mapped[list[BrokerAccountAuditLog]] = relationship(
        back_populates="broker_account",
        cascade="all, delete-orphan",
    )
