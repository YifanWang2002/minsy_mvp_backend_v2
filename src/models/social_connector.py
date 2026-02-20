"""Social connector binding and activity models."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base

if TYPE_CHECKING:
    from src.models.user import User


class SocialConnectorBinding(Base):
    """One connector binding per provider/user."""

    __tablename__ = "social_connector_bindings"
    __table_args__ = (
        CheckConstraint(
            "provider IN ('telegram', 'discord', 'slack', 'whatsapp')",
            name="ck_social_connector_bindings_provider",
        ),
        CheckConstraint(
            "status IN ('connected', 'disconnected')",
            name="ck_social_connector_bindings_status",
        ),
        UniqueConstraint(
            "provider",
            "user_id",
            name="uq_social_connector_bindings_provider_user",
        ),
        UniqueConstraint(
            "provider",
            "external_chat_id",
            name="uq_social_connector_bindings_provider_chat",
        ),
        Index(
            "ix_social_connector_bindings_provider_status",
            "provider",
            "status",
        ),
    )

    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    external_user_id: Mapped[str] = mapped_column(String(128), nullable=False)
    external_chat_id: Mapped[str] = mapped_column(String(128), nullable=False)
    external_username: Mapped[str | None] = mapped_column(String(255), nullable=True)
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="connected",
        server_default="connected",
    )
    bound_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        server_default=func.now(),
    )
    metadata_: Mapped[dict] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )

    user: Mapped[User] = relationship(back_populates="social_connector_bindings")


class SocialConnectorLinkIntent(Base):
    """One-time connect-link intent token storage."""

    __tablename__ = "social_connector_link_intents"
    __table_args__ = (
        CheckConstraint(
            "provider IN ('telegram', 'discord', 'slack', 'whatsapp')",
            name="ck_social_connector_link_intents_provider",
        ),
        UniqueConstraint(
            "provider",
            "token_hash",
            name="uq_social_connector_link_intents_provider_hash",
        ),
        Index(
            "ix_social_connector_link_intents_user_provider_expires",
            "user_id",
            "provider",
            "expires_at",
        ),
    )

    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    token_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    consumed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    metadata_: Mapped[dict] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )

    user: Mapped[User] = relationship(back_populates="social_connector_link_intents")


class SocialConnectorActivity(Base):
    """Inbound connector interactions persisted for timeline rendering."""

    __tablename__ = "social_connector_activities"
    __table_args__ = (
        CheckConstraint(
            "provider IN ('telegram', 'discord', 'slack', 'whatsapp')",
            name="ck_social_connector_activities_provider",
        ),
        CheckConstraint(
            "event_type IN ('choice', 'text')",
            name="ck_social_connector_activities_event_type",
        ),
        UniqueConstraint(
            "provider",
            "external_update_id",
            name="uq_social_connector_activities_provider_update",
        ),
        Index(
            "ix_social_connector_activities_user_provider_created",
            "user_id",
            "provider",
            "created_at",
        ),
    )

    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    event_type: Mapped[str] = mapped_column(String(16), nullable=False)
    choice_value: Mapped[str | None] = mapped_column(String(32), nullable=True)
    message_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    external_update_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    payload: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )

    user: Mapped[User] = relationship(back_populates="social_connector_activities")
