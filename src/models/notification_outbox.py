"""Notification outbox for asynchronous IM channel delivery."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base

if TYPE_CHECKING:
    from src.models.notification_delivery_attempt import NotificationDeliveryAttempt
    from src.models.user import User


class NotificationOutbox(Base):
    """Persistent event record for asynchronous notification delivery."""

    __tablename__ = "notification_outbox"
    __table_args__ = (
        CheckConstraint(
            "channel IN ('telegram', 'discord', 'slack', 'whatsapp')",
            name="ck_notification_outbox_channel",
        ),
        CheckConstraint(
            "status IN ('pending', 'sending', 'sent', 'failed', 'dead')",
            name="ck_notification_outbox_status",
        ),
        CheckConstraint("retry_count >= 0", name="ck_notification_outbox_retry_count_non_negative"),
        UniqueConstraint("channel", "event_key", name="uq_notification_outbox_channel_event_key"),
        Index("ix_notification_outbox_status_next_retry", "status", "next_retry_at"),
        Index("ix_notification_outbox_user_created", "user_id", "created_at"),
    )

    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    channel: Mapped[str] = mapped_column(String(32), nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    event_key: Mapped[str] = mapped_column(String(255), nullable=False)
    payload: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="pending",
        server_default="pending",
    )
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    max_retries: Mapped[int] = mapped_column(Integer, nullable=False, default=3, server_default="3")
    scheduled_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("now()"),
        index=True,
    )
    sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    next_retry_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)

    user: Mapped[User] = relationship(back_populates="notification_outbox_items")
    delivery_attempts: Mapped[list[NotificationDeliveryAttempt]] = relationship(
        back_populates="outbox",
        cascade="all, delete-orphan",
    )
