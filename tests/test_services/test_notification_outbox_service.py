from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy import select

from src.config import settings
from src.models.notification_delivery_attempt import NotificationDeliveryAttempt
from src.models.notification_outbox import NotificationOutbox
from src.models.social_connector import SocialConnectorBinding
from src.models.user import User
from src.services.im_channels import IMDeliveryResult, IMProviderRegistry
from src.services.notification_events import EVENT_POSITION_OPENED
from src.services.notification_outbox_service import NotificationOutboxService


class _SuccessProvider:
    channel = "telegram"

    async def send_event(self, **kwargs):  # noqa: ANN003, D401
        return IMDeliveryResult(
            success=True,
            provider="telegram",
            provider_message_id="msg-1",
            request_payload={"ok": True},
            response_payload={"ok": True},
        )


@pytest.mark.asyncio
async def test_dispatch_due_sends_notification_when_binding_exists(db_session, monkeypatch) -> None:
    monkeypatch.setattr(settings, "notifications_enabled", True)
    monkeypatch.setattr(settings, "telegram_enabled", True)

    user = User(
        email=f"notify_{uuid4().hex}@test.com",
        password_hash="hashed",
        name="Notify User",
        is_active=True,
    )
    db_session.add(user)
    await db_session.flush()
    db_session.add(
        SocialConnectorBinding(
            user_id=user.id,
            provider="telegram",
            external_user_id="u1",
            external_chat_id="chat_1",
            external_username="notify_user",
            status="connected",
            bound_at=datetime.now(UTC),
            metadata_={"locale": "en"},
        )
    )
    await db_session.flush()

    registry = IMProviderRegistry()
    registry.register(_SuccessProvider())
    service = NotificationOutboxService(db_session, provider_registry=registry)
    rows = await service.enqueue_event_for_user(
        user_id=user.id,
        event_type=EVENT_POSITION_OPENED,
        event_key=f"position_opened:{uuid4()}",
        payload={"symbol": "AAPL", "qty": 1},
    )
    assert len(rows) == 1

    stats = await service.dispatch_due(limit=10)
    await db_session.commit()
    assert stats["picked"] == 1
    assert stats["sent"] == 1

    outbox = rows[0]
    await db_session.refresh(outbox)
    assert outbox.status == "sent"
    attempts = (
        await db_session.scalars(
            select(NotificationDeliveryAttempt).where(
                NotificationDeliveryAttempt.outbox_id == outbox.id
            )
        )
    ).all()
    assert len(attempts) == 1


@pytest.mark.asyncio
async def test_dispatch_due_marks_failed_when_binding_missing(db_session, monkeypatch) -> None:
    monkeypatch.setattr(settings, "notifications_enabled", True)
    monkeypatch.setattr(settings, "telegram_enabled", True)
    monkeypatch.setattr(settings, "notifications_retry_backoff_seconds", 1.0)

    user = User(
        email=f"notify_missing_{uuid4().hex}@test.com",
        password_hash="hashed",
        name="Notify User",
        is_active=True,
    )
    db_session.add(user)
    await db_session.flush()

    registry = IMProviderRegistry()
    registry.register(_SuccessProvider())
    service = NotificationOutboxService(db_session, provider_registry=registry)
    rows = await service.enqueue_event_for_user(
        user_id=user.id,
        event_type=EVENT_POSITION_OPENED,
        event_key=f"position_opened:{uuid4()}",
        payload={"symbol": "AAPL", "qty": 1},
    )
    assert len(rows) == 1

    stats = await service.dispatch_due(limit=10)
    await db_session.commit()
    assert stats["picked"] == 1
    assert stats["failed"] == 1

    outbox = await db_session.get(NotificationOutbox, rows[0].id)
    assert outbox is not None
    assert outbox.status == "failed"
    assert outbox.retry_count == 1
    assert outbox.next_retry_at is not None
