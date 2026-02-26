"""Outbox enqueue/dispatch service for IM notifications."""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from packages.shared_settings.schema.settings import settings
from packages.infra.db.models.notification_delivery_attempt import NotificationDeliveryAttempt
from packages.infra.db.models.notification_outbox import NotificationOutbox
from packages.infra.providers.im.im_channels import IMProviderRegistry
from packages.domain.user.services.social_connector_service import SocialConnectorService
from packages.infra.providers.telegram.notification_provider import TelegramNotificationProvider
from packages.domain.notification.services.user_notification_preference_service import (
    UserNotificationPreferenceService,
)
from packages.infra.observability.logger import logger

_DISPATCHABLE_STATUSES: tuple[str, ...] = ("pending", "failed")


class NotificationOutboxService:
    """Persist and dispatch notification events to IM providers."""

    def __init__(
        self,
        db: AsyncSession,
        *,
        provider_registry: IMProviderRegistry | None = None,
    ) -> None:
        self.db = db
        self.social_service = SocialConnectorService(db)
        self.preference_service = UserNotificationPreferenceService(db)
        self.provider_registry = provider_registry or _build_default_provider_registry()

    async def enqueue_event(
        self,
        *,
        user_id: UUID,
        channel: str,
        event_type: str,
        event_key: str,
        payload: dict[str, Any] | None = None,
        scheduled_at: datetime | None = None,
    ) -> NotificationOutbox:
        normalized_channel = str(channel).strip().lower()
        normalized_event_key = str(event_key).strip()
        if not normalized_channel:
            raise ValueError("channel cannot be empty.")
        if not normalized_event_key:
            raise ValueError("event_key cannot be empty.")

        existing = await self.db.scalar(
            select(NotificationOutbox).where(
                NotificationOutbox.channel == normalized_channel,
                NotificationOutbox.event_key == normalized_event_key,
            )
        )
        if existing is not None:
            return existing

        row = NotificationOutbox(
            user_id=user_id,
            channel=normalized_channel,
            event_type=str(event_type).strip().upper(),
            event_key=normalized_event_key,
            payload=dict(payload or {}),
            status="pending",
            retry_count=0,
            max_retries=max(0, int(settings.notifications_retry_max_attempts)),
            scheduled_at=scheduled_at or datetime.now(UTC),
            next_retry_at=None,
            last_error=None,
        )
        self.db.add(row)
        await self.db.flush()
        return row

    async def enqueue_event_for_user(
        self,
        *,
        user_id: UUID,
        event_type: str,
        event_key: str,
        payload: dict[str, Any] | None = None,
        channels: list[str] | None = None,
    ) -> list[NotificationOutbox]:
        selected_channels = channels or await self._resolve_default_channels(user_id=user_id)
        if not selected_channels:
            return []
        rows: list[NotificationOutbox] = []
        for channel in selected_channels:
            row = await self.enqueue_event(
                user_id=user_id,
                channel=channel,
                event_type=event_type,
                event_key=event_key,
                payload=payload,
            )
            rows.append(row)
        return rows

    async def dispatch_due(
        self,
        *,
        limit: int | None = None,
    ) -> dict[str, int]:
        if not settings.notifications_enabled:
            return {"picked": 0, "sent": 0, "failed": 0, "dead": 0, "deferred": 0}

        now = datetime.now(UTC)
        batch_limit = max(1, int(limit or settings.notifications_dispatch_batch_size))
        rows = await self._claim_due_outbox_rows(now=now, limit=batch_limit)
        stats = {"picked": len(rows), "sent": 0, "failed": 0, "dead": 0, "deferred": 0}
        started_at = time.monotonic()
        max_runtime_seconds = max(0.5, float(settings.notifications_dispatch_max_runtime_seconds))
        for index, row in enumerate(rows):
            if (time.monotonic() - started_at) >= max_runtime_seconds:
                deferred_rows = rows[index:]
                for deferred in deferred_rows:
                    if deferred.status == "sending":
                        deferred.status = "pending"
                        deferred.next_retry_at = None
                stats["deferred"] += len(deferred_rows)
                break

            before = row.status
            try:
                await self._deliver_one(row=row, now=datetime.now(UTC))
            except Exception as exc:  # noqa: BLE001
                await self._mark_failed(
                    row=row,
                    now=datetime.now(UTC),
                    error_code=type(exc).__name__,
                    error_message=str(exc) or "unexpected_notification_dispatch_error",
                )
            if row.status == "sent":
                stats["sent"] += 1
            elif row.status == "dead":
                stats["dead"] += 1
            elif row.status == "failed":
                stats["failed"] += 1
            elif before == "sending" and row.status != "sent":
                stats["failed"] += 1
        return stats

    async def _claim_due_outbox_rows(
        self,
        *,
        now: datetime,
        limit: int,
    ) -> list[NotificationOutbox]:
        stale_sending_cutoff = now - timedelta(
            seconds=max(
                float(settings.notifications_dispatch_lock_ttl_seconds),
                float(settings.notifications_dispatch_max_runtime_seconds),
            )
            + float(settings.notifications_delivery_timeout_seconds),
        )
        rows = (
            await self.db.scalars(
                select(NotificationOutbox)
                .where(
                    or_(
                        NotificationOutbox.status.in_(_DISPATCHABLE_STATUSES),
                        and_(
                            NotificationOutbox.status == "sending",
                            NotificationOutbox.updated_at <= stale_sending_cutoff,
                        ),
                    ),
                    NotificationOutbox.scheduled_at <= now,
                    or_(
                        NotificationOutbox.next_retry_at.is_(None),
                        NotificationOutbox.next_retry_at <= now,
                    ),
                )
                .order_by(NotificationOutbox.created_at.asc())
                .limit(limit)
                .with_for_update(skip_locked=True),
            )
        ).all()
        output = list(rows)
        for row in output:
            row.status = "sending"
        await self.db.flush()
        return output

    async def _deliver_one(
        self,
        *,
        row: NotificationOutbox,
        now: datetime,
    ) -> None:
        provider = self.provider_registry.get(row.channel)
        if provider is None:
            await self._mark_failed(
                row=row,
                now=now,
                error_code="UNSUPPORTED_CHANNEL",
                error_message=f"No provider registered for channel={row.channel}",
            )
            return

        enabled = await self.preference_service.is_event_enabled(
            user_id=row.user_id,
            event_type=row.event_type,
            channel=row.channel,
        )
        if not enabled:
            self.db.add(
                NotificationDeliveryAttempt(
                    outbox_id=row.id,
                    provider=row.channel,
                    provider_message_id=None,
                    request_payload={"skipped": True},
                    response_payload={"reason": "preference_disabled"},
                    success=True,
                    error_code=None,
                    error_message=None,
                    attempted_at=now,
                )
            )
            row.status = "sent"
            row.sent_at = now
            row.next_retry_at = None
            row.last_error = None
            return

        binding = await self.social_service.get_connected_binding_for_user(
            user_id=row.user_id,
            provider=row.channel,
        )
        if binding is None:
            await self._mark_failed(
                row=row,
                now=now,
                error_code="BINDING_NOT_FOUND",
                error_message=f"No connected binding for user={row.user_id} channel={row.channel}",
            )
            return

        fallback_locale = None
        if isinstance(binding.metadata_, dict):
            raw_locale = binding.metadata_.get("locale")
            if isinstance(raw_locale, str):
                fallback_locale = raw_locale
        locale = await self.social_service.resolve_user_locale(
            user_id=row.user_id,
            fallback_locale=fallback_locale,
        )
        delivery = await provider.send_event(
            binding=binding,
            event_type=row.event_type,
            payload=row.payload if isinstance(row.payload, dict) else {},
            locale=locale,
        )
        self.db.add(
            NotificationDeliveryAttempt(
                outbox_id=row.id,
                provider=delivery.provider,
                provider_message_id=delivery.provider_message_id,
                request_payload=delivery.request_payload,
                response_payload=delivery.response_payload,
                success=delivery.success,
                error_code=delivery.error_code,
                error_message=delivery.error_message,
                attempted_at=now,
            )
        )
        if delivery.success:
            row.status = "sent"
            row.sent_at = now
            row.next_retry_at = None
            row.last_error = None
            return

        await self._mark_failed(
            row=row,
            now=now,
            error_code=delivery.error_code or "DELIVERY_FAILED",
            error_message=delivery.error_message or "IM provider returned failure.",
            already_recorded_attempt=True,
        )

    async def _mark_failed(
        self,
        *,
        row: NotificationOutbox,
        now: datetime,
        error_code: str,
        error_message: str,
        already_recorded_attempt: bool = False,
    ) -> None:
        if not already_recorded_attempt:
            self.db.add(
                NotificationDeliveryAttempt(
                    outbox_id=row.id,
                    provider=row.channel,
                    provider_message_id=None,
                    request_payload={},
                    response_payload={},
                    success=False,
                    error_code=error_code,
                    error_message=error_message[:500],
                    attempted_at=now,
                )
            )

        row.retry_count = int(row.retry_count) + 1
        row.last_error = f"{error_code}: {error_message[:450]}"
        if row.retry_count > int(row.max_retries):
            row.status = "dead"
            row.next_retry_at = None
            logger.warning(
                "notification moved to dead-letter status outbox_id=%s channel=%s event_type=%s error=%s",
                row.id,
                row.channel,
                row.event_type,
                error_code,
            )
            return

        backoff_seconds = float(settings.notifications_retry_backoff_seconds) * (2 ** (row.retry_count - 1))
        row.status = "failed"
        row.next_retry_at = now + timedelta(seconds=backoff_seconds)

    async def _resolve_default_channels(self, *, user_id: UUID) -> list[str]:
        row = await self.preference_service.get_or_create(user_id=user_id)
        channels: list[str] = []
        if settings.telegram_enabled and bool(row.telegram_enabled):
            channels.append("telegram")
        return channels


def _build_default_provider_registry() -> IMProviderRegistry:
    registry = IMProviderRegistry()
    if settings.telegram_enabled:
        registry.register(TelegramNotificationProvider())
    return registry
