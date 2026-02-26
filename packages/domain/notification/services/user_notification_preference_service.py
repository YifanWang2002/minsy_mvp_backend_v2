"""Service for user notification preference read/write operations."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from packages.infra.db.models.user_notification_preference import UserNotificationPreference
from packages.core.events.notification_events import EVENT_TO_PREFERENCE_FIELD, MANDATORY_NOTIFICATION_EVENTS

_PREFERENCE_FIELDS: frozenset[str] = frozenset(
    {
        "telegram_enabled",
        "backtest_completed_enabled",
        "deployment_started_enabled",
        "position_opened_enabled",
        "position_closed_enabled",
        "risk_triggered_enabled",
        "execution_anomaly_enabled",
    }
)


@dataclass(frozen=True, slots=True)
class NotificationPreferenceView:
    user_id: UUID
    telegram_enabled: bool
    backtest_completed_enabled: bool
    deployment_started_enabled: bool
    position_opened_enabled: bool
    position_closed_enabled: bool
    risk_triggered_enabled: bool
    execution_anomaly_enabled: bool


class UserNotificationPreferenceService:
    """CRUD helpers for notification preferences."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def get_or_create(self, *, user_id: UUID) -> UserNotificationPreference:
        row = await self.db.scalar(
            select(UserNotificationPreference).where(UserNotificationPreference.user_id == user_id)
        )
        if row is not None:
            return row

        row = UserNotificationPreference(user_id=user_id)
        self.db.add(row)
        await self.db.flush()
        return row

    async def get_view(self, *, user_id: UUID) -> NotificationPreferenceView:
        row = await self.get_or_create(user_id=user_id)
        return NotificationPreferenceView(
            user_id=row.user_id,
            telegram_enabled=bool(row.telegram_enabled),
            backtest_completed_enabled=bool(row.backtest_completed_enabled),
            deployment_started_enabled=bool(row.deployment_started_enabled),
            position_opened_enabled=bool(row.position_opened_enabled),
            position_closed_enabled=bool(row.position_closed_enabled),
            risk_triggered_enabled=bool(row.risk_triggered_enabled),
            execution_anomaly_enabled=bool(row.execution_anomaly_enabled),
        )

    async def update(
        self,
        *,
        user_id: UUID,
        updates: dict[str, bool],
    ) -> NotificationPreferenceView:
        row = await self.get_or_create(user_id=user_id)
        for field, value in updates.items():
            if field not in _PREFERENCE_FIELDS:
                continue
            setattr(row, field, bool(value))
        await self.db.flush()
        return await self.get_view(user_id=user_id)

    async def is_event_enabled(
        self,
        *,
        user_id: UUID,
        event_type: str,
        channel: str,
    ) -> bool:
        if event_type in MANDATORY_NOTIFICATION_EVENTS:
            return True
        row = await self.get_or_create(user_id=user_id)
        if channel == "telegram" and not bool(row.telegram_enabled):
            return False
        field = EVENT_TO_PREFERENCE_FIELD.get(event_type)
        if field is None:
            return True
        return bool(getattr(row, field))
