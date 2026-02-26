"""Notification preference management endpoints."""

from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.middleware.auth import get_current_user
from apps.api.schemas.events import NotificationPreferencesResponse
from apps.api.schemas.requests import NotificationPreferencesUpdateRequest
from apps.api.dependencies import get_db
from packages.infra.db.models.user import User
from packages.domain.notification.services.user_notification_preference_service import (
    UserNotificationPreferenceService,
)

router = APIRouter(prefix="/notifications", tags=["notifications"])


@router.get("/preferences", response_model=NotificationPreferencesResponse)
async def get_notification_preferences(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> NotificationPreferencesResponse:
    service = UserNotificationPreferenceService(db)
    view = await service.get_view(user_id=user.id)
    await db.commit()
    return NotificationPreferencesResponse(**asdict(view))


@router.put("/preferences", response_model=NotificationPreferencesResponse)
async def update_notification_preferences(
    payload: NotificationPreferencesUpdateRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> NotificationPreferencesResponse:
    updates = payload.model_dump(exclude_none=True)
    service = UserNotificationPreferenceService(db)
    view = await service.update(user_id=user.id, updates=updates)
    await db.commit()
    return NotificationPreferencesResponse(**asdict(view))
