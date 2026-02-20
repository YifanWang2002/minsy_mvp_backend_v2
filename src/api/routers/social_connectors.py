"""Social connector endpoints for settings-page integrations."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.middleware.auth import get_current_user
from src.api.schemas.auth_schemas import DetailResponse
from src.api.schemas.events import (
    SocialConnectorItem,
    TelegramActivitiesResponse,
    TelegramActivityItem,
    TelegramConnectLinkResponse,
)
from src.api.schemas.requests import TelegramConnectLinkRequest
from src.config import settings
from src.dependencies import get_db
from src.models.user import User
from src.services.social_connector_service import SocialConnectorService
from src.services.telegram_service import TelegramService
from src.util.logger import logger

router = APIRouter(prefix="/social", tags=["social_connectors"])


@router.get("/connectors", response_model=list[SocialConnectorItem])
async def list_connectors(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[SocialConnectorItem]:
    service = SocialConnectorService(db)
    items = await service.list_connectors(user_id=user.id)
    return [
        SocialConnectorItem(
            provider=item.provider,
            status=item.status,
            connected_account=item.connected_account,
            connected_at=item.connected_at,
            supports_connect=item.supports_connect,
        )
        for item in items
    ]


@router.post(
    "/connectors/telegram/connect-link",
    response_model=TelegramConnectLinkResponse,
)
async def create_telegram_connect_link(
    payload: TelegramConnectLinkRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> TelegramConnectLinkResponse:
    service = SocialConnectorService(db)
    try:
        result = await service.create_telegram_connect_link(
            user_id=user.id,
            locale=payload.locale,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    await db.commit()
    return TelegramConnectLinkResponse(
        provider=result.provider,
        connect_url=result.connect_url,
        expires_at=result.expires_at,
    )


@router.get(
    "/connectors/telegram/activities",
    response_model=TelegramActivitiesResponse,
)
async def list_telegram_activities(
    limit: int = Query(default=20, ge=1, le=100),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> TelegramActivitiesResponse:
    service = SocialConnectorService(db)
    rows = await service.list_telegram_activities(user_id=user.id, limit=limit)
    return TelegramActivitiesResponse(
        provider="telegram",
        items=[
            TelegramActivityItem(
                id=row.id,
                event_type=row.event_type,
                choice_value=row.choice_value,
                message_text=row.message_text,
                created_at=row.created_at,
            )
            for row in rows
        ],
    )


@router.post(
    "/connectors/telegram/disconnect",
    response_model=DetailResponse,
)
async def disconnect_telegram(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> DetailResponse:
    service = SocialConnectorService(db)
    updated = await service.disconnect_telegram(user_id=user.id)
    await db.commit()
    if updated:
        return DetailResponse(detail="Telegram disconnected.")
    return DetailResponse(detail="Telegram binding not found.")


@router.post("/webhooks/telegram")
async def telegram_webhook(
    request: Request,
    db: AsyncSession = Depends(get_db),
    x_telegram_bot_api_secret_token: Annotated[
        str | None,
        Header(alias="X-Telegram-Bot-Api-Secret-Token"),
    ] = None,
) -> dict[str, bool]:
    _validate_webhook_secret(x_telegram_bot_api_secret_token)
    try:
        payload: Any = await request.json()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid Telegram webhook payload: {exc}",
        ) from exc
    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid Telegram webhook payload.",
        )

    service = TelegramService(db)
    try:
        await service.handle_webhook_update(payload)
        await db.commit()
    except Exception:  # noqa: BLE001
        await db.rollback()
        logger.exception("Telegram webhook processing failed.")
        return {"ok": True}
    return {"ok": True}


def _validate_webhook_secret(raw_header: str | None) -> None:
    expected = settings.telegram_webhook_secret_token.strip()
    if not expected:
        return
    if raw_header is None or raw_header.strip() != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Telegram webhook secret token.",
        )
