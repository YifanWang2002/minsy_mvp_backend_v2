"""Social connector endpoints for settings-page integrations."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.middleware.auth import get_current_user
from apps.api.schemas.auth_schemas import DetailResponse
from apps.api.schemas.events import (
    SocialConnectorItem,
    TelegramActivitiesResponse,
    TelegramActivityItem,
    TelegramConnectLinkResponse,
    TelegramTestSendResponse,
    TelegramTestTargetResponse,
)
from apps.api.schemas.requests import TelegramConnectLinkRequest, TelegramTestSendRequest
from packages.shared_settings.schema.settings import settings
from apps.api.dependencies import get_db
from packages.infra.db.models.user import User
from packages.domain.user.services.social_connector_service import SocialConnectorService
from apps.api.services.telegram_service import TelegramService
from packages.domain.user.services.telegram_test_batches import build_telegram_test_chart_html
from packages.infra.observability.logger import logger

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


@router.get(
    "/connectors/telegram/test-webapp/chart",
    response_class=HTMLResponse,
)
async def telegram_test_webapp_chart(
    symbol: str = Query(default="NASDAQ:AAPL"),
    interval: str = Query(default="1d"),
    locale: str = Query(default="en"),
    theme: str = Query(default="light"),
    signal_id: str = Query(default="unknown"),
) -> HTMLResponse:
    html = build_telegram_test_chart_html(
        symbol=symbol,
        interval=interval,
        locale=locale,
        theme=theme,
        signal_id=signal_id,
    )
    return HTMLResponse(content=html)


@router.get(
    "/connectors/telegram/test-target",
    response_model=TelegramTestTargetResponse,
)
async def telegram_test_target_status(
    user: User = Depends(get_current_user),  # noqa: ARG001
    db: AsyncSession = Depends(get_db),
) -> TelegramTestTargetResponse:
    configured_email = settings.telegram_test_target_email.strip().lower()
    service = SocialConnectorService(db)

    resolved_user_id = None
    if configured_email:
        resolved_user_id = await db.scalar(
            select(User.id).where(func.lower(User.email) == configured_email)
        )
    binding_any = (
        await service.resolve_connected_telegram_binding_by_email(
            email=configured_email,
            require_connected=False,
        )
        if configured_email
        else None
    )
    binding_connected = (
        await service.resolve_connected_telegram_binding_by_email(
            email=configured_email,
            require_connected=True,
        )
        if configured_email
        else None
    )
    binding = binding_connected or binding_any
    await db.commit()
    return TelegramTestTargetResponse(
        configured_email=configured_email,
        resolved_user_exists=resolved_user_id is not None,
        resolved_binding_connected=binding_connected is not None,
        resolved_chat_id_masked=_mask_chat_id(binding.external_chat_id if binding is not None else None),
        resolved_binding_id=binding.id if binding is not None else None,
        resolved_username=binding.external_username if binding is not None else None,
        resolved_user_id=binding.user_id if binding is not None else resolved_user_id,
    )


@router.post(
    "/connectors/telegram/test-send",
    response_model=TelegramTestSendResponse,
)
async def telegram_test_send(
    payload: TelegramTestSendRequest,
    user: User = Depends(get_current_user),  # noqa: ARG001
    db: AsyncSession = Depends(get_db),
) -> TelegramTestSendResponse:
    service = TelegramService(db)
    try:
        result = await service.send_test_message(text=payload.message)
        await db.commit()
    except ValueError as exc:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # noqa: BLE001
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Telegram test-send failed: {type(exc).__name__}",
        ) from exc

    target_email = str(result.get("configured_email", "")).strip().lower()
    target_chat_masked = _mask_chat_id(result.get("target_chat_id"))
    return TelegramTestSendResponse(
        ok=True,
        actual_target=f"{target_email}/{target_chat_masked}",
        message_id=result.get("message_id"),
        detail="Message sent.",
    )


def _mask_chat_id(raw: object) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if len(text) <= 4:
        return "*" * len(text)
    return f"{text[:2]}***{text[-2:]}"
