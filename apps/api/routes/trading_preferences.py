"""Trading execution preference endpoints."""

from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.middleware.auth import get_current_user
from apps.api.schemas.events import TradingPreferenceResponse
from apps.api.schemas.requests import TradingPreferencesUpdateRequest
from apps.api.dependencies import get_db
from packages.infra.db.models.user import User
from packages.domain.trading.services.trading_preference_service import TradingPreferenceService

router = APIRouter(prefix="/trading", tags=["trading"])


@router.get("/preferences", response_model=TradingPreferenceResponse)
async def get_trading_preferences(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> TradingPreferenceResponse:
    service = TradingPreferenceService(db)
    view = await service.get_view(user_id=user.id)
    await db.commit()
    return TradingPreferenceResponse(**asdict(view))


@router.put("/preferences", response_model=TradingPreferenceResponse)
async def update_trading_preferences(
    payload: TradingPreferencesUpdateRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> TradingPreferenceResponse:
    updates = payload.model_dump(exclude_none=True)
    service = TradingPreferenceService(db)
    try:
        view = await service.update(user_id=user.id, updates=updates)
    except ValueError as exc:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "TRADING_PREFERENCE_INVALID",
                "message": str(exc),
            },
        ) from exc
    await db.commit()
    return TradingPreferenceResponse(**asdict(view))
