"""Auth router endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.middleware.auth import get_current_user
from apps.api.middleware.rate_limit import RateLimiter
from apps.api.schemas.auth_schemas import (
    AuthResponse,
    AuthUser,
    ChangePasswordRequest,
    DetailResponse,
    LoginRequest,
    RefreshRequest,
    RegisterRequest,
    TokenResponse,
    UpdateUserPreferencesRequest,
    UserPreferencesResponse,
    UserResponse,
)
from packages.shared_settings.schema.settings import settings
from apps.api.dependencies import get_db
from packages.infra.db.models.user import User
from packages.infra.db.models.user_settings import (
    DEFAULT_FONT_SCALE,
    DEFAULT_LOCALE,
    DEFAULT_THEME_MODE,
    UserSetting,
)
from packages.domain.user.services.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["auth"])
me_rate_limiter = RateLimiter(limit=settings.auth_rate_limit, window=settings.auth_rate_window)
change_password_rate_limiter = RateLimiter(
    limit=settings.auth_rate_limit,
    window=settings.auth_rate_window,
)


def _resolve_kyc_status(user: User) -> str:
    if user.profiles:
        return user.profiles[0].kyc_status
    return "incomplete"


def _build_preferences_response(setting: UserSetting | None) -> UserPreferencesResponse:
    if setting is None:
        return UserPreferencesResponse(
            theme_mode=DEFAULT_THEME_MODE,
            locale=DEFAULT_LOCALE,
            font_scale=DEFAULT_FONT_SCALE,
            has_persisted=False,
            updated_at=None,
        )
    return UserPreferencesResponse(
        theme_mode=setting.theme_mode,
        locale=setting.locale,
        font_scale=setting.font_scale,
        has_persisted=True,
        updated_at=setting.updated_at,
    )


@router.post("/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
async def register(
    payload: RegisterRequest,
    db: AsyncSession = Depends(get_db),
) -> AuthResponse:
    service = AuthService(db)
    user, token_pair = await service.register(
        email=payload.email,
        password=payload.password,
        name=payload.name,
    )
    return AuthResponse(
        user_id=user.id,
        access_token=token_pair.access_token,
        refresh_token=token_pair.refresh_token,
        expires_in=token_pair.expires_in,
        user=AuthUser(name=user.name, kyc_status="incomplete"),
    )


@router.post("/login", response_model=AuthResponse)
async def login(
    payload: LoginRequest,
    db: AsyncSession = Depends(get_db),
) -> AuthResponse:
    service = AuthService(db)
    user, token_pair = await service.login(email=payload.email, password=payload.password)
    return AuthResponse(
        user_id=user.id,
        access_token=token_pair.access_token,
        refresh_token=token_pair.refresh_token,
        expires_in=token_pair.expires_in,
        user=AuthUser(name=user.name, kyc_status=_resolve_kyc_status(user)),
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh(
    payload: RefreshRequest,
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    service = AuthService(db)
    token_pair = await service.refresh(payload.refresh_token)
    return TokenResponse(
        access_token=token_pair.access_token,
        refresh_token=token_pair.refresh_token,
        expires_in=token_pair.expires_in,
    )


@router.get("/me", response_model=UserResponse)
async def me(
    _: None = Depends(me_rate_limiter),
    user: User = Depends(get_current_user),
) -> UserResponse:
    return UserResponse(
        user_id=user.id,
        email=user.email,
        name=user.name,
        kyc_status=_resolve_kyc_status(user),
        created_at=user.created_at,
    )


@router.get("/preferences", response_model=UserPreferencesResponse)
async def get_preferences(
    _: None = Depends(me_rate_limiter),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> UserPreferencesResponse:
    setting = await db.scalar(select(UserSetting).where(UserSetting.user_id == user.id))
    return _build_preferences_response(setting)


@router.put("/preferences", response_model=UserPreferencesResponse)
async def put_preferences(
    payload: UpdateUserPreferencesRequest,
    _: None = Depends(me_rate_limiter),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> UserPreferencesResponse:
    setting = await db.scalar(select(UserSetting).where(UserSetting.user_id == user.id))
    if setting is None:
        setting = UserSetting(user_id=user.id)
        db.add(setting)

    setting.theme_mode = payload.theme_mode
    setting.locale = payload.locale
    setting.font_scale = payload.font_scale

    await db.commit()
    await db.refresh(setting)
    return _build_preferences_response(setting)


@router.post("/change-password", response_model=DetailResponse)
async def change_password(
    payload: ChangePasswordRequest,
    _: None = Depends(change_password_rate_limiter),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> DetailResponse:
    service = AuthService(db)
    await service.change_password(
        user=user,
        current_password=payload.current_password,
        new_password=payload.new_password,
    )
    return DetailResponse(detail="Password updated successfully.")
