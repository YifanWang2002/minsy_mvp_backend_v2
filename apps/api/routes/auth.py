"""Auth router endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.dependencies import get_db
from apps.api.middleware.auth import get_current_user
from apps.api.middleware.rate_limit import RateLimiter
from apps.api.schemas.auth_schemas import (
    AuthResponse,
    AuthUser,
    ChangePasswordRequest,
    DetailResponse,
    LoginRequest,
    OnboardingStatusResponse,
    RefreshRequest,
    RegisterRequest,
    TokenResponse,
    UpdateOnboardingStatusRequest,
    UpdateUserPreferencesRequest,
    UserPreferencesResponse,
    UserResponse,
)
from packages.domain.user.services.auth_service import AuthService
from packages.infra.db.models.user import User
from packages.infra.db.models.user_settings import (
    DEFAULT_FONT_SCALE,
    DEFAULT_LOCALE,
    DEFAULT_ONBOARDING_STATUS,
    DEFAULT_THEME_MODE,
    UserSetting,
)
from packages.shared_settings.schema.settings import settings

router = APIRouter(prefix="/auth", tags=["auth"])
me_rate_limiter = RateLimiter(
    limit=settings.auth_rate_limit, window=settings.auth_rate_window
)
change_password_rate_limiter = RateLimiter(
    limit=settings.auth_rate_limit,
    window=settings.auth_rate_window,
)
_ONBOARDING_SECTION_KEYS = ("home", "strategies", "portfolio")
_ONBOARDING_ALLOWED_STATUS = {"pending", "completed", "canceled"}


def _resolve_kyc_status(user: User) -> str:
    if user.profiles:
        return user.profiles[0].kyc_status
    return "incomplete"


def _resolve_user_tier(user: User) -> str:
    normalized = str(user.current_tier or "").strip().lower()
    if normalized in {"free", "go", "plus", "pro"}:
        return normalized
    return "free"


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


def _normalize_onboarding_status(raw: dict | None) -> dict[str, str]:
    normalized = {key: "pending" for key in _ONBOARDING_SECTION_KEYS}
    if not isinstance(raw, dict):
        return normalized
    for key in _ONBOARDING_SECTION_KEYS:
        value = str(raw.get(key, "")).strip().lower()
        if value in _ONBOARDING_ALLOWED_STATUS:
            normalized[key] = value
    return normalized


def _build_onboarding_response(setting: UserSetting | None) -> OnboardingStatusResponse:
    if setting is None:
        normalized = _normalize_onboarding_status(DEFAULT_ONBOARDING_STATUS)
        return OnboardingStatusResponse(
            home=normalized["home"],
            strategies=normalized["strategies"],
            portfolio=normalized["portfolio"],
            has_persisted=False,
            updated_at=None,
        )

    normalized = _normalize_onboarding_status(
        setting.onboarding_status
        if isinstance(setting.onboarding_status, dict)
        else {},
    )
    return OnboardingStatusResponse(
        home=normalized["home"],
        strategies=normalized["strategies"],
        portfolio=normalized["portfolio"],
        has_persisted=True,
        updated_at=setting.updated_at,
    )


@router.post(
    "/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED
)
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
        user=AuthUser(
            name=user.name, kyc_status="incomplete", tier=_resolve_user_tier(user)
        ),
    )


@router.post("/login", response_model=AuthResponse)
async def login(
    payload: LoginRequest,
    db: AsyncSession = Depends(get_db),
) -> AuthResponse:
    service = AuthService(db)
    user, token_pair = await service.login(
        email=payload.email, password=payload.password
    )
    return AuthResponse(
        user_id=user.id,
        access_token=token_pair.access_token,
        refresh_token=token_pair.refresh_token,
        expires_in=token_pair.expires_in,
        user=AuthUser(
            name=user.name,
            kyc_status=_resolve_kyc_status(user),
            tier=_resolve_user_tier(user),
        ),
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
        tier=_resolve_user_tier(user),
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


@router.get("/onboarding-status", response_model=OnboardingStatusResponse)
async def get_onboarding_status(
    _: None = Depends(me_rate_limiter),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> OnboardingStatusResponse:
    setting = await db.scalar(select(UserSetting).where(UserSetting.user_id == user.id))
    return _build_onboarding_response(setting)


@router.put("/onboarding-status", response_model=OnboardingStatusResponse)
async def put_onboarding_status(
    payload: UpdateOnboardingStatusRequest,
    _: None = Depends(me_rate_limiter),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> OnboardingStatusResponse:
    setting = await db.scalar(select(UserSetting).where(UserSetting.user_id == user.id))
    if setting is None:
        setting = UserSetting(user_id=user.id)
        db.add(setting)

    status_map = _normalize_onboarding_status(
        setting.onboarding_status
        if isinstance(setting.onboarding_status, dict)
        else {},
    )
    status_map[payload.section] = payload.status
    setting.onboarding_status = status_map

    await db.commit()
    await db.refresh(setting)
    return _build_onboarding_response(setting)


@router.post("/onboarding-status/reset", response_model=OnboardingStatusResponse)
async def reset_onboarding_status(
    _: None = Depends(me_rate_limiter),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> OnboardingStatusResponse:
    setting = await db.scalar(select(UserSetting).where(UserSetting.user_id == user.id))
    if setting is None:
        setting = UserSetting(user_id=user.id)
        db.add(setting)

    setting.onboarding_status = {key: "pending" for key in _ONBOARDING_SECTION_KEYS}
    await db.commit()
    await db.refresh(setting)
    return _build_onboarding_response(setting)


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
