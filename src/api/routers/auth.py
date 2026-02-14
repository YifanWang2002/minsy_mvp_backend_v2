"""Auth router endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.middleware.auth import get_current_user
from src.api.middleware.rate_limit import RateLimiter
from src.api.schemas.auth_schemas import (
    AuthResponse,
    AuthUser,
    LoginRequest,
    RefreshRequest,
    RegisterRequest,
    TokenResponse,
    UserResponse,
)
from src.config import settings
from src.dependencies import get_db
from src.models.user import User
from src.services.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["auth"])
me_rate_limiter = RateLimiter(limit=settings.auth_rate_limit, window=settings.auth_rate_window)


def _resolve_kyc_status(user: User) -> str:
    if user.profiles:
        return user.profiles[0].kyc_status
    return "incomplete"


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
