"""Auth API request/response schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


class RegisterRequest(BaseModel):
    email: str
    password: str = Field(min_length=6, max_length=128)
    name: str = Field(min_length=1, max_length=120)


class LoginRequest(BaseModel):
    email: str
    password: str = Field(min_length=6, max_length=128)


class RefreshRequest(BaseModel):
    refresh_token: str


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(min_length=6, max_length=128)
    new_password: str = Field(min_length=6, max_length=128)


class AuthUser(BaseModel):
    name: str
    kyc_status: str
    tier: Literal["free", "go", "plus", "pro"] = "free"


class AuthResponse(BaseModel):
    user_id: UUID
    access_token: str
    refresh_token: str
    expires_in: int
    user: AuthUser


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    expires_in: int


class UserResponse(BaseModel):
    user_id: UUID
    email: str
    name: str
    kyc_status: str
    tier: Literal["free", "go", "plus", "pro"] = "free"
    created_at: datetime


ThemeModeValue = Literal["light", "dark", "system"]
LocaleValue = Literal["en", "zh"]
FontScaleValue = Literal["small", "default", "large"]
OnboardingSectionValue = Literal["home", "strategies", "portfolio"]
OnboardingStatusValue = Literal["pending", "completed", "canceled"]


class UserPreferencesResponse(BaseModel):
    theme_mode: ThemeModeValue
    locale: LocaleValue
    font_scale: FontScaleValue
    has_persisted: bool
    updated_at: datetime | None = None


class UpdateUserPreferencesRequest(BaseModel):
    theme_mode: ThemeModeValue
    locale: LocaleValue
    font_scale: FontScaleValue


class OnboardingStatusResponse(BaseModel):
    home: OnboardingStatusValue
    strategies: OnboardingStatusValue
    portfolio: OnboardingStatusValue
    has_persisted: bool
    updated_at: datetime | None = None


class UpdateOnboardingStatusRequest(BaseModel):
    section: OnboardingSectionValue
    status: OnboardingStatusValue


class DetailResponse(BaseModel):
    detail: str
