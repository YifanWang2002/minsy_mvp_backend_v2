"""Auth API request/response schemas."""

from __future__ import annotations

from datetime import datetime
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
    created_at: datetime


class DetailResponse(BaseModel):
    detail: str
