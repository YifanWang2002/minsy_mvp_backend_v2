"""Authentication service for register/login/token workflows."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import bcrypt
import jwt
from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.config import settings
from src.models.user import User, UserProfile


@dataclass(slots=True)
class TokenPair:
    """Access/refresh token tuple."""

    access_token: str
    refresh_token: str
    expires_in: int


class AuthService:
    """Auth domain logic around users and JWT tokens."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def register(self, email: str, password: str, name: str) -> tuple[User, TokenPair]:
        """Register new user and default profile, then sign tokens."""
        existing = await self.db.scalar(select(User).where(User.email == email))
        if existing is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Email already registered.",
            )

        password_hash = self.hash_password(password)
        user = User(email=email, password_hash=password_hash, name=name, is_active=True)
        self.db.add(user)
        await self.db.flush()

        profile = UserProfile(
            user_id=user.id,
            kyc_status="incomplete",
            trading_years_bucket=None,
            risk_tolerance=None,
            return_expectation=None,
        )
        self.db.add(profile)
        await self.db.commit()

        return user, self.sign_tokens(user.id)

    async def login(self, email: str, password: str) -> tuple[User, TokenPair]:
        """Authenticate by email/password and return token pair."""
        stmt = select(User).options(selectinload(User.profiles)).where(User.email == email)
        user = await self.db.scalar(stmt)
        if user is None or not self.verify_password(password, user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password.",
            )
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User is inactive.",
            )

        return user, self.sign_tokens(user.id)

    async def refresh(self, refresh_token: str) -> TokenPair:
        """Validate refresh token and issue a new token pair."""
        payload = self.decode_token(refresh_token)
        if payload.get("type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token type.",
            )

        user_id = self._extract_user_id(payload)
        user = await self.db.scalar(select(User).where(User.id == user_id))
        if user is None or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive.",
            )
        return self.sign_tokens(user_id)

    async def change_password(
        self,
        user: User,
        current_password: str,
        new_password: str,
    ) -> None:
        """Change user password after verifying the current password."""
        if not self.verify_password(current_password, user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Current password is incorrect.",
            )
        if current_password == new_password:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="New password must be different from current password.",
            )

        user.password_hash = self.hash_password(new_password)
        await self.db.commit()

    def sign_tokens(self, user_id: UUID) -> TokenPair:
        """Sign access token (+24h) and refresh token (+7d)."""
        now = datetime.now(UTC)
        access_exp = now + timedelta(minutes=settings.access_token_expire_minutes)
        refresh_exp = now + timedelta(days=settings.refresh_token_expire_days)

        access_payload = {
            "sub": str(user_id),
            "type": "access",
            "iat": int(now.timestamp()),
            "exp": int(access_exp.timestamp()),
            "jti": str(uuid4()),
        }
        refresh_payload = {
            "sub": str(user_id),
            "type": "refresh",
            "iat": int(now.timestamp()),
            "exp": int(refresh_exp.timestamp()),
            "jti": str(uuid4()),
        }

        access_token = jwt.encode(
            access_payload,
            settings.secret_key,
            algorithm=settings.jwt_algorithm,
        )
        refresh_token = jwt.encode(
            refresh_payload,
            settings.secret_key,
            algorithm=settings.jwt_algorithm,
        )

        return TokenPair(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=settings.access_token_expire_minutes * 60,
        )

    async def get_current_user(self, token: str) -> User:
        """Decode access token and fetch current user."""
        payload = self.decode_token(token)
        if payload.get("type") != "access":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid access token type.",
            )

        user_id = self._extract_user_id(payload)
        stmt = select(User).options(selectinload(User.profiles)).where(User.id == user_id)
        user = await self.db.scalar(stmt)
        if user is None or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive.",
            )
        return user

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash plain password with bcrypt."""
        return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

    @staticmethod
    def verify_password(password: str, password_hash: str) -> bool:
        """Verify plain password against bcrypt hash."""
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))

    @staticmethod
    def _extract_user_id(payload: dict) -> UUID:
        sub = payload.get("sub")
        if not isinstance(sub, str):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token subject.",
            )
        try:
            return UUID(sub)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token subject.",
            ) from exc

    @staticmethod
    def decode_token(token: str) -> dict:
        """Decode JWT and normalize auth-related errors."""
        try:
            payload = jwt.decode(
                token,
                settings.secret_key,
                algorithms=[settings.jwt_algorithm],
            )
            if not isinstance(payload, dict):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token payload.",
                )
            return payload
        except jwt.ExpiredSignatureError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired.",
            ) from exc
        except jwt.InvalidTokenError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token.",
            ) from exc
