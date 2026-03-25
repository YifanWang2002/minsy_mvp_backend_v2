"""Authentication service for register/login/token workflows."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import bcrypt
import jwt
from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from packages.domain.user.services.clerk_token_verifier import ClerkTokenVerifier
from packages.infra.db.models.user import User, UserProfile
from packages.infra.providers.clerk.client import ClerkClient
from packages.shared_settings.schema.settings import settings


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
        self._clerk_token_verifier = ClerkTokenVerifier()
        self._clerk_client = ClerkClient()

    async def register(self, email: str, password: str, name: str) -> tuple[User, TokenPair]:
        """Register new user and default profile, then sign tokens."""
        existing = await self.db.scalar(select(User).where(User.email == email))
        if existing is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Email already registered.",
            )

        password_hash = self.hash_password(password)
        user = User(
            email=email,
            password_hash=password_hash,
            name=name,
            auth_provider="legacy_password",
            is_active=True,
        )
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
        if user.password_hash is None or not user.password_hash.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password authentication is not available for this account.",
            )
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
        auth_mode = settings.auth_mode.strip().lower()
        if auth_mode == "clerk":
            return await self._get_current_user_from_clerk_token(token)
        if auth_mode == "hybrid":
            return await self._get_current_user_hybrid(token)

        return await self._get_current_user_from_legacy_token(token)

    async def _get_current_user_hybrid(self, token: str) -> User:
        algorithm = self._token_algorithm(token)
        if algorithm == "RS256":
            return await self._get_current_user_from_clerk_token(token)
        if algorithm == settings.jwt_algorithm.strip().upper():
            return await self._get_current_user_from_legacy_token(token)

        clerk_error: HTTPException | None = None
        try:
            return await self._get_current_user_from_clerk_token(token)
        except HTTPException as exc:
            clerk_error = exc

        try:
            return await self._get_current_user_from_legacy_token(token)
        except HTTPException:
            if clerk_error is not None:
                raise clerk_error
            raise

    async def _get_current_user_from_legacy_token(self, token: str) -> User:
        """Decode a legacy access token and fetch the current user."""
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

    async def _get_current_user_from_clerk_token(self, token: str) -> User:
        """Verify a Clerk session token and fetch or provision the local user."""
        verified = self._clerk_token_verifier.verify(token)
        return await self._get_or_provision_clerk_user(verified.clerk_user_id)

    async def _get_or_provision_clerk_user(self, clerk_user_id: str) -> User:
        existing = await self.db.scalar(
            select(User)
            .options(selectinload(User.profiles))
            .where(User.clerk_user_id == clerk_user_id)
        )
        if existing is not None:
            if not existing.is_active:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found or inactive.",
                )
            return existing

        remote_user = await self._clerk_client.get_user(clerk_user_id)
        if remote_user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive.",
            )

        email = self._extract_clerk_email(remote_user)
        if not email:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authenticated Clerk user is missing an email address.",
            )

        user = await self.db.scalar(
            select(User).options(selectinload(User.profiles)).where(User.email == email)
        )
        should_commit = False
        if user is None:
            user = User(
                email=email,
                password_hash=None,
                name=self._extract_clerk_name(remote_user, fallback_email=email),
                clerk_user_id=clerk_user_id,
                auth_provider=self._derive_auth_provider(remote_user),
                is_active=True,
            )
            self.db.add(user)
            await self.db.flush()
            self.db.add(
                UserProfile(
                    user_id=user.id,
                    kyc_status="incomplete",
                    trading_years_bucket=None,
                    risk_tolerance=None,
                    return_expectation=None,
                )
            )
            should_commit = True
        else:
            if user.clerk_user_id and user.clerk_user_id != clerk_user_id:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Email is already linked to a different Clerk account.",
                )
            if not user.is_active:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found or inactive.",
                )
            if not user.clerk_user_id:
                user.clerk_user_id = clerk_user_id
                should_commit = True
            derived_provider = self._derive_auth_provider(remote_user)
            if user.auth_provider == "legacy_password" and derived_provider != user.auth_provider:
                user.auth_provider = derived_provider
                should_commit = True
            if not user.name.strip():
                user.name = self._extract_clerk_name(remote_user, fallback_email=email)
                should_commit = True

        if should_commit:
            await self.db.commit()
            user = await self.db.scalar(
                select(User)
                .options(selectinload(User.profiles))
                .where(User.id == user.id)
            )

        if user is None:
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
    def verify_password(password: str, password_hash: str | None) -> bool:
        """Verify plain password against bcrypt hash."""
        if password_hash is None or not password_hash.strip():
            return False
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

    @staticmethod
    def _token_algorithm(token: str) -> str:
        try:
            header = jwt.get_unverified_header(token)
        except jwt.InvalidTokenError:
            return ""
        return str(header.get("alg") or "").strip().upper()

    @staticmethod
    def _extract_clerk_email(remote_user: dict[str, Any]) -> str:
        primary_email_id = str(remote_user.get("primary_email_address_id") or "").strip()
        email_addresses = remote_user.get("email_addresses")
        if not isinstance(email_addresses, list):
            return ""

        fallback = ""
        for item in email_addresses:
            if not isinstance(item, dict):
                continue
            email = str(item.get("email_address") or "").strip().lower()
            if not email:
                continue
            if not fallback:
                fallback = email
            if primary_email_id and str(item.get("id") or "").strip() == primary_email_id:
                return email
        return fallback

    @staticmethod
    def _extract_clerk_name(remote_user: dict[str, Any], *, fallback_email: str) -> str:
        first_name = str(remote_user.get("first_name") or "").strip()
        last_name = str(remote_user.get("last_name") or "").strip()
        full_name = " ".join(part for part in (first_name, last_name) if part).strip()
        if full_name:
            return full_name[:120]

        username = str(remote_user.get("username") or "").strip()
        if username:
            return username[:120]

        local_part = fallback_email.split("@", 1)[0].strip() or "User"
        return local_part[:120]

    @staticmethod
    def _derive_auth_provider(remote_user: dict[str, Any]) -> str:
        external_accounts = remote_user.get("external_accounts")
        if isinstance(external_accounts, list):
            for item in external_accounts:
                if not isinstance(item, dict):
                    continue
                provider = str(item.get("provider") or "").strip().lower()
                if provider == "google":
                    return "google_oauth"
                if provider == "apple":
                    return "apple_oauth"
        return "clerk"
