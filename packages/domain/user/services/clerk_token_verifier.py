"""Manual Clerk session token verification utilities."""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Any

import jwt
from fastapi import HTTPException, status

from packages.shared_settings.schema.settings import settings


@dataclass(slots=True)
class VerifiedClerkSession:
    """Normalized Clerk session claims used by the auth service."""

    clerk_user_id: str
    session_id: str | None
    authorized_party: str | None
    claims: dict[str, Any]


class ClerkTokenVerifier:
    """Verify Clerk-issued session tokens with the configured JWT public key."""

    def __init__(
        self,
        *,
        jwt_key: str | None = None,
        authorized_parties: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        self._jwt_key = (jwt_key or settings.clerk_jwt_key).strip()
        raw_parties = (
            authorized_parties
            if authorized_parties is not None
            else settings.clerk_authorized_parties
        )
        self._authorized_parties = tuple(
            str(value).strip() for value in raw_parties if str(value).strip()
        )

    @property
    def is_configured(self) -> bool:
        return bool(self._jwt_key)

    @staticmethod
    def token_algorithm(token: str) -> str:
        """Best-effort read of the JWT algorithm without validating the token."""
        try:
            header = jwt.get_unverified_header(token)
        except jwt.InvalidTokenError:
            return ""
        return str(header.get("alg") or "").strip().upper()

    def verify(self, token: str) -> VerifiedClerkSession:
        """Verify a Clerk session JWT and return normalized claims."""
        normalized = token.strip()
        if not normalized:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token.",
            )
        if not self.is_configured:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Clerk token verification is not configured.",
            )

        try:
            payload = jwt.decode(
                normalized,
                self._jwt_key,
                algorithms=["RS256"],
                options={"require": ["sub", "exp", "iat"]},
            )
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

        if not isinstance(payload, dict):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload.",
            )

        clerk_user_id = str(payload.get("sub") or "").strip()
        if not clerk_user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token subject.",
            )

        authorized_party = str(payload.get("azp") or "").strip() or None
        if self._authorized_parties and authorized_party is not None:
            if not self._is_allowed_authorized_party(authorized_party):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authorized party.",
                )

        session_id = str(payload.get("sid") or "").strip() or None
        return VerifiedClerkSession(
            clerk_user_id=clerk_user_id,
            session_id=session_id,
            authorized_party=authorized_party,
            claims=payload,
        )

    def _is_allowed_authorized_party(self, authorized_party: str) -> bool:
        for allowed in self._authorized_parties:
            if not allowed:
                continue
            if "*" in allowed:
                if fnmatch(authorized_party, allowed):
                    return True
                continue
            if authorized_party == allowed:
                return True
        return False
