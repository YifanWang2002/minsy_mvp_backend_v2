"""Authentication dependency for protected endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.dependencies import get_db
from packages.infra.db.models.user import User
from packages.domain.user.services.auth_service import AuthService


async def get_current_user(
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
    db: AsyncSession = Depends(get_db),
) -> User:
    """Parse Bearer token, validate JWT and return current user ORM object."""
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1].strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization header format.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = parts[1].strip()
    service = AuthService(db)
    return await service.get_current_user(token)
