from __future__ import annotations

import asyncio
from urllib.parse import quote_plus
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from packages.infra.db.models.user import User
from packages.infra.providers.clerk.client import ClerkClient


def test_000_auth_me_accepts_real_clerk_session_token(
    api_test_client: TestClient,
) -> None:
    email = f"pytest-clerk-live-{uuid4().hex[:12]}@example.com"
    clerk_user_id, session_token = asyncio.run(_create_live_clerk_session(email))

    try:
        response = api_test_client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {session_token}"},
        )
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["email"] == email
        assert payload["user_id"]

        persisted = asyncio.run(_find_local_user_by_email(email))
        assert persisted is not None
        assert persisted.clerk_user_id == clerk_user_id
        assert persisted.password_hash is None
    finally:
        asyncio.run(_delete_local_user(email))
        asyncio.run(_delete_live_clerk_user(clerk_user_id))


async def _create_live_clerk_session(email: str) -> tuple[str, str]:
    client = ClerkClient()
    user = await client.create_user(
        {
            "email_address": [email],
            "password": "PytestClerkLive#123",
            "first_name": "Pytest",
            "last_name": "Clerk",
            "external_id": f"pytest-clerk-live-{uuid4().hex[:8]}",
        }
    )
    session = await client.create_session(str(user["id"]))
    token = await client.create_session_token(
        str(session["id"]),
        authorized_party="http://localhost:3000",
    )
    return str(user["id"]), str(token["jwt"])


async def _find_local_user_by_email(email: str) -> User | None:
    engine = create_async_engine(_database_url(), future=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with session_factory() as session:
            return await session.scalar(select(User).where(User.email == email))
    finally:
        await engine.dispose()


async def _delete_local_user(email: str) -> None:
    engine = create_async_engine(_database_url(), future=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with session_factory() as session:
            user = await session.scalar(select(User).where(User.email == email))
            if user is None:
                return
            await session.delete(user)
            await session.commit()
    finally:
        await engine.dispose()


async def _delete_live_clerk_user(clerk_user_id: str) -> None:
    if not clerk_user_id.strip():
        return
    client = ClerkClient()
    try:
        await client.delete_user(clerk_user_id)
    except Exception:
        return


def _database_url() -> str:
    from packages.shared_settings.schema.settings import settings

    password = quote_plus(settings.postgres_password)
    return (
        "postgresql+asyncpg://"
        f"{settings.postgres_user}:{password}"
        f"@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"
    )
