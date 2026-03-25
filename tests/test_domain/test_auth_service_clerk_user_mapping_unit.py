from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import jwt
import pytest
from sqlalchemy import Select, select

from packages.domain.user.services.auth_service import AuthService
from packages.infra.db.models.user import User, UserProfile
from packages.infra.providers.clerk.client import ClerkClient
from packages.shared_settings.schema.settings import settings

_TEST_PRIVATE_KEY = """-----BEGIN PRIVATE KEY-----
MIICeQIBADANBgkqhkiG9w0BAQEFAASCAmMwggJfAgEAAoGBANZduwEN0ylXkggr
8mjL/3DLT/fpXxLq/sTpTtv0z7mP8nCGYq8zMJLCZQW6vOlgAuNVZYlXvXHzY0uQ
6Whmmt4z2CO2dF/L9qzK7ZFYgKF+NboG/TYSO2EsjuPuTSNm38KLsAeQKqtyLQtN
XN3LnkL6aE+QQ+3WwWRVSMWgx+3ZAgMBAAECgYEAj+M+YMi80mU7WkzVW86CWV2/
AbMd4/7kn5vTGQVMYUvj+e/aUatUkU32rU/Y+fU+OwXZL8U7Hj+2iMRuR2uHyx9f
7lqQYjXq0qqPELvF3jzAh7aVHZuzAjZnxqCAwU0gxSBjyVydLFWJikFQqVvCO7Jo
gi7PdWB119DusZWCx2ECQQD6sWKuUaUXAOMN9P0FTNW8BAAvXVKwVnqkjiFb07L6
aY68ha1HgVRaKLwNywmFzjASBQTutOXwQNp0JNTYZgSlAkEA2ud5bPu+6x8+J7bD
0aUf+Y55eNrcqXV1H8mP1grbqos1lVBy1v3nWu4jchzOgkqczZIEyhOBRb+nw5AN
Y+aaJQJBAN8SMMkEhW5ur5ufv/WTZSykMrXyyL14djEu96gKPFxuyUAfgwz5m+GO
FagAXzzdOBEQvk7aUTDzxG9Mxsi4HrECQQDJT8KddU74j7zrbOrcq8yiBmKzwCLa
PMi/uO/sWgP17RwT+u4BxXK0bvhuAwvvSoq1iqmY5SMnb7/q21lVHEd5AkEAiSnk
IxaMzu6RfCNZWokIQMjgFiQNyrajHZKW8hZp+STdJnZxadi0KgNdLUI0KOiWzLbv
gQmU5UikcqXDyQR0Pw==
-----END PRIVATE KEY-----"""

_TEST_PUBLIC_KEY = """-----BEGIN PUBLIC KEY-----
MIGfMA0GCSqGSIb3DQEBAQUAA4GNADCBiQKBgQDWXbsBDdMpV5IIK/Joy/9wy0/3
6V8S6v7E6U7b9M+5j/JwhmKvMzCSwmUFurzpYALjVWWJV71x82NLkOloZpreM9gj
tnRfy/asyu2RWIChfjW6Bv02EjthLI7j7k0jZt/Ci7AHkCqrci0LTVzdy55C+mhP
kEPt1sFkVUjFoMft2QIDAQAB
-----END PUBLIC KEY-----"""


def _issue_clerk_token(sub: str) -> str:
    now = datetime.now(UTC)
    return jwt.encode(
        {
            "sub": sub,
            "sid": "sess_123",
            "azp": "https://dev.minsyai.com",
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(minutes=10)).timestamp()),
        },
        _TEST_PRIVATE_KEY,
        algorithm="RS256",
    )


@pytest.fixture
def db_session() -> "_FakeAsyncSession":
    return _FakeAsyncSession()


@pytest.fixture(autouse=True)
def clerk_auth_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "auth_mode", "hybrid")
    monkeypatch.setattr(settings, "clerk_jwt_key", _TEST_PUBLIC_KEY)
    monkeypatch.setattr(
        settings,
        "clerk_authorized_parties",
        ["https://dev.minsyai.com"],
    )


@pytest.mark.asyncio
async def test_auth_service_accepts_existing_clerk_user_mapping(
    db_session: "_FakeAsyncSession",
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user = User(
        email="mapped@example.com",
        password_hash=None,
        name="Mapped User",
        clerk_user_id="user_clerk_mapped",
        auth_provider="clerk",
        is_active=True,
    )
    db_session.add(user)
    await db_session.flush()
    db_session.add(UserProfile(user_id=user.id, kyc_status="complete"))
    await db_session.commit()

    remote_lookup = AsyncMock(side_effect=AssertionError("remote Clerk lookup should not run"))
    monkeypatch.setattr(ClerkClient, "get_user", remote_lookup)

    service = AuthService(db_session)
    resolved = await service.get_current_user(_issue_clerk_token("user_clerk_mapped"))

    assert resolved.id == user.id
    assert resolved.email == "mapped@example.com"
    remote_lookup.assert_not_called()


@pytest.mark.asyncio
async def test_auth_service_provisions_new_local_user_from_clerk(
    db_session: "_FakeAsyncSession",
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote_lookup = AsyncMock(
        return_value={
            "id": "user_clerk_new",
            "first_name": "Fresh",
            "last_name": "Trader",
            "primary_email_address_id": "idn_123",
            "email_addresses": [
                {"id": "idn_123", "email_address": "fresh@example.com"},
            ],
            "external_accounts": [{"provider": "google"}],
        }
    )
    monkeypatch.setattr(ClerkClient, "get_user", remote_lookup)

    service = AuthService(db_session)
    resolved = await service.get_current_user(_issue_clerk_token("user_clerk_new"))

    assert resolved.email == "fresh@example.com"
    assert resolved.name == "Fresh Trader"
    assert resolved.clerk_user_id == "user_clerk_new"
    assert resolved.auth_provider == "google_oauth"

    persisted = await db_session.scalar(
        select(User).where(User.email == "fresh@example.com")
    )
    assert persisted is not None
    assert persisted.clerk_user_id == "user_clerk_new"


class _FakeAsyncSession:
    def __init__(self) -> None:
        self.users: list[User] = []
        self.profiles: list[UserProfile] = []

    def add(self, obj: Any) -> None:
        if isinstance(obj, User):
            if getattr(obj, "id", None) is None:
                obj.id = uuid4()
            if obj.profiles is None:
                obj.profiles = []
            self.users.append(obj)
            return

        if isinstance(obj, UserProfile):
            if getattr(obj, "id", None) is None:
                obj.id = uuid4()
            self.profiles.append(obj)
            for user in self.users:
                if user.id == obj.user_id:
                    user.profiles.append(obj)
                    break

    async def flush(self) -> None:
        return None

    async def commit(self) -> None:
        return None

    async def scalar(self, statement: Select[Any]) -> Any | None:
        entity = statement.column_descriptions[0].get("entity")
        if entity is not User:
            return None

        for user in self.users:
            if self._matches(user, statement):
                return user
        return None

    def _matches(self, user: User, statement: Select[Any]) -> bool:
        for criterion in statement._where_criteria:
            left = getattr(criterion.left, "key", "")
            right = getattr(criterion.right, "value", None)
            if getattr(user, left) != right:
                return False
        return True
