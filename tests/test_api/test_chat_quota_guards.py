"""Quota guard tests for chat routes."""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from apps.api.main import create_app
from apps.api.routes import chat as chat_routes
from packages.domain.billing.quota_service import QuotaExceededError
from packages.domain.billing.usage_service import UsageMetric


class _FakeDbSession:
    async def commit(self) -> None:
        return None

    async def refresh(self, _obj: object) -> None:
        return None


@pytest.fixture()
def app(monkeypatch):
    with monkeypatch.context() as patch_ctx:
        patch_ctx.setattr("apps.api.main.lifespan", _noop_lifespan)
        test_app = create_app()
        yield test_app


@pytest.fixture()
def client(app):
    return TestClient(app)


@asynccontextmanager
async def _noop_lifespan(_):
    yield


def _fake_user():
    return SimpleNamespace(
        id=uuid4(),
        current_tier="free",
        profiles=[],
    )


class _FakeOrchestrator:
    def __init__(self, _db) -> None:
        self._db = _db

    async def create_session(self, *, user_id, parent_session_id, metadata):
        del user_id, parent_session_id, metadata
        return SimpleNamespace(
            id=uuid4(),
            current_phase="kyc",
            status="active",
            metadata_={},
        )


def test_new_thread_returns_402_when_quota_exceeded(client, app, monkeypatch):
    user = _fake_user()
    db = _FakeDbSession()

    class _QuotaServiceRaise:
        def __init__(self, _usage) -> None:
            del _usage

        async def assert_quota_available(self, **_kwargs):
            raise QuotaExceededError(
                metric=UsageMetric.AI_TOKENS_MONTHLY_TOTAL,
                tier="free",
                used=100,
                limit=100,
                remaining=0,
                reset_at=None,
            )

    async def _override_user():
        return user

    async def _override_db():
        yield db

    monkeypatch.setattr(chat_routes, "ChatOrchestrator", _FakeOrchestrator)
    monkeypatch.setattr(chat_routes, "QuotaService", _QuotaServiceRaise)
    app.dependency_overrides[chat_routes.get_current_user] = _override_user
    app.dependency_overrides[chat_routes.get_db] = _override_db

    response = client.post(
        "/api/v1/chat/new-thread",
        json={"metadata": {}},
    )

    assert response.status_code == 402
    detail = response.json()["detail"]
    assert detail["code"] == "QUOTA_EXCEEDED"
    assert detail["metric"] == UsageMetric.AI_TOKENS_MONTHLY_TOTAL

    app.dependency_overrides.clear()


def test_new_thread_checks_ai_token_quota_before_create(client, app, monkeypatch):
    user = _fake_user()
    db = _FakeDbSession()
    quota_calls: list[dict] = []

    class _QuotaServiceCapture:
        def __init__(self, _usage) -> None:
            del _usage

        async def assert_quota_available(self, **kwargs):
            quota_calls.append(kwargs)

    async def _override_user():
        return user

    async def _override_db():
        yield db

    monkeypatch.setattr(chat_routes, "ChatOrchestrator", _FakeOrchestrator)
    monkeypatch.setattr(chat_routes, "QuotaService", _QuotaServiceCapture)
    app.dependency_overrides[chat_routes.get_current_user] = _override_user
    app.dependency_overrides[chat_routes.get_db] = _override_db

    response = client.post(
        "/api/v1/chat/new-thread",
        json={"metadata": {}},
    )

    assert response.status_code == 201
    assert len(quota_calls) == 1
    assert quota_calls[0]["metric"] == UsageMetric.AI_TOKENS_MONTHLY_TOTAL
    assert quota_calls[0]["increment"] == 1

    app.dependency_overrides.clear()
