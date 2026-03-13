"""Unit coverage for billing overview SSE stream route."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from apps.api.main import create_app
from apps.api.routes import billing as billing_routes
from apps.api.schemas.billing_schemas import (
    BillingOverviewResponse,
    BillingQuotaMetricResponse,
    BillingSubscriptionResponse,
)


class _FakeDbSession:
    def __init__(self, *, tier: str = "free") -> None:
        self._tier = tier

    def expire_all(self) -> None:
        return None

    async def scalar(self, _stmt):
        return self._tier

    async def rollback(self) -> None:
        return None


@asynccontextmanager
async def _noop_lifespan(_):
    yield


@pytest.fixture()
def app(monkeypatch):
    with monkeypatch.context() as patch_ctx:
        patch_ctx.setattr("apps.api.main.lifespan", _noop_lifespan)
        test_app = create_app()
        yield test_app


@pytest.fixture()
def client(app):
    return TestClient(app)


def _override_dependencies(*, app, user, db):
    async def _override_user():
        return user

    async def _override_db():
        yield db

    app.dependency_overrides[billing_routes.get_current_user] = _override_user
    app.dependency_overrides[billing_routes.get_db] = _override_db


def test_overview_stream_emits_snapshot_event(client, app, monkeypatch):
    user = SimpleNamespace(
        id=uuid4(),
        current_tier="free",
        email="stream-test@example.com",
    )
    db = _FakeDbSession(tier="free")
    _override_dependencies(app=app, user=user, db=db)

    async def _fake_overview(*, db, user_id, current_tier):
        del db, user_id, current_tier
        return BillingOverviewResponse(
            tier="free",
            subscription=BillingSubscriptionResponse(
                status="inactive",
                tier="free",
            ),
            quotas=[
                BillingQuotaMetricResponse(
                    metric="cpu_tokens_monthly_total",
                    used=12,
                    limit=30,
                    remaining=18,
                    reset_at=datetime(2026, 4, 1, tzinfo=UTC),
                )
            ],
            cost_model={"cpu_bars_per_token": 52560.0},
        )

    monkeypatch.setattr(
        billing_routes,
        "_build_billing_overview_response",
        _fake_overview,
    )

    event_name = None
    payload_raw = None
    with client.stream(
        "GET",
        "/api/v1/billing/overview/stream?poll_seconds=0.5&max_events=1",
    ) as response:
        assert response.status_code == 200
        for line in response.iter_lines():
            if not line:
                continue
            if line.startswith("event: "):
                event_name = line.removeprefix("event: ").strip()
            if line.startswith("data: "):
                payload_raw = line.removeprefix("data: ").strip()
                break

    assert event_name == "snapshot"
    assert payload_raw is not None
    payload = json.loads(payload_raw)
    overview = payload["overview"]
    quota = overview["quotas"][0]
    assert overview["tier"] == "free"
    assert quota["metric"] == "cpu_tokens_monthly_total"
    assert quota["used"] == 12
    assert quota["limit"] == 30
    assert quota["remaining"] == 18

    app.dependency_overrides.clear()
