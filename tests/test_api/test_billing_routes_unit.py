"""Unit coverage for billing checkout/change-plan route logic."""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from apps.api.main import create_app
from apps.api.routes import billing as billing_routes
from packages.shared_settings.schema.settings import settings


class _FakeDbSession:
    def __init__(self) -> None:
        self.commit_calls = 0

    async def commit(self) -> None:
        self.commit_calls += 1


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


def _fake_user(*, tier: str = "free") -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid4(),
        current_tier=tier,
        email="billing-test@example.com",
    )


def _override_dependencies(*, app, user, db):
    async def _override_user():
        return user

    async def _override_db():
        yield db

    app.dependency_overrides[billing_routes.get_current_user] = _override_user
    app.dependency_overrides[billing_routes.get_db] = _override_db


def _patch_price_settings(monkeypatch):
    monkeypatch.setattr(settings, "stripe_price_go_monthly", "price_go_monthly")
    monkeypatch.setattr(settings, "stripe_price_plus_monthly", "price_plus_monthly")
    monkeypatch.setattr(settings, "stripe_price_pro_monthly", "price_pro_monthly")
    monkeypatch.setattr(settings, "stripe_publishable_key", "pk_test_123")


@pytest.fixture(autouse=True)
def _stripe_ready(monkeypatch):
    monkeypatch.setattr(billing_routes, "_require_stripe_ready", lambda: None)


@pytest.fixture(autouse=True)
def _no_remote_active_subscription(monkeypatch):
    async def _fake_remote_active(*, customer_id):
        del customer_id
        return None

    monkeypatch.setattr(
        billing_routes,
        "_latest_remote_active_subscription_for_customer",
        _fake_remote_active,
    )


def test_checkout_session_applies_7day_trial_for_first_paid_user(
    client,
    app,
    monkeypatch,
):
    _patch_price_settings(monkeypatch)
    user = _fake_user()
    db = _FakeDbSession()
    _override_dependencies(app=app, user=user, db=db)

    captured: dict[str, object] = {}

    async def _fake_latest_active(*, db, user_id):
        del db, user_id
        return None

    async def _fake_has_paid_history(*, db, user_id):
        del db, user_id
        return False

    async def _fake_resolve_customer(*, db, user):
        del db, user
        return SimpleNamespace(stripe_customer_id="cus_123")

    async def _fake_create_checkout_session(**kwargs):
        captured.update(kwargs)
        return {
            "id": "cs_test_1",
            "url": "https://checkout.stripe.test/session-1",
        }

    monkeypatch.setattr(
        billing_routes,
        "_latest_active_subscription_for_user",
        _fake_latest_active,
    )
    monkeypatch.setattr(
        billing_routes,
        "_user_has_paid_subscription_history",
        _fake_has_paid_history,
    )
    monkeypatch.setattr(
        billing_routes,
        "_resolve_or_create_customer",
        _fake_resolve_customer,
    )
    monkeypatch.setattr(
        billing_routes.stripe_client,
        "create_checkout_session",
        _fake_create_checkout_session,
    )

    response = client.post("/api/v1/billing/checkout-session", json={"plan": "go"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"] == "cs_test_1"
    assert captured["trial_days"] == 7
    assert captured["price_id"] == "price_go_monthly"
    assert db.commit_calls == 1

    app.dependency_overrides.clear()


def test_checkout_session_uses_no_trial_for_returning_paid_user(
    client,
    app,
    monkeypatch,
):
    _patch_price_settings(monkeypatch)
    user = _fake_user()
    db = _FakeDbSession()
    _override_dependencies(app=app, user=user, db=db)

    captured: dict[str, object] = {}

    async def _fake_latest_active(*, db, user_id):
        del db, user_id
        return None

    async def _fake_has_paid_history(*, db, user_id):
        del db, user_id
        return True

    async def _fake_resolve_customer(*, db, user):
        del db, user
        return SimpleNamespace(stripe_customer_id="cus_123")

    async def _fake_create_checkout_session(**kwargs):
        captured.update(kwargs)
        return {
            "id": "cs_test_2",
            "url": "https://checkout.stripe.test/session-2",
        }

    monkeypatch.setattr(
        billing_routes,
        "_latest_active_subscription_for_user",
        _fake_latest_active,
    )
    monkeypatch.setattr(
        billing_routes,
        "_user_has_paid_subscription_history",
        _fake_has_paid_history,
    )
    monkeypatch.setattr(
        billing_routes,
        "_resolve_or_create_customer",
        _fake_resolve_customer,
    )
    monkeypatch.setattr(
        billing_routes.stripe_client,
        "create_checkout_session",
        _fake_create_checkout_session,
    )

    response = client.post("/api/v1/billing/checkout-session", json={"plan": "go"})

    assert response.status_code == 200
    assert captured["trial_days"] == 0

    app.dependency_overrides.clear()


def test_checkout_session_rejects_when_active_subscription_exists(
    client,
    app,
    monkeypatch,
):
    _patch_price_settings(monkeypatch)
    user = _fake_user(tier="plus")
    db = _FakeDbSession()
    _override_dependencies(app=app, user=user, db=db)

    async def _fake_latest_active(*, db, user_id):
        del db, user_id
        return SimpleNamespace(id=uuid4(), status="active")

    called = {"checkout": 0}

    async def _fake_create_checkout_session(**kwargs):
        del kwargs
        called["checkout"] += 1
        return {}

    monkeypatch.setattr(
        billing_routes,
        "_latest_active_subscription_for_user",
        _fake_latest_active,
    )
    monkeypatch.setattr(
        billing_routes.stripe_client,
        "create_checkout_session",
        _fake_create_checkout_session,
    )

    response = client.post("/api/v1/billing/checkout-session", json={"plan": "pro"})

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["code"] == "BILLING_ACTIVE_SUBSCRIPTION_EXISTS"
    assert called["checkout"] == 0

    app.dependency_overrides.clear()


def test_checkout_session_rejects_when_remote_active_subscription_exists(
    client,
    app,
    monkeypatch,
):
    _patch_price_settings(monkeypatch)
    user = _fake_user(tier="plus")
    db = _FakeDbSession()
    _override_dependencies(app=app, user=user, db=db)

    async def _fake_latest_active(*, db, user_id):
        del db, user_id
        return None

    async def _fake_resolve_customer(*, db, user):
        del db, user
        return SimpleNamespace(stripe_customer_id="cus_456")

    async def _fake_remote_active(*, customer_id):
        assert customer_id == "cus_456"
        return {
            "id": "sub_remote_1",
            "status": "active",
            "customer": customer_id,
            "items": {
                "data": [
                    {"id": "si_remote_1", "price": {"id": "price_plus_monthly"}}
                ]
            },
        }

    sync_calls: list[dict] = []

    class _FakeSyncService:
        def __init__(self, _db, *, stripe_client):
            del _db, stripe_client

        async def sync_subscription_payload(self, *, subscription_payload, fallback_user_id):
            sync_calls.append(
                {
                    "subscription_payload": subscription_payload,
                    "fallback_user_id": fallback_user_id,
                }
            )
            return None

    called = {"checkout": 0}

    async def _fake_create_checkout_session(**kwargs):
        del kwargs
        called["checkout"] += 1
        return {}

    monkeypatch.setattr(
        billing_routes,
        "_latest_active_subscription_for_user",
        _fake_latest_active,
    )
    monkeypatch.setattr(
        billing_routes,
        "_resolve_or_create_customer",
        _fake_resolve_customer,
    )
    monkeypatch.setattr(
        billing_routes,
        "_latest_remote_active_subscription_for_customer",
        _fake_remote_active,
    )
    monkeypatch.setattr(billing_routes, "SubscriptionSyncService", _FakeSyncService)
    monkeypatch.setattr(
        billing_routes.stripe_client,
        "create_checkout_session",
        _fake_create_checkout_session,
    )

    response = client.post("/api/v1/billing/checkout-session", json={"plan": "pro"})

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["code"] == "BILLING_ACTIVE_SUBSCRIPTION_EXISTS"
    assert called["checkout"] == 0
    assert len(sync_calls) == 1
    assert sync_calls[0]["fallback_user_id"] == user.id
    assert db.commit_calls == 1

    app.dependency_overrides.clear()


def test_change_plan_without_active_subscription_redirects_to_checkout(
    client,
    app,
    monkeypatch,
):
    _patch_price_settings(monkeypatch)
    user = _fake_user()
    db = _FakeDbSession()
    _override_dependencies(app=app, user=user, db=db)

    captured: dict[str, object] = {}

    async def _fake_latest_active(*, db, user_id):
        del db, user_id
        return None

    async def _fake_has_paid_history(*, db, user_id):
        del db, user_id
        return False

    async def _fake_resolve_customer(*, db, user):
        del db, user
        return SimpleNamespace(stripe_customer_id="cus_abc")

    async def _fake_create_checkout_session(**kwargs):
        captured.update(kwargs)
        return {
            "id": "cs_test_3",
            "url": "https://checkout.stripe.test/session-3",
        }

    monkeypatch.setattr(
        billing_routes,
        "_latest_active_subscription_for_user",
        _fake_latest_active,
    )
    monkeypatch.setattr(
        billing_routes,
        "_user_has_paid_subscription_history",
        _fake_has_paid_history,
    )
    monkeypatch.setattr(
        billing_routes,
        "_resolve_or_create_customer",
        _fake_resolve_customer,
    )
    monkeypatch.setattr(
        billing_routes.stripe_client,
        "create_checkout_session",
        _fake_create_checkout_session,
    )

    response = client.post("/api/v1/billing/change-plan", json={"plan": "plus"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["action"] == "checkout_redirect"
    assert payload["target_tier"] == "plus"
    assert captured["trial_days"] == 7
    assert captured["price_id"] == "price_plus_monthly"

    app.dependency_overrides.clear()


def test_change_plan_upgrade_updates_existing_subscription_immediately(
    client,
    app,
    monkeypatch,
):
    _patch_price_settings(monkeypatch)
    user = _fake_user(tier="go")
    db = _FakeDbSession()
    _override_dependencies(app=app, user=user, db=db)

    async def _fake_latest_active(*, db, user_id):
        del db, user_id
        return SimpleNamespace(
            stripe_subscription_id="sub_123",
            tier="go",
        )

    async def _fake_retrieve_subscription(_subscription_id):
        return {
            "id": "sub_123",
            "items": {
                "data": [
                    {
                        "id": "si_123",
                        "price": {"id": "price_go_monthly"},
                    }
                ]
            },
            "latest_invoice": {"status": "paid"},
            "current_period_end": 1_893_456_000,
        }

    captured: dict[str, object] = {}

    async def _fake_update_subscription_price(
        _subscription_id,
        *,
        subscription_item_id,
        price_id,
        proration_behavior,
        payment_behavior,
        metadata,
    ):
        captured.update(
            {
                "subscription_item_id": subscription_item_id,
                "price_id": price_id,
                "proration_behavior": proration_behavior,
                "payment_behavior": payment_behavior,
                "metadata": metadata,
            }
        )
        return {
            "id": "sub_123",
            "status": "active",
            "items": {
                "data": [
                    {
                        "id": subscription_item_id,
                        "price": {"id": price_id},
                    }
                ]
            },
        }

    sync_calls: list[dict] = []

    class _FakeSyncService:
        def __init__(self, _db, *, stripe_client):
            del _db, stripe_client

        async def sync_subscription_payload(self, *, subscription_payload, fallback_user_id):
            sync_calls.append(
                {
                    "subscription_payload": subscription_payload,
                    "fallback_user_id": fallback_user_id,
                }
            )
            return None

    monkeypatch.setattr(
        billing_routes,
        "_latest_active_subscription_for_user",
        _fake_latest_active,
    )
    monkeypatch.setattr(
        billing_routes.stripe_client,
        "retrieve_subscription",
        _fake_retrieve_subscription,
    )
    monkeypatch.setattr(
        billing_routes.stripe_client,
        "update_subscription_price",
        _fake_update_subscription_price,
    )
    monkeypatch.setattr(billing_routes, "SubscriptionSyncService", _FakeSyncService)

    response = client.post("/api/v1/billing/change-plan", json={"plan": "plus"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["action"] == "updated"
    assert payload["current_tier"] == "go"
    assert payload["target_tier"] == "plus"
    assert captured["price_id"] == "price_plus_monthly"
    assert captured["proration_behavior"] == "always_invoice"
    assert captured["payment_behavior"] == "pending_if_incomplete"
    assert captured["metadata"] is None
    assert len(sync_calls) == 1

    app.dependency_overrides.clear()


def test_change_plan_downgrade_holds_entitlements_until_period_end_when_paid(
    client,
    app,
    monkeypatch,
):
    _patch_price_settings(monkeypatch)
    user = _fake_user(tier="pro")
    db = _FakeDbSession()
    _override_dependencies(app=app, user=user, db=db)

    period_end = 1_893_456_789

    async def _fake_latest_active(*, db, user_id):
        del db, user_id
        return SimpleNamespace(
            stripe_subscription_id="sub_789",
            tier="pro",
        )

    async def _fake_retrieve_subscription(_subscription_id):
        return {
            "id": "sub_789",
            "items": {
                "data": [
                    {
                        "id": "si_789",
                        "price": {"id": "price_pro_monthly"},
                    }
                ]
            },
            "latest_invoice": {"status": "paid"},
            "current_period_end": period_end,
        }

    captured: dict[str, object] = {}

    async def _fake_update_subscription_price(
        _subscription_id,
        *,
        subscription_item_id,
        price_id,
        proration_behavior,
        payment_behavior,
        metadata,
    ):
        captured.update(
            {
                "subscription_item_id": subscription_item_id,
                "price_id": price_id,
                "proration_behavior": proration_behavior,
                "payment_behavior": payment_behavior,
                "metadata": metadata,
            }
        )
        return {
            "id": "sub_789",
            "status": "active",
            "items": {
                "data": [
                    {
                        "id": subscription_item_id,
                        "price": {"id": price_id},
                    }
                ]
            },
            "metadata": metadata,
        }

    class _FakeSyncService:
        def __init__(self, _db, *, stripe_client):
            del _db, stripe_client

        async def sync_subscription_payload(self, *, subscription_payload, fallback_user_id):
            del subscription_payload, fallback_user_id
            return None

    monkeypatch.setattr(
        billing_routes,
        "_latest_active_subscription_for_user",
        _fake_latest_active,
    )
    monkeypatch.setattr(
        billing_routes.stripe_client,
        "retrieve_subscription",
        _fake_retrieve_subscription,
    )
    monkeypatch.setattr(
        billing_routes.stripe_client,
        "update_subscription_price",
        _fake_update_subscription_price,
    )
    monkeypatch.setattr(billing_routes, "SubscriptionSyncService", _FakeSyncService)

    response = client.post("/api/v1/billing/change-plan", json={"plan": "plus"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["action"] == "updated"
    assert payload["current_tier"] == "pro"
    assert payload["target_tier"] == "plus"
    assert captured["price_id"] == "price_plus_monthly"
    assert captured["proration_behavior"] == "none"
    assert captured["payment_behavior"] == "allow_incomplete"
    assert captured["metadata"] == {
        "entitlements_hold_until": str(period_end),
        "pending_tier": "plus",
    }

    app.dependency_overrides.clear()


def test_active_subscription_statuses_include_paused():
    assert "paused" in billing_routes._ACTIVE_SUBSCRIPTION_STATUSES


def _tier_rank(tier: str) -> int:
    return {"free": 0, "go": 1, "plus": 2, "pro": 3}.get(tier, 0)


@pytest.mark.parametrize("target_tier", ["go", "plus", "pro"])
def test_change_plan_state_machine_from_free_to_paid_redirects_checkout(
    client,
    app,
    monkeypatch,
    target_tier,
):
    _patch_price_settings(monkeypatch)
    user = _fake_user(tier="free")
    db = _FakeDbSession()
    _override_dependencies(app=app, user=user, db=db)

    captured: dict[str, object] = {}

    async def _fake_latest_active(*, db, user_id):
        del db, user_id
        return None

    async def _fake_has_paid_history(*, db, user_id):
        del db, user_id
        return False

    async def _fake_resolve_customer(*, db, user):
        del db, user
        return SimpleNamespace(stripe_customer_id="cus_state_free")

    async def _fake_create_checkout_session(**kwargs):
        captured.update(kwargs)
        return {
            "id": f"cs_state_{target_tier}",
            "url": f"https://checkout.stripe.test/state-{target_tier}",
        }

    monkeypatch.setattr(
        billing_routes,
        "_latest_active_subscription_for_user",
        _fake_latest_active,
    )
    monkeypatch.setattr(
        billing_routes,
        "_user_has_paid_subscription_history",
        _fake_has_paid_history,
    )
    monkeypatch.setattr(
        billing_routes,
        "_resolve_or_create_customer",
        _fake_resolve_customer,
    )
    monkeypatch.setattr(
        billing_routes.stripe_client,
        "create_checkout_session",
        _fake_create_checkout_session,
    )

    response = client.post("/api/v1/billing/change-plan", json={"plan": target_tier})

    assert response.status_code == 200
    payload = response.json()
    assert payload["action"] == "checkout_redirect"
    assert payload["current_tier"] == "free"
    assert payload["target_tier"] == target_tier
    assert captured["trial_days"] == 7
    assert captured["price_id"] == {
        "go": "price_go_monthly",
        "plus": "price_plus_monthly",
        "pro": "price_pro_monthly",
    }[target_tier]

    app.dependency_overrides.clear()


@pytest.mark.parametrize(
    ("current_tier", "target_tier"),
    [
        ("go", "go"),
        ("go", "plus"),
        ("go", "pro"),
        ("plus", "go"),
        ("plus", "plus"),
        ("plus", "pro"),
        ("pro", "go"),
        ("pro", "plus"),
        ("pro", "pro"),
    ],
)
def test_change_plan_state_machine_for_paid_tiers_all_paths(
    client,
    app,
    monkeypatch,
    current_tier,
    target_tier,
):
    _patch_price_settings(monkeypatch)
    user = _fake_user(tier=current_tier)
    db = _FakeDbSession()
    _override_dependencies(app=app, user=user, db=db)

    period_end = 1_893_456_111
    current_price = {
        "go": "price_go_monthly",
        "plus": "price_plus_monthly",
        "pro": "price_pro_monthly",
    }[current_tier]
    target_price = {
        "go": "price_go_monthly",
        "plus": "price_plus_monthly",
        "pro": "price_pro_monthly",
    }[target_tier]

    async def _fake_latest_active(*, db, user_id):
        del db, user_id
        return SimpleNamespace(
            stripe_subscription_id=f"sub_{current_tier}",
            tier=current_tier,
        )

    async def _fake_retrieve_subscription(_subscription_id):
        return {
            "id": f"sub_{current_tier}",
            "items": {
                "data": [
                    {
                        "id": f"si_{current_tier}",
                        "price": {"id": current_price},
                    }
                ]
            },
            "latest_invoice": {"status": "paid"},
            "current_period_end": period_end,
        }

    captured_updates: list[dict[str, object]] = []

    async def _fake_update_subscription_price(
        _subscription_id,
        *,
        subscription_item_id,
        price_id,
        proration_behavior,
        payment_behavior,
        metadata,
    ):
        captured_updates.append(
            {
                "subscription_item_id": subscription_item_id,
                "price_id": price_id,
                "proration_behavior": proration_behavior,
                "payment_behavior": payment_behavior,
                "metadata": metadata,
            }
        )
        return {
            "id": f"sub_{current_tier}",
            "status": "active",
            "items": {
                "data": [
                    {
                        "id": subscription_item_id,
                        "price": {"id": price_id},
                    }
                ]
            },
            "metadata": metadata or {},
        }

    sync_calls: list[dict] = []

    class _FakeSyncService:
        def __init__(self, _db, *, stripe_client):
            del _db, stripe_client

        async def sync_subscription_payload(self, *, subscription_payload, fallback_user_id):
            sync_calls.append(
                {
                    "subscription_payload": subscription_payload,
                    "fallback_user_id": fallback_user_id,
                }
            )
            return None

    monkeypatch.setattr(
        billing_routes,
        "_latest_active_subscription_for_user",
        _fake_latest_active,
    )
    monkeypatch.setattr(
        billing_routes.stripe_client,
        "retrieve_subscription",
        _fake_retrieve_subscription,
    )
    monkeypatch.setattr(
        billing_routes.stripe_client,
        "update_subscription_price",
        _fake_update_subscription_price,
    )
    monkeypatch.setattr(billing_routes, "SubscriptionSyncService", _FakeSyncService)

    response = client.post("/api/v1/billing/change-plan", json={"plan": target_tier})

    assert response.status_code == 200
    payload = response.json()
    assert payload["current_tier"] == current_tier
    assert payload["target_tier"] == target_tier

    if current_tier == target_tier:
        assert payload["action"] == "noop"
        assert captured_updates == []
        assert len(sync_calls) == 1
    else:
        assert payload["action"] == "updated"
        assert len(captured_updates) == 1
        update_call = captured_updates[0]
        assert update_call["price_id"] == target_price
        if _tier_rank(target_tier) > _tier_rank(current_tier):
            assert update_call["proration_behavior"] == "always_invoice"
            assert update_call["payment_behavior"] == "pending_if_incomplete"
            assert update_call["metadata"] is None
        else:
            assert update_call["proration_behavior"] == "none"
            assert update_call["payment_behavior"] == "allow_incomplete"
            assert update_call["metadata"] == {
                "entitlements_hold_until": str(period_end),
                "pending_tier": target_tier,
            }
        assert len(sync_calls) == 1

    app.dependency_overrides.clear()


def test_change_plan_state_machine_recovers_remote_active_subscription_and_updates(
    client,
    app,
    monkeypatch,
):
    _patch_price_settings(monkeypatch)
    user = _fake_user(tier="free")
    db = _FakeDbSession()
    _override_dependencies(app=app, user=user, db=db)

    async def _fake_latest_active(*, db, user_id):
        del db, user_id
        return None

    async def _fake_resolve_customer(*, db, user):
        del db, user
        return SimpleNamespace(stripe_customer_id="cus_remote_recover")

    async def _fake_remote_active(*, customer_id):
        assert customer_id == "cus_remote_recover"
        return {
            "id": "sub_remote_recover",
            "customer": customer_id,
            "status": "active",
            "items": {
                "data": [
                    {
                        "id": "si_remote_recover",
                        "price": {"id": "price_go_monthly"},
                    }
                ]
            },
            "latest_invoice": {"status": "paid"},
            "current_period_end": 1_893_456_222,
        }

    update_calls: list[dict[str, object]] = []

    async def _fake_update_subscription_price(
        _subscription_id,
        *,
        subscription_item_id,
        price_id,
        proration_behavior,
        payment_behavior,
        metadata,
    ):
        update_calls.append(
            {
                "subscription_item_id": subscription_item_id,
                "price_id": price_id,
                "proration_behavior": proration_behavior,
                "payment_behavior": payment_behavior,
                "metadata": metadata,
            }
        )
        return {
            "id": "sub_remote_recover",
            "customer": "cus_remote_recover",
            "status": "active",
            "items": {
                "data": [
                    {
                        "id": "si_remote_recover",
                        "price": {"id": price_id},
                    }
                ]
            },
            "latest_invoice": {"status": "paid"},
        }

    sync_calls: list[dict] = []

    class _FakeSyncService:
        def __init__(self, _db, *, stripe_client):
            del _db, stripe_client

        async def sync_subscription_payload(self, *, subscription_payload, fallback_user_id):
            sync_calls.append(
                {
                    "subscription_payload": subscription_payload,
                    "fallback_user_id": fallback_user_id,
                }
            )
            return None

    monkeypatch.setattr(
        billing_routes,
        "_latest_active_subscription_for_user",
        _fake_latest_active,
    )
    monkeypatch.setattr(
        billing_routes,
        "_resolve_or_create_customer",
        _fake_resolve_customer,
    )
    monkeypatch.setattr(
        billing_routes,
        "_latest_remote_active_subscription_for_customer",
        _fake_remote_active,
    )
    monkeypatch.setattr(
        billing_routes.stripe_client,
        "update_subscription_price",
        _fake_update_subscription_price,
    )
    monkeypatch.setattr(billing_routes, "SubscriptionSyncService", _FakeSyncService)

    response = client.post("/api/v1/billing/change-plan", json={"plan": "plus"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["action"] == "updated"
    assert payload["current_tier"] == "go"
    assert payload["target_tier"] == "plus"
    assert len(update_calls) == 1
    assert update_calls[0]["price_id"] == "price_plus_monthly"
    assert len(sync_calls) == 1

    app.dependency_overrides.clear()
