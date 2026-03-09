"""State-machine unit tests for subscription sync and tier/quota effects."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import uuid4

from packages.domain.billing.subscription_sync_service import SubscriptionSyncService
from packages.shared_settings.schema.settings import settings


class _FakeDb:
    def __init__(self, *, scalar_results: list[object | None]) -> None:
        self._scalar_results = list(scalar_results)
        self.added: list[object] = []
        self.flush_calls = 0

    async def scalar(self, _query):
        if not self._scalar_results:
            return None
        return self._scalar_results.pop(0)

    def add(self, obj: object) -> None:
        self.added.append(obj)

    async def flush(self) -> None:
        self.flush_calls += 1


class _CaptureSyncService(SubscriptionSyncService):
    def __init__(self, db: _FakeDb, *, fixed_user_id):
        super().__init__(db, stripe_client=SimpleNamespace())
        self._fixed_user_id = fixed_user_id
        self.synced_tiers: list[str] = []

    async def _resolve_user_id(self, *, stripe_customer_id: str, fallback_user_id):
        del stripe_customer_id, fallback_user_id
        return self._fixed_user_id

    async def _sync_user_tier(self, *, user_id, target_tier: str) -> None:
        assert user_id == self._fixed_user_id
        self.synced_tiers.append(target_tier)


def _patch_price_settings(monkeypatch) -> None:
    monkeypatch.setattr(settings, "stripe_price_go_monthly", "price_go_monthly")
    monkeypatch.setattr(settings, "stripe_price_plus_monthly", "price_plus_monthly")
    monkeypatch.setattr(settings, "stripe_price_pro_monthly", "price_pro_monthly")
    monkeypatch.setattr(settings, "stripe_product_go", "prod_state_machine")


def _subscription_payload(*, price_id: str, status: str = "active", metadata: dict | None = None):
    return {
        "id": "sub_state_1",
        "customer": "cus_state_1",
        "status": status,
        "items": {
            "data": [
                {
                    "id": "si_state_1",
                    "price": {
                        "id": price_id,
                        "product": "prod_state_machine",
                    },
                }
            ]
        },
        "latest_invoice": {"id": "in_state_1", "status": "paid"},
        "current_period_start": int(datetime.now(UTC).timestamp()),
        "current_period_end": int((datetime.now(UTC) + timedelta(days=30)).timestamp()),
        "metadata": metadata or {},
    }


async def test_sync_keeps_higher_entitlements_until_hold_expires(monkeypatch) -> None:
    _patch_price_settings(monkeypatch)
    user_id = uuid4()
    customer = SimpleNamespace(id=uuid4())
    existing_record = SimpleNamespace(tier="pro")
    db = _FakeDb(scalar_results=[customer, existing_record])
    service = _CaptureSyncService(db, fixed_user_id=user_id)

    hold_until = int((datetime.now(UTC) + timedelta(days=7)).timestamp())
    payload = _subscription_payload(
        price_id="price_plus_monthly",
        status="active",
        metadata={
            "entitlements_hold_until": str(hold_until),
            "pending_tier": "plus",
        },
    )

    record = await service.sync_subscription_payload(
        subscription_payload=payload,
        fallback_user_id=user_id,
        latest_event_id="evt_hold",
    )

    assert record is not None
    assert record.tier == "pro"
    assert record.pending_tier == "plus"
    assert record.pending_price_id == "price_plus_monthly"
    assert service.synced_tiers == ["pro"]


async def test_sync_applies_downgrade_after_hold_window(monkeypatch) -> None:
    _patch_price_settings(monkeypatch)
    user_id = uuid4()
    customer = SimpleNamespace(id=uuid4())
    existing_record = SimpleNamespace(tier="pro")
    db = _FakeDb(scalar_results=[customer, existing_record])
    service = _CaptureSyncService(db, fixed_user_id=user_id)

    hold_until = int((datetime.now(UTC) - timedelta(minutes=1)).timestamp())
    payload = _subscription_payload(
        price_id="price_plus_monthly",
        status="active",
        metadata={
            "entitlements_hold_until": str(hold_until),
            "pending_tier": "plus",
        },
    )

    record = await service.sync_subscription_payload(
        subscription_payload=payload,
        fallback_user_id=user_id,
        latest_event_id="evt_after_hold",
    )

    assert record is not None
    assert record.tier == "plus"
    assert record.pending_tier is None
    assert service.synced_tiers == ["plus"]


async def test_sync_sets_free_tier_when_subscription_status_inactive(monkeypatch) -> None:
    _patch_price_settings(monkeypatch)
    user_id = uuid4()
    customer = SimpleNamespace(id=uuid4())
    db = _FakeDb(scalar_results=[customer, None])
    service = _CaptureSyncService(db, fixed_user_id=user_id)

    payload = _subscription_payload(
        price_id="price_pro_monthly",
        status="canceled",
    )

    record = await service.sync_subscription_payload(
        subscription_payload=payload,
        fallback_user_id=user_id,
        latest_event_id="evt_canceled",
    )

    assert record is not None
    assert record.tier == "free"
    assert service.synced_tiers == ["free"]


async def test_mark_subscription_ended_sets_free_and_clears_pending(monkeypatch) -> None:
    _patch_price_settings(monkeypatch)
    user_id = uuid4()
    record = SimpleNamespace(
        user_id=user_id,
        status="active",
        tier="plus",
        pending_tier="go",
        pending_price_id="price_go_monthly",
        ended_at=None,
        latest_event_id=None,
        synced_at=None,
    )
    db = _FakeDb(scalar_results=[record])
    service = _CaptureSyncService(db, fixed_user_id=user_id)

    ended = await service.mark_subscription_ended(
        stripe_subscription_id="sub_state_1",
        latest_event_id="evt_deleted",
    )

    assert ended is record
    assert record.status == "canceled"
    assert record.tier == "free"
    assert record.pending_tier is None
    assert record.pending_price_id is None
    assert service.synced_tiers == ["free"]
