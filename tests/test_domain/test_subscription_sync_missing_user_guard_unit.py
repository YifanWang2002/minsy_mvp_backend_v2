"""Guards against billing sync writes for missing users."""

from __future__ import annotations

from uuid import uuid4

from packages.domain.billing.subscription_sync_service import SubscriptionSyncService


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


class _FakeStripeClient:
    def __init__(self, *, subscription_payload: dict[str, object]) -> None:
        self._subscription_payload = dict(subscription_payload)
        self.requested_subscription_ids: list[str] = []

    async def retrieve_subscription(self, subscription_id: str) -> dict[str, object]:
        self.requested_subscription_ids.append(subscription_id)
        return dict(self._subscription_payload)


async def test_resolve_user_id_returns_none_when_fallback_user_missing() -> None:
    fallback_user_id = uuid4()
    db = _FakeDb(scalar_results=[None, None])
    service = SubscriptionSyncService(
        db,
        stripe_client=_FakeStripeClient(subscription_payload={}),
    )

    resolved = await service._resolve_user_id(
        stripe_customer_id="cus_missing",
        fallback_user_id=fallback_user_id,
    )

    assert resolved is None
    assert db.flush_calls == 0


async def test_checkout_sync_skips_missing_user_mapping_without_fk_failure() -> None:
    missing_user_id = uuid4()
    db = _FakeDb(scalar_results=[None, None])
    stripe = _FakeStripeClient(
        subscription_payload={
            "id": "sub_missing_user",
            "customer": "cus_missing_user",
            "status": "active",
            "items": {"data": []},
        }
    )
    service = SubscriptionSyncService(db, stripe_client=stripe)

    result = await service.sync_from_checkout_session(
        checkout_session={
            "customer": "cus_missing_user",
            "subscription": "sub_missing_user",
            "metadata": {"user_id": str(missing_user_id)},
            "client_reference_id": str(missing_user_id),
            "customer_details": {"email": "1@test.com"},
        },
        latest_event_id="evt_missing_user",
    )

    assert result is None
    assert db.added == []
    assert db.flush_calls == 0
    assert stripe.requested_subscription_ids == ["sub_missing_user"]
