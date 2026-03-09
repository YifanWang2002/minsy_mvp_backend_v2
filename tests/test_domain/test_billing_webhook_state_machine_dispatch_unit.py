"""Dispatch matrix tests for Stripe webhook -> subscription state transitions."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from apps.api.services.billing_webhook_service import BillingWebhookService


class _FakeDb:
    async def flush(self) -> None:
        return None


class _PinnedService(BillingWebhookService):
    def __init__(self, *, row: SimpleNamespace, sync_impl) -> None:
        super().__init__(_FakeDb(), stripe_client=SimpleNamespace())
        self._row = row
        self._sync = sync_impl

    async def _get_or_create_event_row(self, **_kwargs):
        return self._row


class _SyncRecorder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def sync_from_checkout_session(self, **_kwargs):
        self.calls.append("checkout")
        return SimpleNamespace(user_id="u_state", stripe_customer_id="cus_state")

    async def sync_subscription_payload(self, **_kwargs):
        self.calls.append("subscription")
        return SimpleNamespace(user_id="u_state", stripe_customer_id="cus_state")

    async def mark_subscription_ended(self, **_kwargs):
        self.calls.append("deleted")
        return SimpleNamespace(user_id="u_state", stripe_customer_id="cus_state")

    async def apply_invoice_event(self, **_kwargs):
        self.calls.append("invoice")
        return SimpleNamespace(user_id="u_state", stripe_customer_id="cus_state")


def _row() -> SimpleNamespace:
    return SimpleNamespace(
        processed_at=None,
        failed_at=None,
        processing_error=None,
        user_id=None,
        stripe_customer_id=None,
        event_type="unknown",
        livemode=False,
        payload={},
    )


@pytest.mark.parametrize(
    ("event_type", "expected_call", "object_payload"),
    [
        ("checkout.session.completed", "checkout", {"id": "cs_1", "customer": "cus_1"}),
        (
            "customer.subscription.updated",
            "subscription",
            {"id": "sub_1", "customer": "cus_1", "status": "active"},
        ),
        (
            "customer.subscription.deleted",
            "deleted",
            {"id": "sub_1", "customer": "cus_1", "status": "canceled"},
        ),
        ("invoice.paid", "invoice", {"id": "in_1", "subscription": "sub_1", "customer": "cus_1"}),
    ],
)
async def test_webhook_dispatch_matrix_routes_to_expected_sync_handler(
    event_type: str,
    expected_call: str,
    object_payload: dict,
) -> None:
    row = _row()
    recorder = _SyncRecorder()
    service = _PinnedService(row=row, sync_impl=recorder)

    event = {
        "id": f"evt_{expected_call}",
        "type": event_type,
        "livemode": False,
        "data": {"object": object_payload},
    }

    result = await service.process_event(event=event)

    assert result["status"] == "processed"
    assert result["duplicate"] is False
    assert recorder.calls == [expected_call]
    assert row.user_id == "u_state"
    assert row.stripe_customer_id == "cus_state"
    assert row.processed_at is not None
    assert row.failed_at is None
    assert row.processing_error is None


async def test_webhook_unknown_event_keeps_state_stable_and_marks_processed() -> None:
    row = _row()
    recorder = _SyncRecorder()
    service = _PinnedService(row=row, sync_impl=recorder)

    event = {
        "id": "evt_unknown_state",
        "type": "unknown.event.type",
        "livemode": False,
        "data": {"object": {"id": "obj_unknown", "customer": "cus_1"}},
    }

    result = await service.process_event(event=event)

    assert result["status"] == "processed"
    assert result["duplicate"] is False
    assert recorder.calls == []
    assert row.user_id is None
    assert row.stripe_customer_id is None
    assert isinstance(row.processed_at, datetime)
    assert row.processed_at.tzinfo is UTC
