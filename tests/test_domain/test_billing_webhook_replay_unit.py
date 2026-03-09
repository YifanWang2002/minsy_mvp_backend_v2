"""Unit tests for Stripe webhook replay behavior."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

from apps.api.services.billing_webhook_service import BillingWebhookService


class _FakeDb:
    async def flush(self) -> None:
        return None


class _ServiceWithPinnedRow(BillingWebhookService):
    def __init__(self, *, row: SimpleNamespace, sync_impl) -> None:
        super().__init__(_FakeDb(), stripe_client=SimpleNamespace())
        self._row = row
        self._sync = sync_impl

    async def _get_or_create_event_row(self, **_kwargs):
        return self._row


class _SyncRaise:
    async def sync_from_checkout_session(self, **_kwargs):
        raise RuntimeError("boom")

    async def sync_subscription_payload(self, **_kwargs):
        raise RuntimeError("boom")

    async def mark_subscription_ended(self, **_kwargs):
        raise RuntimeError("boom")

    async def apply_invoice_event(self, **_kwargs):
        raise RuntimeError("boom")


class _SyncSuccess:
    async def sync_from_checkout_session(self, **_kwargs):
        return None

    async def sync_subscription_payload(self, **_kwargs):
        return SimpleNamespace(user_id="u1", stripe_customer_id="cus_1")

    async def mark_subscription_ended(self, **_kwargs):
        return None

    async def apply_invoice_event(self, **_kwargs):
        return None


async def test_failed_event_is_not_marked_processed_and_can_be_retried() -> None:
    row = SimpleNamespace(
        processed_at=None,
        failed_at=None,
        processing_error=None,
        user_id=None,
        stripe_customer_id=None,
        event_type="unknown",
        livemode=False,
        payload={},
    )
    service = _ServiceWithPinnedRow(row=row, sync_impl=_SyncRaise())

    event = {
        "id": "evt_retry",
        "type": "customer.subscription.updated",
        "livemode": False,
        "data": {"object": {"id": "sub_1", "customer": "cus_1"}},
    }

    try:
        await service.process_event(event=event)
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass

    assert row.processed_at is None
    assert row.failed_at is not None
    assert "RuntimeError" in str(row.processing_error)

    service._sync = _SyncSuccess()
    result = await service.process_event(event=event)

    assert result["status"] == "processed"
    assert result["duplicate"] is False
    assert row.processed_at is not None
    assert row.failed_at is None
    assert row.processing_error is None


async def test_processed_event_returns_duplicate_without_reprocessing() -> None:
    row = SimpleNamespace(
        processed_at=datetime.now(UTC),
        failed_at=None,
        processing_error=None,
        user_id=None,
        stripe_customer_id=None,
        event_type="invoice.payment_succeeded",
        livemode=False,
        payload={},
    )
    service = _ServiceWithPinnedRow(row=row, sync_impl=_SyncRaise())

    event = {
        "id": "evt_done",
        "type": "invoice.payment_succeeded",
        "livemode": False,
        "data": {"object": {"id": "in_1", "customer": "cus_1"}},
    }

    result = await service.process_event(event=event)
    assert result == {
        "event_id": "evt_done",
        "event_type": "invoice.payment_succeeded",
        "status": "duplicate",
        "duplicate": True,
    }
