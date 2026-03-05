"""Stripe webhook processing entrypoint for billing sync."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from packages.domain.billing.subscription_sync_service import SubscriptionSyncService
from packages.infra.db.models.billing_webhook_event import BillingWebhookEvent
from packages.infra.providers.stripe.client import (
    StripeClient,
    StripeClientConfigError,
    StripeWebhookSignatureError,
)


class BillingWebhookService:
    """Verify, dedupe, and process Stripe webhook events."""

    def __init__(self, db: AsyncSession, *, stripe_client: StripeClient) -> None:
        self._db = db
        self._stripe = stripe_client
        self._sync = SubscriptionSyncService(db, stripe_client=stripe_client)

    async def process_stripe_webhook(
        self,
        *,
        payload: bytes,
        signature_header: str | None,
    ) -> dict[str, Any]:
        event = self._stripe.construct_event(
            payload=payload,
            signature_header=signature_header,
        )
        return await self.process_event(event=event)

    async def process_event(self, *, event: dict[str, Any]) -> dict[str, Any]:
        event_id = str(event.get("id") or "").strip()
        event_type = str(event.get("type") or "unknown").strip() or "unknown"
        livemode = bool(event.get("livemode"))
        data_raw = event.get("data")
        data = dict(data_raw) if isinstance(data_raw, dict) else {}
        object_raw = data.get("object")
        object_payload = dict(object_raw) if isinstance(object_raw, dict) else {}

        if not event_id:
            raise ValueError("Missing Stripe event id.")

        persisted = await self._db.scalar(
            select(BillingWebhookEvent).where(BillingWebhookEvent.stripe_event_id == event_id)
        )
        if persisted is not None and persisted.processed_at is not None:
            return {
                "event_id": event_id,
                "event_type": event_type,
                "status": "duplicate",
                "duplicate": True,
            }

        if persisted is None:
            persisted = BillingWebhookEvent(
                stripe_event_id=event_id,
                event_type=event_type,
                stripe_customer_id=_extract_stripe_customer_id(object_payload),
                livemode=livemode,
                payload=event,
                received_at=datetime.now(UTC),
            )
            self._db.add(persisted)
            await self._db.flush()

        subscription = None
        try:
            if event_type == "checkout.session.completed":
                subscription = await self._sync.sync_from_checkout_session(
                    checkout_session=object_payload,
                    latest_event_id=event_id,
                )
            elif event_type in {
                "customer.subscription.created",
                "customer.subscription.updated",
                "customer.subscription.resumed",
                "customer.subscription.paused",
                "customer.subscription.pending_update_applied",
                "customer.subscription.pending_update_expired",
                "customer.subscription.trial_will_end",
            }:
                subscription = await self._sync.sync_subscription_payload(
                    subscription_payload=object_payload,
                    latest_event_id=event_id,
                )
            elif event_type == "customer.subscription.deleted":
                subscription_id = str(object_payload.get("id") or "").strip()
                if subscription_id:
                    subscription = await self._sync.mark_subscription_ended(
                        stripe_subscription_id=subscription_id,
                        ended_at=datetime.now(UTC),
                        latest_event_id=event_id,
                    )
            elif event_type in {
                "invoice.payment_succeeded",
                "invoice.payment_failed",
                "invoice.finalized",
            }:
                subscription = await self._sync.apply_invoice_event(
                    invoice_payload=object_payload,
                    latest_event_id=event_id,
                )

            if subscription is not None:
                persisted.user_id = subscription.user_id
                persisted.stripe_customer_id = subscription.stripe_customer_id

            persisted.processing_error = None
            persisted.processed_at = datetime.now(UTC)
            await self._db.flush()
            return {
                "event_id": event_id,
                "event_type": event_type,
                "status": "processed",
                "duplicate": False,
            }
        except Exception as exc:  # noqa: BLE001
            persisted.processing_error = f"{type(exc).__name__}: {exc}"
            persisted.processed_at = datetime.now(UTC)
            await self._db.flush()
            raise


__all__ = [
    "BillingWebhookService",
    "StripeClientConfigError",
    "StripeWebhookSignatureError",
]


def _extract_stripe_customer_id(payload: dict[str, Any]) -> str | None:
    value = payload.get("customer")
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, dict):
        text = str(value.get("id") or "").strip()
        return text or None
    return None
