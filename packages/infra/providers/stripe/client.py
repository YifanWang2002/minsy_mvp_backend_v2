"""Stripe SDK wrapper used by billing services/routes."""

from __future__ import annotations

import asyncio
from typing import Any

import stripe

from packages.shared_settings.schema.settings import settings


class StripeClientConfigError(RuntimeError):
    """Raised when Stripe is called without required configuration."""


class StripeWebhookSignatureError(ValueError):
    """Raised when webhook payload/signature validation fails."""


class StripeClient:
    """Thin async wrapper around the Stripe Python SDK."""

    def __init__(self, *, api_key: str | None = None) -> None:
        key = (api_key or settings.stripe_secret_key).strip()
        self._api_key = key
        if key:
            stripe.api_key = key

    @property
    def is_configured(self) -> bool:
        return bool(self._api_key)

    def _ensure_configured(self) -> None:
        if not self.is_configured:
            raise StripeClientConfigError(
                "Stripe secret key is not configured (STRIPE_SECRET_KEY).",
            )

    @staticmethod
    def _as_dict(payload: Any) -> dict[str, Any]:
        if payload is None:
            return {}
        if isinstance(payload, dict):
            return dict(payload)
        if hasattr(payload, "to_dict_recursive"):
            try:
                converted = payload.to_dict_recursive()
            except Exception:  # noqa: BLE001
                converted = {}
            if isinstance(converted, dict):
                return dict(converted)
        return {}

    async def create_customer(
        self,
        *,
        email: str | None,
        metadata: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        self._ensure_configured()
        payload = {
            "email": email,
            "metadata": metadata or {},
        }
        created = await asyncio.to_thread(stripe.Customer.create, **payload)
        return self._as_dict(created)

    async def create_checkout_session(
        self,
        *,
        customer_id: str,
        price_id: str,
        success_url: str,
        cancel_url: str,
        client_reference_id: str,
        metadata: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        self._ensure_configured()
        payload = {
            "mode": "subscription",
            "customer": customer_id,
            "line_items": [{"price": price_id, "quantity": 1}],
            "success_url": success_url,
            "cancel_url": cancel_url,
            "client_reference_id": client_reference_id,
            "allow_promotion_codes": True,
            "metadata": metadata or {},
        }
        created = await asyncio.to_thread(stripe.checkout.Session.create, **payload)
        return self._as_dict(created)

    async def create_billing_portal_session(
        self,
        *,
        customer_id: str,
        return_url: str,
    ) -> dict[str, Any]:
        self._ensure_configured()
        payload = {
            "customer": customer_id,
            "return_url": return_url,
        }
        created = await asyncio.to_thread(stripe.billing_portal.Session.create, **payload)
        return self._as_dict(created)

    async def retrieve_subscription(
        self,
        stripe_subscription_id: str,
    ) -> dict[str, Any]:
        self._ensure_configured()
        payload = {
            "expand": ["items.data.price", "latest_invoice"],
        }
        retrieved = await asyncio.to_thread(
            stripe.Subscription.retrieve,
            stripe_subscription_id,
            **payload,
        )
        return self._as_dict(retrieved)

    async def retrieve_customer(
        self,
        stripe_customer_id: str,
    ) -> dict[str, Any]:
        self._ensure_configured()
        retrieved = await asyncio.to_thread(stripe.Customer.retrieve, stripe_customer_id)
        return self._as_dict(retrieved)

    def construct_event(
        self,
        *,
        payload: bytes,
        signature_header: str | None,
        webhook_secret: str | None = None,
    ) -> dict[str, Any]:
        self._ensure_configured()
        resolved_webhook_secret = (webhook_secret or settings.stripe_webhook_secret).strip()
        if not resolved_webhook_secret:
            raise StripeClientConfigError(
                "Stripe webhook secret is not configured (STRIPE_WEBHOOK_SECRET).",
            )
        if not isinstance(signature_header, str) or not signature_header.strip():
            raise StripeWebhookSignatureError("Missing Stripe-Signature header.")

        try:
            event = stripe.Webhook.construct_event(
                payload=payload,
                sig_header=signature_header.strip(),
                secret=resolved_webhook_secret,
            )
        except Exception as exc:  # noqa: BLE001
            raise StripeWebhookSignatureError(str(exc)) from exc
        return self._as_dict(event)


stripe_client = StripeClient()
