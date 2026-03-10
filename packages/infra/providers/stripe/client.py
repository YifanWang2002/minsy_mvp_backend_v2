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
        trial_days: int | None = None,
        promotion_code_id: str | None = None,
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
            "payment_method_collection": "always",
            "metadata": metadata or {},
        }
        resolved_trial_days = max(int(trial_days or 0), 0)
        normalized_promotion_code_id = (
            promotion_code_id.strip() if isinstance(promotion_code_id, str) else ""
        )
        if normalized_promotion_code_id:
            payload["discounts"] = [{"promotion_code": normalized_promotion_code_id}]
        if resolved_trial_days > 0:
            payload["subscription_data"] = {
                "trial_period_days": resolved_trial_days,
                "trial_settings": {
                    "end_behavior": {
                        "missing_payment_method": "cancel",
                    },
                },
            }
        created = await asyncio.to_thread(stripe.checkout.Session.create, **payload)
        return self._as_dict(created)

    async def create_billing_portal_session(
        self,
        *,
        customer_id: str,
        return_url: str,
        flow_data: dict[str, Any] | None = None,
        configuration_id: str | None = None,
    ) -> dict[str, Any]:
        self._ensure_configured()
        payload = {
            "customer": customer_id,
            "return_url": return_url,
        }
        if isinstance(flow_data, dict) and flow_data:
            payload["flow_data"] = flow_data
        if isinstance(configuration_id, str) and configuration_id.strip():
            payload["configuration"] = configuration_id.strip()
        created = await asyncio.to_thread(
            stripe.billing_portal.Session.create, **payload
        )
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

    async def list_subscriptions(
        self,
        *,
        customer_id: str,
        status: str = "all",
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        self._ensure_configured()
        payload = {
            "customer": customer_id,
            "status": status,
            "limit": max(1, min(int(limit), 100)),
            "expand": ["data.items.data.price", "data.latest_invoice"],
        }
        listed = await asyncio.to_thread(stripe.Subscription.list, **payload)
        listed_dict = self._as_dict(listed)
        rows = listed_dict.get("data")
        if not isinstance(rows, list):
            return []
        normalized: list[dict[str, Any]] = []
        for row in rows:
            if isinstance(row, dict):
                normalized.append(dict(row))
        return normalized

    async def update_subscription_price(
        self,
        stripe_subscription_id: str,
        *,
        subscription_item_id: str,
        price_id: str,
        proration_behavior: str,
        payment_behavior: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        self._ensure_configured()
        normalized_payment_behavior = (
            payment_behavior.strip() if isinstance(payment_behavior, str) else ""
        )
        payload: dict[str, Any] = {
            "items": [
                {
                    "id": subscription_item_id,
                    "price": price_id,
                    "quantity": 1,
                }
            ],
            "proration_behavior": proration_behavior,
            "expand": ["items.data.price", "latest_invoice", "pending_update"],
        }
        if normalized_payment_behavior:
            payload["payment_behavior"] = normalized_payment_behavior
        # Stripe rejects `pending_if_incomplete` when `cancel_at_period_end` is present.
        if normalized_payment_behavior.lower() != "pending_if_incomplete":
            payload["cancel_at_period_end"] = False
        if metadata is not None:
            payload["metadata"] = metadata
        updated = await asyncio.to_thread(
            stripe.Subscription.modify,
            stripe_subscription_id,
            **payload,
        )
        return self._as_dict(updated)

    async def retrieve_customer(
        self,
        stripe_customer_id: str,
    ) -> dict[str, Any]:
        self._ensure_configured()
        retrieved = await asyncio.to_thread(
            stripe.Customer.retrieve, stripe_customer_id
        )
        return self._as_dict(retrieved)

    def construct_event(
        self,
        *,
        payload: bytes,
        signature_header: str | None,
        webhook_secret: str | None = None,
    ) -> dict[str, Any]:
        self._ensure_configured()
        resolved_webhook_secret = (
            webhook_secret or settings.stripe_webhook_secret
        ).strip()
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
