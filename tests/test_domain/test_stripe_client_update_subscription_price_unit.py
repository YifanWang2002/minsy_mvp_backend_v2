"""Unit coverage for Stripe subscription price update payload composition."""

from __future__ import annotations

import stripe

from packages.infra.providers.stripe.client import StripeClient


async def test_update_subscription_price_omits_cancel_at_period_end_for_pending_if_incomplete(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_modify(subscription_id: str, **kwargs):  # noqa: ANN003
        captured["subscription_id"] = subscription_id
        captured["kwargs"] = kwargs
        return {"id": subscription_id, "status": "active"}

    monkeypatch.setattr(stripe.Subscription, "modify", _fake_modify)

    client = StripeClient(api_key="sk_test_unit")
    await client.update_subscription_price(
        "sub_upgrade",
        subscription_item_id="si_123",
        price_id="price_plus",
        proration_behavior="always_invoice",
        payment_behavior="pending_if_incomplete",
        metadata=None,
    )

    kwargs = dict(captured["kwargs"])
    assert kwargs["payment_behavior"] == "pending_if_incomplete"
    assert "cancel_at_period_end" not in kwargs


async def test_update_subscription_price_keeps_cancel_at_period_end_for_non_pending_behavior(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_modify(subscription_id: str, **kwargs):  # noqa: ANN003
        captured["subscription_id"] = subscription_id
        captured["kwargs"] = kwargs
        return {"id": subscription_id, "status": "active"}

    monkeypatch.setattr(stripe.Subscription, "modify", _fake_modify)

    client = StripeClient(api_key="sk_test_unit")
    await client.update_subscription_price(
        "sub_downgrade",
        subscription_item_id="si_456",
        price_id="price_go",
        proration_behavior="none",
        payment_behavior="allow_incomplete",
        metadata={"pending_tier": "go"},
    )

    kwargs = dict(captured["kwargs"])
    assert kwargs["payment_behavior"] == "allow_incomplete"
    assert kwargs["cancel_at_period_end"] is False
    assert kwargs["metadata"] == {"pending_tier": "go"}
