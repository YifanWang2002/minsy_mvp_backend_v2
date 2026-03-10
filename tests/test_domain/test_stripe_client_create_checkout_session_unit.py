"""Unit coverage for Stripe checkout session payload composition."""

from __future__ import annotations

import stripe

from packages.infra.providers.stripe.client import StripeClient


async def test_create_checkout_session_uses_allow_promotion_codes_when_no_preapplied_code(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_create(**kwargs):  # noqa: ANN003
        captured["kwargs"] = kwargs
        return {"id": "cs_test_plain", "url": "https://checkout.stripe.test/plain"}

    monkeypatch.setattr(stripe.checkout.Session, "create", _fake_create)

    client = StripeClient(api_key="sk_test_unit")
    await client.create_checkout_session(
        customer_id="cus_plain",
        price_id="price_go",
        success_url="https://app.example.com/success",
        cancel_url="https://app.example.com/cancel",
        client_reference_id="user_plain",
        metadata={"user_id": "user_plain"},
        trial_days=7,
        promotion_code_id=None,
    )

    kwargs = dict(captured["kwargs"])
    assert kwargs["allow_promotion_codes"] is True
    assert "discounts" not in kwargs


async def test_create_checkout_session_uses_discounts_and_omits_allow_promotion_codes(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_create(**kwargs):  # noqa: ANN003
        captured["kwargs"] = kwargs
        return {"id": "cs_test_discount", "url": "https://checkout.stripe.test/discount"}

    monkeypatch.setattr(stripe.checkout.Session, "create", _fake_create)

    client = StripeClient(api_key="sk_test_unit")
    await client.create_checkout_session(
        customer_id="cus_discount",
        price_id="price_plus",
        success_url="https://app.example.com/success",
        cancel_url="https://app.example.com/cancel",
        client_reference_id="user_discount",
        metadata={"user_id": "user_discount"},
        trial_days=7,
        promotion_code_id="promo_123",
    )

    kwargs = dict(captured["kwargs"])
    assert kwargs["discounts"] == [{"promotion_code": "promo_123"}]
    assert "allow_promotion_codes" not in kwargs
