"""Unit tests for unified billing pricing settings."""

from __future__ import annotations

from packages.shared_settings.schema.settings import settings


def test_unified_openai_pricing_overrides_legacy(monkeypatch) -> None:
    monkeypatch.setattr(
        settings,
        "billing_pricing_json",
        {
            "openai_pricing": {
                "gpt-5.2": {
                    "input_per_1k_tokens": 0.00175,
                    "output_per_1k_tokens": 0.014,
                }
            }
        },
    )
    monkeypatch.setattr(
        settings,
        "openai_pricing_json",
        {"legacy": {"input_per_1k_tokens": 99}},
    )

    pricing = settings.openai_pricing

    assert "gpt-5.2" in pricing
    assert "legacy" not in pricing
    assert pricing["gpt-5.2"]["input_per_1k_tokens"] == 0.00175


def test_unified_tier_limits_and_legacy_cpu_key_compat(monkeypatch) -> None:
    monkeypatch.setattr(
        settings,
        "billing_pricing_json",
        {
            "tier_limits": {
                "plus": {"cpu_tokens_monthly_total": 200},
            }
        },
    )
    monkeypatch.setattr(settings, "billing_config_json", {})

    limits = settings.billing_tier_limits

    assert limits["plus"]["cpu_tokens_monthly_total"] == 200

    monkeypatch.setattr(settings, "billing_pricing_json", {})
    monkeypatch.setattr(
        settings,
        "billing_config_json",
        {"tier_limits": {"plus": {"cpu_jobs_monthly_total": 111}}},
    )
    compat_limits = settings.billing_tier_limits
    assert compat_limits["plus"]["cpu_tokens_monthly_total"] == 111


def test_unified_cost_model_overrides_legacy(monkeypatch) -> None:
    monkeypatch.setattr(
        settings,
        "billing_pricing_json",
        {
            "cost_model": {
                "ai_usage_unit_usd": 0.00002,
                "cpu_bars_per_token": 60000,
            }
        },
    )
    monkeypatch.setattr(
        settings,
        "billing_cost_model_json",
        {"ai_usage_unit_usd": 0.0002, "cpu_bars_per_token": 999},
    )

    model = settings.billing_cost_model

    assert model["ai_usage_unit_usd"] == 0.00002
    assert model["cpu_bars_per_token"] == 60000


def test_defaults_include_go_tier_and_price(monkeypatch) -> None:
    monkeypatch.setattr(settings, "billing_pricing_json", {})
    monkeypatch.setattr(settings, "billing_config_json", {})
    monkeypatch.setattr(settings, "billing_cost_model_json", {})

    limits = settings.billing_tier_limits
    model = settings.billing_cost_model

    assert "go" in limits
    assert limits["go"]["ai_tokens_monthly_total"] > limits["free"]["ai_tokens_monthly_total"]
    assert model["go_price_usd"] == 8.0


def test_stripe_go_price_and_product_map(monkeypatch) -> None:
    monkeypatch.setattr(settings, "stripe_price_go_monthly", "price_go")
    monkeypatch.setattr(settings, "stripe_price_plus_monthly", "price_plus")
    monkeypatch.setattr(settings, "stripe_price_pro_monthly", "price_pro")
    monkeypatch.setattr(settings, "stripe_product_go", "prod_go")

    assert settings.stripe_price_to_tier_map == {
        "price_go": "go",
        "price_plus": "plus",
        "price_pro": "pro",
    }
    assert settings.stripe_product_to_tier_map == {"prod_go": "go"}
