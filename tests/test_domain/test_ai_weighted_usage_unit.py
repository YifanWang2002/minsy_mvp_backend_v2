"""Unit tests for weighted AI usage conversion."""

from __future__ import annotations

from packages.domain.billing.usage_service import compute_weighted_ai_usage_from_openai


def test_weighted_ai_usage_respects_input_output_price_gap() -> None:
    usage = compute_weighted_ai_usage_from_openai(
        raw_usage={
            "input_tokens": 1_000,
            "output_tokens": 500,
            "output_tokens_details": {"reasoning_tokens": 100},
        },
        model="gpt-5.2",
        pricing={
            "gpt-5.2": {
                "input_per_token": 0.002,
                "output_per_token": 0.004,
            }
        },
        billing_cost_model={"token_cost_per_1k_usd": 1.0},
    )

    # Raw = 1000 input + 500 output (thinking is billed together with output) = 1500
    assert usage.raw_total_tokens == 1_500
    # Cost = 1000*0.002 + 500*0.004 = 4.0, unit=1.0/1000=0.001 => 4000
    assert usage.weighted_total_tokens == 4_000
    assert usage.estimated_cost_usd == 4.0
    assert usage.reasoning_tokens == 0
    assert usage.output_tokens == 500


def test_weighted_ai_usage_falls_back_to_raw_when_pricing_missing() -> None:
    usage = compute_weighted_ai_usage_from_openai(
        raw_usage={"input_tokens": 120, "output_tokens": 30},
        model="unknown-model",
        pricing={},
        billing_cost_model={"token_cost_per_1k_usd": 0.01},
    )

    assert usage.raw_total_tokens == 150
    assert usage.weighted_total_tokens == 150
    # fallback path uses raw_total * usd_per_internal_token
    assert usage.estimated_cost_usd > 0


def test_weighted_ai_usage_supports_explicit_internal_unit_override() -> None:
    usage = compute_weighted_ai_usage_from_openai(
        raw_usage={"input_tokens": 10, "output_tokens": 10},
        model="gpt-5.2",
        pricing={"gpt-5.2": {"input_per_token": 0.0005, "output_per_token": 0.0005}},
        billing_cost_model={"ai_usage_unit_usd": 0.00025},
    )

    # total cost = 20 * 0.0005 = 0.01 ; unit = 0.00025 => 40
    assert usage.weighted_total_tokens == 40
    assert usage.usd_per_internal_token == 0.00025


def test_weighted_ai_usage_resolves_versioned_model_name_to_base_pricing() -> None:
    usage = compute_weighted_ai_usage_from_openai(
        raw_usage={"input_tokens": 100, "output_tokens": 50},
        model="gpt-5.2-2025-12-11",
        pricing={
            "default": {"input_per_token": 0.0001, "output_per_token": 0.0001},
            "gpt-5.2": {"input_per_token": 0.001, "output_per_token": 0.002},
        },
        billing_cost_model={"ai_usage_unit_usd": 0.0005},
    )

    assert usage.input_per_token_usd == 0.001
    assert usage.output_per_token_usd == 0.002
    assert usage.estimated_cost_usd == 0.2
    assert usage.weighted_total_tokens == 400
