"""Unit tests for CPU token conversion from backtest bars."""

from __future__ import annotations

from packages.domain.billing.usage_service import compute_cpu_tokens_from_bars


def test_cpu_tokens_round_up_from_bars() -> None:
    usage = compute_cpu_tokens_from_bars(
        estimated_bars=250_001,
        billing_cost_model={"cpu_bars_per_token": 100_000},
    )

    assert usage.estimated_bars == 250_001
    assert usage.bars_per_token == 100_000
    assert usage.token_quantity == 3


def test_cpu_tokens_use_default_scale_when_not_configured() -> None:
    usage = compute_cpu_tokens_from_bars(
        estimated_bars=100_000,
        billing_cost_model={},
    )

    assert usage.bars_per_token == 100_000
    assert usage.token_quantity == 1


def test_cpu_tokens_fallback_to_minimum_one_when_estimate_missing() -> None:
    usage = compute_cpu_tokens_from_bars(
        estimated_bars=0,
        billing_cost_model={"cpu_bars_per_token": 50_000},
    )

    assert usage.estimated_bars == 0
    assert usage.token_quantity == 1
