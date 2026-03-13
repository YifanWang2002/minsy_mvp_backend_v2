"""Unit tests for OpenAI cost snapshot pricing resolution."""

from __future__ import annotations

from packages.infra.observability.openai_cost import build_turn_usage_snapshot


def test_turn_usage_snapshot_uses_base_model_pricing_for_versioned_model() -> None:
    snapshot = build_turn_usage_snapshot(
        raw_usage={"input_tokens": 100, "output_tokens": 50},
        model="gpt-5.2-2025-12-11",
        response_id="resp_123",
        pricing={
            "default": {"input_per_token": 0.0001, "output_per_token": 0.0001},
            "gpt-5.2": {"input_per_token": 0.001, "output_per_token": 0.002},
        },
        cost_tracking_enabled=True,
    )

    assert snapshot is not None
    assert snapshot["model"] == "gpt-5.2-2025-12-11"
    assert snapshot["cost_usd"] == 0.2


def test_turn_usage_snapshot_falls_back_to_default_when_base_model_missing() -> None:
    snapshot = build_turn_usage_snapshot(
        raw_usage={"input_tokens": 100, "output_tokens": 50},
        model="unknown-model-2026-01-01",
        response_id="resp_456",
        pricing={"default": {"input_per_token": 0.0001, "output_per_token": 0.0002}},
        cost_tracking_enabled=True,
    )

    assert snapshot is not None
    assert snapshot["cost_usd"] == 0.02


def test_turn_usage_snapshot_includes_reasoning_cached_and_cost_breakdown() -> None:
    snapshot = build_turn_usage_snapshot(
        raw_usage={
            "input_tokens": 200,
            "output_tokens": 80,
            "input_tokens_details": {"cached_tokens": 120},
            "output_tokens_details": {"reasoning_tokens": 30},
        },
        model="gpt-5.4-2026-03-01",
        reasoning_effort="high",
        response_id="resp_789",
        pricing={"gpt-5.4": {"input_per_token": 0.001, "output_per_token": 0.002}},
        cost_tracking_enabled=True,
    )

    assert snapshot is not None
    assert snapshot["resolved_model"] == "gpt-5.4-2026-03-01"
    assert snapshot["reasoning_effort"] == "high"
    assert snapshot["cached_input_tokens"] == 120
    assert snapshot["reasoning_tokens"] == 30
    assert snapshot["cost_breakdown"]["cached_input_pricing_mode"] == "raw_input_equivalent"
