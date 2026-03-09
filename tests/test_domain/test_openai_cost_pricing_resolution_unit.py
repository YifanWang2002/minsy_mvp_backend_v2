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
