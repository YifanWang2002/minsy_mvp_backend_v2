from __future__ import annotations

from datetime import UTC, datetime

from src.observability.openai_cost import (
    build_turn_usage_snapshot,
    merge_session_openai_cost_metadata,
    normalize_openai_usage,
    read_session_openai_cost_totals,
)


def test_normalize_openai_usage_accepts_prompt_completion_shape() -> None:
    normalized = normalize_openai_usage(
        {
            "prompt_tokens": 123,
            "completion_tokens": 45,
        }
    )

    assert normalized == {
        "input_tokens": 123,
        "output_tokens": 45,
        "total_tokens": 168,
    }


def test_build_turn_usage_snapshot_includes_model_and_cost() -> None:
    snapshot = build_turn_usage_snapshot(
        raw_usage={"input_tokens": 1000, "output_tokens": 200, "total_tokens": 1200},
        model="gpt-5.2",
        response_id="resp_abc",
        at=datetime(2026, 2, 20, 12, 34, 56, tzinfo=UTC),
        pricing={
            "gpt-5.2": {
                "input_per_token": 0.00000125,
                "output_per_token": 0.00001,
            }
        },
        cost_tracking_enabled=True,
    )

    assert snapshot is not None
    assert snapshot["model"] == "gpt-5.2"
    assert snapshot["response_id"] == "resp_abc"
    assert snapshot["input_tokens"] == 1000
    assert snapshot["output_tokens"] == 200
    assert snapshot["total_tokens"] == 1200
    assert snapshot["cost_usd"] == 0.00325
    assert snapshot["at"] == "2026-02-20T12:34:56Z"


def test_merge_session_openai_cost_metadata_accumulates_totals_and_by_model() -> None:
    metadata: dict[str, object] = {}
    first_usage = {
        "model": "gpt-5.2",
        "response_id": "resp_1",
        "input_tokens": 100,
        "output_tokens": 40,
        "total_tokens": 140,
        "cost_usd": 0.010000,
        "at": "2026-02-20T01:00:00Z",
    }
    second_usage = {
        "model": "gpt-5.2",
        "response_id": "resp_2",
        "input_tokens": 50,
        "output_tokens": 20,
        "total_tokens": 70,
        "cost_usd": 0.005000,
        "at": "2026-02-20T01:05:00Z",
    }

    metadata, totals = merge_session_openai_cost_metadata(metadata, first_usage)
    metadata, totals = merge_session_openai_cost_metadata(metadata, second_usage)
    totals_from_reader = read_session_openai_cost_totals(metadata)

    assert totals == {
        "turn_count": 2,
        "input_tokens": 150,
        "output_tokens": 60,
        "total_tokens": 210,
        "cost_usd": 0.015,
    }
    assert totals_from_reader == totals

    openai_cost = metadata["openai_cost"]
    assert isinstance(openai_cost, dict)
    by_model = openai_cost["by_model"]
    assert isinstance(by_model, dict)
    model_totals = by_model["gpt-5.2"]
    assert model_totals["turn_count"] == 2
    assert model_totals["cost_usd"] == 0.015
    assert openai_cost["last_turn"]["response_id"] == "resp_2"

