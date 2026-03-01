from __future__ import annotations

from apps.api.agents.phases import Phase
from apps.api.orchestration.constants import (
    _STRATEGY_STAGE_ARTIFACT_OPS,
    _STRATEGY_STAGE_SCHEMA_ONLY,
    resolve_phase_max_output_tokens,
)
from packages.domain.session.services.openai_stream_service import _build_stream_kwargs


def test_000_resolve_phase_max_output_tokens_prefers_strategy_stage() -> None:
    assert (
        resolve_phase_max_output_tokens(
            phase=Phase.STRATEGY.value,
            stage=_STRATEGY_STAGE_SCHEMA_ONLY,
        )
        == 5000
    )
    assert (
        resolve_phase_max_output_tokens(
            phase=Phase.STRATEGY.value,
            stage=_STRATEGY_STAGE_ARTIFACT_OPS,
        )
        == 4200
    )


def test_010_resolve_phase_max_output_tokens_uses_phase_default() -> None:
    assert resolve_phase_max_output_tokens(phase=Phase.KYC.value, stage=None) == 900
    assert resolve_phase_max_output_tokens(phase="unknown", stage=None) == 1200


def test_020_stream_kwargs_include_output_budget_when_provided() -> None:
    kwargs = _build_stream_kwargs(
        model="gpt-5.2",
        input_text="hello",
        instructions="rule",
        max_output_tokens=1234,
        previous_response_id=None,
        tools=None,
        tool_choice=None,
        reasoning=None,
    )
    assert kwargs["max_output_tokens"] == 1234

    no_budget_kwargs = _build_stream_kwargs(
        model="gpt-5.2",
        input_text="hello",
        instructions="rule",
        max_output_tokens=None,
        previous_response_id=None,
        tools=None,
        tool_choice=None,
        reasoning=None,
    )
    assert "max_output_tokens" not in no_budget_kwargs
