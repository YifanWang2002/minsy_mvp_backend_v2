from __future__ import annotations

import asyncio
import json
from uuid import uuid4

from apps.api.agents.handler_protocol import PhaseContext, RuntimePolicy
from apps.api.agents.handlers.pre_strategy_handler import PreStrategyHandler
from apps.api.agents.phases import Phase
from apps.api.agents.skills.pre_strategy_skills import (
    build_pre_strategy_dynamic_state,
    build_pre_strategy_static_instructions,
)


def test_build_prompt_does_not_prefill_profile_from_current_user_message() -> None:
    handler = PreStrategyHandler()
    ctx = PhaseContext(
        user_id=uuid4(),
        session_artifacts={Phase.PRE_STRATEGY.value: handler.init_artifacts()},
        runtime_policy=RuntimePolicy(),
    )

    prompt = handler.build_prompt(
        ctx,
        "Let's do dogecoin and proceed with the download.",
    )

    phase_data = ctx.session_artifacts[Phase.PRE_STRATEGY.value]
    assert phase_data["profile"] == {}
    assert "- target_market: none" in prompt.enriched_input
    assert "- target_instrument: none" in prompt.enriched_input
    assert "download_consent_detected_this_turn" not in prompt.enriched_input
    assert "symbol_newly_provided_this_turn_hint" not in prompt.enriched_input
    assert "inferred_instrument_from_user_message" not in prompt.enriched_input
    assert [tool.get("server_label") for tool in (prompt.tools or [])] == ["market_data"]


def test_validate_patch_still_accepts_canonical_symbol_formats() -> None:
    handler = PreStrategyHandler()

    validated = handler._validate_patch({"target_instrument": "DOGE/USDT"})

    assert validated["target_instrument"] == "DOGEUSD"
    assert validated["target_market"] == "crypto"


def test_dynamic_state_omits_message_derived_hint_fields() -> None:
    state = build_pre_strategy_dynamic_state(
        missing_fields=["target_market", "target_instrument"],
        collected_fields={},
        kyc_profile={},
    )

    assert "download_consent_detected_this_turn" not in state
    assert "symbol_newly_provided_this_turn_hint" not in state
    assert "inferred_instrument_from_user_message" not in state
    assert "download_requires_user_confirmation: true" in state


def test_static_instructions_delegate_message_inference_to_ai() -> None:
    instructions = build_pre_strategy_static_instructions(language="en")

    assert "current user message" in instructions
    assert "code-provided consent flag" in instructions
    assert "symbol_newly_provided_this_turn_hint" not in instructions
    assert "download_consent_detected_this_turn" not in instructions


def test_post_process_blocks_strategy_transition_when_data_choice_is_pending(monkeypatch) -> None:
    handler = PreStrategyHandler()
    monkeypatch.setattr(
        "apps.api.agents.handlers.pre_strategy_handler._is_symbol_available_locally",
        lambda **kwargs: False,
    )

    ctx = PhaseContext(
        user_id=uuid4(),
        session_artifacts={Phase.PRE_STRATEGY.value: handler.init_artifacts()},
        session_id=uuid4(),
        runtime_policy=RuntimePolicy(),
        turn_context={"mcp_tool_calls": []},
    )

    result = asyncio.run(
        handler.post_process(
            ctx,
            [
                {
                    "target_market": "crypto",
                    "target_instrument": "PEPEUSD",
                    "opportunity_frequency_bucket": "few_per_week",
                    "holding_period_bucket": "swing_days",
                }
            ],
            None,  # type: ignore[arg-type]
        )
    )

    pre_data = result.artifacts[Phase.PRE_STRATEGY.value]
    assert result.completed is False
    assert result.next_phase is None
    assert result.missing_fields == []
    assert pre_data["profile"]["target_instrument"] == "PEPEUSD"
    assert pre_data["runtime"]["instrument_data_status"] == "awaiting_user_choice"


def test_post_process_requires_regime_ready_and_family_choice_before_transition(
    monkeypatch,
) -> None:
    handler = PreStrategyHandler()
    monkeypatch.setattr(
        "apps.api.agents.handlers.pre_strategy_handler._is_symbol_available_locally",
        lambda **kwargs: False,
    )

    turn_context = {
        "mcp_tool_calls": [
            {
                "name": "market_data_fetch_missing_ranges",
                "status": "success",
                "arguments": json.dumps(
                    {
                        "market": "crypto",
                        "symbol": "PEPEUSD",
                    }
                ),
            },
        ]
    }
    ctx = PhaseContext(
        user_id=uuid4(),
        session_artifacts={Phase.PRE_STRATEGY.value: handler.init_artifacts()},
        session_id=uuid4(),
        runtime_policy=RuntimePolicy(),
        turn_context=turn_context,
    )

    result = asyncio.run(
        handler.post_process(
            ctx,
            [
                {
                    "target_market": "crypto",
                    "target_instrument": "PEPEUSD",
                    "opportunity_frequency_bucket": "few_per_week",
                    "holding_period_bucket": "swing_days",
                }
            ],
            None,  # type: ignore[arg-type]
        )
    )

    pre_data = result.artifacts[Phase.PRE_STRATEGY.value]
    assert result.completed is False
    assert result.next_phase is None
    assert pre_data["runtime"]["instrument_data_status"] == "download_started"
    assert pre_data["runtime"]["regime_snapshot_status"] == "pending"


def test_post_process_transitions_after_regime_ready_and_family_choice(monkeypatch) -> None:
    handler = PreStrategyHandler()
    monkeypatch.setattr(
        "apps.api.agents.handlers.pre_strategy_handler._is_symbol_available_locally",
        lambda **kwargs: False,
    )

    regime_output = {
        "ok": True,
        "timeframe_plan": {
            "primary": "1h",
            "secondary": "4h",
            "mapping_reason": "test mapping",
        },
        "primary": {
            "timeframe": "1h",
            "summary": "Regime summary",
            "family_scores": {
                "trend_continuation": 0.62,
                "mean_reversion": 0.21,
                "volatility_regime": 0.17,
                "recommended_family": "trend_continuation",
                "confidence": 0.41,
            },
            "choice_option_subtitles": {
                "trend_continuation": "Recommended: trend evidence stronger.",
                "mean_reversion": "Less preferred: chop not dominant.",
                "volatility_regime": "Less preferred: vol expansion mild.",
            },
        },
        "secondary": {
            "timeframe": "4h",
            "summary": "Secondary summary",
            "family_scores": {
                "trend_continuation": 0.48,
                "mean_reversion": 0.31,
                "volatility_regime": 0.21,
                "recommended_family": "trend_continuation",
                "confidence": 0.17,
            },
            "features": {},
        },
        "snapshot_id": "snapshot-123",
    }
    turn_context = {
        "mcp_tool_calls": [
            {
                "name": "market_data_fetch_missing_ranges",
                "status": "success",
                "arguments": json.dumps(
                    {
                        "market": "crypto",
                        "symbol": "PEPEUSD",
                    }
                ),
            },
            {
                "name": "pre_strategy_get_regime_snapshot",
                "status": "success",
                "arguments": json.dumps(
                    {
                        "market": "crypto",
                        "symbol": "PEPEUSD",
                        "opportunity_frequency_bucket": "few_per_week",
                        "holding_period_bucket": "swing_days",
                    }
                ),
                "output": json.dumps(regime_output),
            },
        ]
    }
    ctx = PhaseContext(
        user_id=uuid4(),
        session_artifacts={Phase.PRE_STRATEGY.value: handler.init_artifacts()},
        session_id=uuid4(),
        runtime_policy=RuntimePolicy(),
        turn_context=turn_context,
    )

    result = asyncio.run(
        handler.post_process(
            ctx,
            [
                {
                    "target_market": "crypto",
                    "target_instrument": "PEPEUSD",
                    "opportunity_frequency_bucket": "few_per_week",
                    "holding_period_bucket": "swing_days",
                    "strategy_family_choice": "trend_continuation",
                }
            ],
            None,  # type: ignore[arg-type]
        )
    )

    pre_data = result.artifacts[Phase.PRE_STRATEGY.value]
    assert result.completed is True
    assert result.next_phase == "strategy"
    assert pre_data["profile"]["strategy_family_choice"] == "trend_continuation"
    assert pre_data["runtime"]["regime_snapshot_status"] == "ready"
    assert pre_data["runtime"]["timeframe_plan"]["primary"] == "1h"
