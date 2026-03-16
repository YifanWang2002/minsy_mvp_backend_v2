from __future__ import annotations

from apps.api.agents.skills.kyc_skills import build_kyc_static_instructions
from apps.api.agents.skills.pre_strategy_skills import (
    build_pre_strategy_static_instructions,
)


def test_kyc_skill_contract_uses_compact_session_state_keys() -> None:
    instructions = build_kyc_static_instructions(language="en")

    assert "has_missing_fields" in instructions
    assert "next_missing_field" in instructions


def test_pre_strategy_skill_contract_uses_compact_session_state_keys() -> None:
    instructions = build_pre_strategy_static_instructions(language="en")

    assert "has_missing_fields" in instructions
    assert "next_missing_field" in instructions
    assert "allowed_instruments_for_target_market" in instructions
    assert "mapped_tradingview_symbol_for_target_instrument" in instructions
    assert "strategy_family_choice" in instructions


def test_pre_strategy_skill_contract_requires_regime_tool_before_family_recommendation() -> None:
    instructions = build_pre_strategy_static_instructions(language="en")

    assert "MUST call `pre_strategy_get_regime_snapshot` before any family recommendation" in instructions
    assert "do NOT:" in instructions
    assert "rank/recommend strategy families" in instructions
