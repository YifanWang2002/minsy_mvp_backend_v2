from __future__ import annotations

from apps.api.agents.skills.kyc_skills import build_kyc_static_instructions
from apps.api.agents.skills.pre_strategy_skills import (
    build_pre_strategy_static_instructions,
)


def test_kyc_skill_contract_uses_compact_session_state_keys() -> None:
    instructions = build_kyc_static_instructions(language="en")

    assert "has_missing` and `next_missing" in instructions
    assert "has_missing_fields" not in instructions
    assert "next_missing_field" not in instructions


def test_pre_strategy_skill_contract_uses_compact_session_state_keys() -> None:
    instructions = build_pre_strategy_static_instructions(language="en")

    assert "has_missing` and `next_missing" in instructions
    assert "allowed_instruments` in `[SESSION STATE]`" in instructions
    assert "mapped_tradingview_symbol` from `[SESSION STATE]`" in instructions
    assert "has_missing_fields" not in instructions
    assert "next_missing_field" not in instructions
    assert "allowed_instruments_for_target_market" not in instructions
    assert "mapped_tradingview_symbol_for_target_instrument" not in instructions
