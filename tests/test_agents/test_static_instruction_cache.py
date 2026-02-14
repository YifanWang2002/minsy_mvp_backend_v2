from __future__ import annotations

from src.agents.skills.deployment_skills import build_deployment_static_instructions
from src.agents.skills.kyc_skills import build_kyc_static_instructions
from src.agents.skills.pre_strategy_skills import build_pre_strategy_static_instructions
from src.agents.skills.strategy_skills import build_strategy_static_instructions
from src.agents.skills.stress_test_skills import build_stress_test_static_instructions


def test_kyc_static_instruction_cache_hits_same_key() -> None:
    first = build_kyc_static_instructions(language="zh", phase_stage=None)
    second = build_kyc_static_instructions(language="zh", phase_stage=None)
    assert first is second


def test_pre_strategy_static_instruction_cache_hits_same_key() -> None:
    first = build_pre_strategy_static_instructions(language="en", phase_stage=None)
    second = build_pre_strategy_static_instructions(language="en", phase_stage=None)
    assert first is second


def test_strategy_static_instruction_cache_distinguishes_stage() -> None:
    schema_only = build_strategy_static_instructions(language="en", phase_stage="schema_only")
    schema_only_again = build_strategy_static_instructions(language="en", phase_stage="schema_only")
    artifact_ops = build_strategy_static_instructions(language="en", phase_stage="artifact_ops")

    assert schema_only is schema_only_again
    assert schema_only is not artifact_ops


def test_stress_test_static_instruction_cache_distinguishes_stage() -> None:
    bootstrap = build_stress_test_static_instructions(language="en", phase_stage="bootstrap")
    bootstrap_again = build_stress_test_static_instructions(language="en", phase_stage="bootstrap")
    feedback = build_stress_test_static_instructions(language="en", phase_stage="feedback")

    assert bootstrap is bootstrap_again
    assert bootstrap is not feedback


def test_deployment_static_instruction_cache_distinguishes_language() -> None:
    english = build_deployment_static_instructions(language="en")
    english_again = build_deployment_static_instructions(language="en")
    chinese = build_deployment_static_instructions(language="zh")

    assert english is english_again
    assert english is not chinese
