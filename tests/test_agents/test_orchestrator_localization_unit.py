"""Unit tests for orchestrator-facing localization and strategy prompt profiles."""

from __future__ import annotations

from uuid import uuid4

import pytest

from apps.api.agents.handler_protocol import PhaseContext
from apps.api.agents.handlers.deployment_handler import DeploymentHandler
from apps.api.agents.handlers.kyc_handler import KYCHandler
from apps.api.agents.phases import Phase
from apps.api.agents.skills.strategy_skills import build_strategy_static_instructions
from apps.api.i18n import is_zh_locale, normalize_locale, resolve_user_locale


def test_strategy_prompt_profile_full_bootstrap_includes_spec_and_schema() -> None:
    text = build_strategy_static_instructions(
        language="en",
        phase_stage="schema_only",
        prompt_profile="full_bootstrap",
    )

    assert "[DSL SPEC]" in text
    assert "[DSL JSON SCHEMA]" in text


def test_strategy_prompt_profile_compact_omits_spec_and_schema() -> None:
    text = build_strategy_static_instructions(
        language="en",
        phase_stage="schema_only",
        prompt_profile="compact",
    )

    assert "[DSL SPEC]" not in text
    assert "[DSL JSON SCHEMA]" not in text


def test_strategy_artifact_ops_instructions_include_trade_snapshot_contract() -> None:
    text = build_strategy_static_instructions(
        language="en",
        phase_stage="artifact_ops",
        prompt_profile="compact",
    )

    assert "backtest_trade_snapshots" in text
    assert "## 问题诊断" in text
    assert "## 证据 Bar" in text
    assert "## 修改建议" in text


@pytest.mark.asyncio
async def test_resolve_user_locale_prefers_user_setting_over_fallback() -> None:
    class _Db:
        async def scalar(self, _stmt):
            return "zh"

    resolved = await resolve_user_locale(_Db(), user_id=uuid4(), fallback="en-US")
    assert resolved == "zh"


def test_locale_helpers_normalize_to_supported_values() -> None:
    assert normalize_locale("zh-CN") == "zh"
    assert normalize_locale("en_US") == "en"
    assert normalize_locale("ja", default="zh") == "zh"
    assert is_zh_locale("zh-Hans")
    assert not is_zh_locale("en")


def test_kyc_fallback_choice_prompt_uses_chinese_when_locale_is_zh() -> None:
    handler = KYCHandler()
    ctx = PhaseContext(
        user_id=uuid4(),
        session_artifacts={Phase.KYC.value: handler.init_artifacts()},
        language="zh-CN",
    )
    payload = handler.build_fallback_choice_prompt(
        missing_fields=["risk_tolerance"],
        ctx=ctx,
    )

    assert payload is not None
    assert payload["question"] == "你的风险偏好是什么？"
    assert any("保守" in str(option.get("label", "")) for option in payload["options"])


def test_deployment_fallback_confirmation_prompt_uses_chinese_when_locale_is_zh() -> (
    None
):
    handler = DeploymentHandler()
    ctx = PhaseContext(
        user_id=uuid4(),
        session_artifacts={Phase.DEPLOYMENT.value: handler.init_artifacts()},
        language="zh",
    )
    payload = handler.build_fallback_choice_prompt(
        missing_fields=["deployment_confirmation_status"],
        ctx=ctx,
    )

    assert payload is not None
    assert "确认" in str(payload.get("question", ""))
    option_labels = [
        str(option.get("label", "")) for option in payload.get("options", [])
    ]
    assert "确认部署" in option_labels
