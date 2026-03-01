from __future__ import annotations

from uuid import UUID

from apps.api.agents.phases import Phase
from apps.api.orchestration.constants import (
    _PRE_STRATEGY_STAGE_INTENT_COLLECTION,
    _STRATEGY_STAGE_ARTIFACT_OPS,
    get_phase_tool_matrix_rows,
)
from apps.api.orchestration.prompt_builder import PromptBuilderMixin


class _DummyPromptBuilder(PromptBuilderMixin):
    @staticmethod
    def _coerce_uuid_text(value: object) -> str | None:
        if not isinstance(value, str):
            return None
        text = value.strip()
        if not text:
            return None
        try:
            return str(UUID(text))
        except ValueError:
            return None


def test_000_market_data_matrix_pre_strategy_includes_fetch_workflow_tools() -> None:
    rows = get_phase_tool_matrix_rows(
        phase=Phase.PRE_STRATEGY.value,
        stage=_PRE_STRATEGY_STAGE_INTENT_COLLECTION,
        reachable_only=True,
    )
    market_rows = [row for row in rows if row.server_label == "market_data"]
    allowed = {
        tool_name
        for row in market_rows
        for tool_name in row.allowed_tools
    }

    assert "check_symbol_available" in allowed
    assert "get_symbol_data_coverage" in allowed
    assert "market_data_detect_missing_ranges" in allowed
    assert "market_data_fetch_missing_ranges" in allowed
    assert "market_data_get_sync_job" in allowed


def test_010_strategy_artifact_ops_matrix_exposes_enriched_market_tools() -> None:
    rows = get_phase_tool_matrix_rows(
        phase=Phase.STRATEGY.value,
        stage=_STRATEGY_STAGE_ARTIFACT_OPS,
        reachable_only=True,
    )
    market_rows = [row for row in rows if row.server_label == "market_data"]
    allowed = {
        tool_name
        for row in market_rows
        for tool_name in row.allowed_tools
    }
    assert "get_symbol_data_coverage" in allowed
    assert "market_data_detect_missing_ranges" in allowed
    assert "market_data_fetch_missing_ranges" in allowed
    assert "market_data_get_sync_job" in allowed
    assert "get_symbol_metadata" in allowed


def test_020_runtime_policy_uses_matrix_stage_for_strategy_artifact_ops() -> None:
    builder = _DummyPromptBuilder()
    policy = builder._build_phase_runtime_policy(
        phase=Phase.STRATEGY.value,
        artifacts={
            Phase.STRATEGY.value: {
                "profile": {
                    "strategy_id": "00000000-0000-4000-8000-000000000001",
                }
            }
        },
    )

    assert policy.phase_stage == _STRATEGY_STAGE_ARTIFACT_OPS
    assert policy.tool_mode == "replace"
    assert policy.allowed_tools is not None
    market_tool = next(
        item for item in policy.allowed_tools if item.get("server_label") == "market_data"
    )
    assert "market_data_fetch_missing_ranges" in tuple(market_tool["allowed_tools"])
    assert "get_symbol_metadata" in tuple(market_tool["allowed_tools"])


def test_030_legacy_market_data_constant_keeps_sync_toolchain() -> None:
    from apps.api.orchestration.constants import _MARKET_DATA_MINIMAL_TOOL_NAMES

    assert _MARKET_DATA_MINIMAL_TOOL_NAMES == (
        "check_symbol_available",
        "get_available_symbols",
        "get_symbol_data_coverage",
        "market_data_detect_missing_ranges",
        "market_data_get_sync_job",
        "get_symbol_metadata",
        "market_data_fetch_missing_ranges",
    )
