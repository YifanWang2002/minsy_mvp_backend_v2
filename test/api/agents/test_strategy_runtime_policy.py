from __future__ import annotations

from apps.api.agents.phases import Phase
from apps.api.agents.skills.strategy_skills import build_strategy_dynamic_state
from apps.api.orchestration import ChatOrchestrator


def test_strategy_schema_only_runtime_policy_exposes_indicator_and_market_data_tools() -> None:
    orchestrator = ChatOrchestrator(None)  # type: ignore[arg-type]

    policy = orchestrator._build_phase_runtime_policy(
        phase=Phase.STRATEGY.value,
        artifacts={
            Phase.STRATEGY.value: {"profile": {}, "missing_fields": ["strategy_id"]},
        },
    )

    assert policy.phase_stage == "schema_only"
    assert policy.tool_mode == "replace"
    assert policy.allowed_tools is not None

    by_server = {
        tool["server_label"]: set(tool.get("allowed_tools", []))
        for tool in policy.allowed_tools
    }
    assert "strategy" in by_server
    assert "market_data" in by_server
    assert "get_indicator_catalog" in by_server["strategy"]
    assert "get_indicator_detail" in by_server["strategy"]
    assert "check_symbol_available" in by_server["market_data"]
    assert "market_data_get_sync_job" in by_server["market_data"]


def test_strategy_dynamic_state_includes_pre_strategy_data_readiness() -> None:
    state = build_strategy_dynamic_state(
        missing_fields=["strategy_id"],
        collected_fields={},
        pre_strategy_fields={
            "target_market": "crypto",
            "target_instrument": "PEPEUSD",
        },
        pre_strategy_runtime={
            "instrument_data_status": "download_started",
            "instrument_data_symbol": "PEPEUSD",
            "instrument_data_market": "crypto",
            "instrument_available_locally": False,
        },
        session_id="abc-123",
    )

    assert "- pre_strategy_instrument_data_status: download_started" in state
    assert "- pre_strategy_instrument_data_symbol: PEPEUSD" in state
    assert "- pre_strategy_instrument_data_market: crypto" in state
    assert "- pre_strategy_instrument_available_locally: false" in state
