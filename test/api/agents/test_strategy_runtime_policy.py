from __future__ import annotations

from apps.api.agents.phases import Phase
from apps.api.agents.skills.strategy_skills import build_strategy_dynamic_state
from apps.api.orchestration import ChatOrchestrator


def test_strategy_schema_only_runtime_policy_exposes_indicator_and_market_data_tools() -> (
    None
):
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
    assert "get_indicator_detail" not in by_server["strategy"]
    assert "check_symbol_available" in by_server["market_data"]
    assert "market_data_get_sync_job" in by_server["market_data"]


def test_strategy_dynamic_state_includes_pre_strategy_data_readiness() -> None:
    state = build_strategy_dynamic_state(
        missing_fields=["strategy_id"],
        collected_fields={},
        pre_strategy_fields={
            "target_market": "crypto",
            "target_instrument": "PEPEUSD",
            "strategy_family_choice": "trend_continuation",
        },
        pre_strategy_runtime={
            "instrument_data_status": "download_started",
            "instrument_data_symbol": "PEPEUSD",
            "instrument_data_market": "crypto",
            "instrument_available_locally": False,
            "timeframe_plan": {
                "primary": "1h",
                "secondary": "4h",
                "mapping_reason": "test",
            },
            "regime_summary_short": "trend is stronger than range",
        },
        session_id="abc-123",
    )

    assert "- pre_strategy_instrument_data_status: download_started" in state
    assert "- pre_strategy_instrument_data_symbol: PEPEUSD" in state
    assert "- pre_strategy_instrument_data_market: crypto" in state
    assert "- pre_strategy_instrument_available_locally: false" in state
    assert "- pre_strategy_strategy_family_choice: trend_continuation" in state
    assert "- pre_strategy_timeframe_primary: 1h" in state
    assert "- pre_strategy_market_regime_summary: trend is stronger than range" in state


def test_strategy_artifact_ops_runtime_policy_allows_trade_snapshots() -> None:
    orchestrator = ChatOrchestrator(None)  # type: ignore[arg-type]

    policy = orchestrator._build_phase_runtime_policy(
        phase=Phase.STRATEGY.value,
        artifacts={
            Phase.STRATEGY.value: {
                "profile": {"strategy_id": "8c6f5452-f4af-4f19-b2f1-9ea956c49a4e"},
                "missing_fields": [],
            },
        },
    )

    assert policy.phase_stage == "artifact_ops"
    assert policy.allowed_tools is not None

    by_server = {
        tool["server_label"]: set(tool.get("allowed_tools", []))
        for tool in policy.allowed_tools
    }
    assert "backtest" in by_server
    assert "backtest_trade_snapshots" in by_server["backtest"]


def test_strategy_dynamic_state_includes_trade_snapshot_request_and_pending_patch() -> (
    None
):
    state = build_strategy_dynamic_state(
        missing_fields=[],
        collected_fields={
            "strategy_id": "8c6f5452-f4af-4f19-b2f1-9ea956c49a4e",
            "strategy_market": "crypto",
            "strategy_primary_symbol": "BTCUSD",
            "strategy_tickers_csv": "BTCUSD",
            "strategy_timeframe": "5m",
        },
        pre_strategy_fields={},
        pre_strategy_runtime={},
        session_id="session-1",
        choice_selection={
            "choice_id": "trade_patch_apply_decision",
            "selected_option_id": "apply_recommended_patch",
        },
        trade_snapshot_request={
            "job_id": "job-1",
            "trade_index": 2,
            "filters": {"side": "long"},
        },
        pending_trade_patch={
            "strategy_id": "8c6f5452-f4af-4f19-b2f1-9ea956c49a4e",
            "patch_ops": [{"op": "replace", "path": "/foo", "value": 1}],
        },
    )

    assert "- structured_choice_selection_present: true" in state
    assert "- trade_snapshot_request_present: true" in state
    assert "- pending_trade_patch_present: true" in state
