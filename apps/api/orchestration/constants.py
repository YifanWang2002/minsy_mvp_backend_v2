"""Constants for chat orchestrator modules."""

from __future__ import annotations

import re

from apps.api.agents.handlers.kyc_handler import KYCHandler

_AGENT_UI_TAG = "AGENT_UI_JSON"
_AGENT_STATE_PATCH_TAG = "AGENT_STATE_PATCH"
_STRATEGY_CARD_GENUI_TYPE = "strategy_card"
_STRATEGY_REF_GENUI_TYPE = "strategy_ref"
_BACKTEST_CHARTS_GENUI_TYPE = "backtest_charts"
_STOP_CRITERIA_TURN_LIMIT = 10
_PHASE_CARRYOVER_TAG = "PHASE CARRYOVER MEMORY"
_PHASE_CARRYOVER_MAX_TURNS = 4
_PHASE_CARRYOVER_MAX_CHARS_PER_UTTERANCE = 220
_PHASE_CARRYOVER_META_KEY = "phase_carryover_memory"
_STREAM_RECOVERY_META_KEY = "stream_recovery"
_STREAM_RECOVERY_STATE_STREAMING = "streaming"
_STREAM_RECOVERY_STATE_COMPLETED = "completed"
_STREAM_RECOVERY_STATE_FAILED = "failed"
_OPENAI_STREAM_HARD_TIMEOUT_SECONDS = 300.0
_UUID_CANDIDATE_PATTERN = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
_BACKTEST_COMPLETED_TOOL_NAMES: frozenset[str] = frozenset(
    {"backtest_get_job", "backtest_create_job"}
)
_BACKTEST_DEFAULT_CHARTS: tuple[str, ...] = (
    "equity_curve",
    "underwater_curve",
    "monthly_return_table",
    "holding_period_pnl_bins",
)
_STRATEGY_SCHEMA_ONLY_TOOL_NAMES: tuple[str, ...] = (
    "strategy_validate_dsl",
    "strategy_upsert_dsl",
)
_STRATEGY_ARTIFACT_OPS_TOOL_NAMES: tuple[str, ...] = (
    "strategy_validate_dsl",
    "strategy_upsert_dsl",
    "strategy_get_dsl",
    "strategy_list_tunable_params",
    "strategy_patch_dsl",
    "strategy_list_versions",
    "strategy_get_version_dsl",
    "strategy_diff_versions",
    "strategy_rollback_dsl",
    "get_indicator_detail",
    "get_indicator_catalog",
)
_MARKET_DATA_MINIMAL_TOOL_NAMES: tuple[str, ...] = ("get_symbol_data_coverage",)
_BACKTEST_BOOTSTRAP_TOOL_NAMES: tuple[str, ...] = (
    "backtest_create_job",
    "backtest_get_job",
)
_BACKTEST_FEEDBACK_TOOL_NAMES: tuple[str, ...] = (
    "backtest_create_job",
    "backtest_get_job",
    "backtest_entry_hour_pnl_heatmap",
    "backtest_entry_weekday_pnl",
    "backtest_monthly_return_table",
    "backtest_holding_period_pnl_bins",
    "backtest_long_short_breakdown",
    "backtest_exit_reason_breakdown",
    "backtest_underwater_curve",
    "backtest_rolling_metrics",
)
_TRADING_DEPLOYMENT_TOOL_NAMES: tuple[str, ...] = (
    "trading_create_paper_deployment",
    "trading_list_deployments",
    "trading_start_deployment",
    "trading_pause_deployment",
    "trading_stop_deployment",
    "trading_get_positions",
    "trading_get_orders",
)
_MCP_CONTEXT_ENABLED_SERVER_LABELS: frozenset[str] = frozenset(
    {"strategy", "backtest", "market_data", "stress", "trading"}
)

# Singleton for KYC-specific helpers (profile loading from UserProfile)
_kyc_handler = KYCHandler()

__all__ = [
    "_AGENT_UI_TAG",
    "_AGENT_STATE_PATCH_TAG",
    "_STRATEGY_CARD_GENUI_TYPE",
    "_STRATEGY_REF_GENUI_TYPE",
    "_BACKTEST_CHARTS_GENUI_TYPE",
    "_STOP_CRITERIA_TURN_LIMIT",
    "_PHASE_CARRYOVER_TAG",
    "_PHASE_CARRYOVER_MAX_TURNS",
    "_PHASE_CARRYOVER_MAX_CHARS_PER_UTTERANCE",
    "_PHASE_CARRYOVER_META_KEY",
    "_STREAM_RECOVERY_META_KEY",
    "_STREAM_RECOVERY_STATE_STREAMING",
    "_STREAM_RECOVERY_STATE_COMPLETED",
    "_STREAM_RECOVERY_STATE_FAILED",
    "_OPENAI_STREAM_HARD_TIMEOUT_SECONDS",
    "_UUID_CANDIDATE_PATTERN",
    "_BACKTEST_COMPLETED_TOOL_NAMES",
    "_BACKTEST_DEFAULT_CHARTS",
    "_STRATEGY_SCHEMA_ONLY_TOOL_NAMES",
    "_STRATEGY_ARTIFACT_OPS_TOOL_NAMES",
    "_MARKET_DATA_MINIMAL_TOOL_NAMES",
    "_BACKTEST_BOOTSTRAP_TOOL_NAMES",
    "_BACKTEST_FEEDBACK_TOOL_NAMES",
    "_TRADING_DEPLOYMENT_TOOL_NAMES",
    "_MCP_CONTEXT_ENABLED_SERVER_LABELS",
    "_kyc_handler",
]
