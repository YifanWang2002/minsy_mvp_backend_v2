"""Constants for chat orchestrator modules."""

from __future__ import annotations

import re
from dataclasses import dataclass

from apps.api.agents.handlers.kyc_handler import KYCHandler
from apps.api.agents.phases import Phase

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
_INSTRUCTION_CONTEXT_META_KEY = "instruction_context"
_INSTRUCTION_REFRESH_EVERY_PHASE_TURNS = 20
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

_TOOL_MUTABILITY_READ_ONLY = "read-only"
_TOOL_MUTABILITY_MUTATION = "mutation"
_TOOL_MUTABILITY_DANGEROUS = "dangerous"
_TOOL_MUTABILITY_ORDER: tuple[str, ...] = (
    _TOOL_MUTABILITY_READ_ONLY,
    _TOOL_MUTABILITY_MUTATION,
    _TOOL_MUTABILITY_DANGEROUS,
)

_PRE_STRATEGY_STAGE_INTENT_COLLECTION = "intent_collection"
_STRATEGY_STAGE_SCHEMA_ONLY = "schema_only"
_STRATEGY_STAGE_ARTIFACT_OPS = "artifact_ops"
_STRESS_STAGE_BOOTSTRAP = "bootstrap"
_STRESS_STAGE_FEEDBACK = "feedback"
_STRESS_STAGE_RESERVED = "stress_reserved"
_DEPLOYMENT_STAGE_READY = "deployment_ready"
_DEPLOYMENT_STAGE_DEPLOYED = "deployment_deployed"
_DEPLOYMENT_STAGE_BLOCKED = "deployment_blocked"
_DEPLOYMENT_STAGE_BROKER_DISCOVERY = "deployment_broker_discovery"

_PHASE_MAX_OUTPUT_TOKENS_DEFAULT = 1200
_PHASE_MAX_OUTPUT_TOKENS_BY_PHASE: dict[str, int] = {
    Phase.KYC.value: 900,
    Phase.PRE_STRATEGY.value: 1200,
    Phase.STRATEGY.value: 4200,
    Phase.STRESS_TEST.value: 2200,
    Phase.DEPLOYMENT.value: 1200,
}
_PHASE_MAX_OUTPUT_TOKENS_BY_STAGE: dict[tuple[str, str], int] = {
    (Phase.STRATEGY.value, _STRATEGY_STAGE_SCHEMA_ONLY): 5000,
    (Phase.STRATEGY.value, _STRATEGY_STAGE_ARTIFACT_OPS): 4200,
}

_PRE_STRATEGY_MARKET_DATA_READ_ONLY_TOOL_NAMES: tuple[str, ...] = (
    "check_symbol_available",
    "get_available_symbols",
    "get_symbol_data_coverage",
    "market_data_detect_missing_ranges",
    "market_data_get_sync_job",
    "get_symbol_metadata",
)
_PRE_STRATEGY_MARKET_DATA_MUTATION_TOOL_NAMES: tuple[str, ...] = (
    "market_data_fetch_missing_ranges",
)
_PRE_STRATEGY_MARKET_DATA_DANGEROUS_TOOL_NAMES: tuple[str, ...] = ()

_STRATEGY_SCHEMA_ONLY_READ_ONLY_TOOL_NAMES: tuple[str, ...] = (
    "strategy_validate_dsl",
)
_STRATEGY_SCHEMA_ONLY_MUTATION_TOOL_NAMES: tuple[str, ...] = (
    "strategy_upsert_dsl",
)
_STRATEGY_SCHEMA_ONLY_DANGEROUS_TOOL_NAMES: tuple[str, ...] = ()

_STRATEGY_ARTIFACT_OPS_READ_ONLY_TOOL_NAMES: tuple[str, ...] = (
    "strategy_validate_dsl",
    "strategy_get_dsl",
    "strategy_list_tunable_params",
    "strategy_list_versions",
    "strategy_get_version_dsl",
    "strategy_diff_versions",
    "get_indicator_detail",
    "get_indicator_catalog",
)
_STRATEGY_ARTIFACT_OPS_MUTATION_TOOL_NAMES: tuple[str, ...] = (
    "strategy_upsert_dsl",
    "strategy_patch_dsl",
    "strategy_rollback_dsl",
)
_STRATEGY_ARTIFACT_OPS_DANGEROUS_TOOL_NAMES: tuple[str, ...] = ()

_STRATEGY_MARKET_DATA_READ_ONLY_TOOL_NAMES: tuple[str, ...] = (
    "check_symbol_available",
    "get_available_symbols",
    "get_symbol_data_coverage",
    "get_symbol_metadata",
    "get_symbol_candles",
    "market_data_detect_missing_ranges",
    "market_data_get_sync_job",
)
_STRATEGY_MARKET_DATA_MUTATION_TOOL_NAMES: tuple[str, ...] = (
    "market_data_fetch_missing_ranges",
)
_STRATEGY_MARKET_DATA_DANGEROUS_TOOL_NAMES: tuple[str, ...] = ()

_BACKTEST_BOOTSTRAP_READ_ONLY_TOOL_NAMES: tuple[str, ...] = (
    "backtest_get_job",
)
_BACKTEST_BOOTSTRAP_MUTATION_TOOL_NAMES: tuple[str, ...] = (
    "backtest_create_job",
)
_BACKTEST_BOOTSTRAP_DANGEROUS_TOOL_NAMES: tuple[str, ...] = ()

_BACKTEST_FEEDBACK_READ_ONLY_TOOL_NAMES: tuple[str, ...] = (
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
_BACKTEST_FEEDBACK_MUTATION_TOOL_NAMES: tuple[str, ...] = (
    "backtest_create_job",
)
_BACKTEST_FEEDBACK_DANGEROUS_TOOL_NAMES: tuple[str, ...] = ()

_TRADING_DEPLOYMENT_READ_ONLY_TOOL_NAMES: tuple[str, ...] = (
    "trading_capabilities",
    "trading_list_deployments",
    "trading_get_positions",
    "trading_get_orders",
)
_TRADING_DEPLOYMENT_MUTATION_TOOL_NAMES: tuple[str, ...] = (
    "trading_create_paper_deployment",
    "trading_start_deployment",
    "trading_pause_deployment",
    "trading_stop_deployment",
)
_TRADING_DEPLOYMENT_DANGEROUS_TOOL_NAMES: tuple[str, ...] = ()

# Reserved: stress tools are implemented server-side but not open in the MVP flow.
_STRESS_RESERVED_READ_ONLY_TOOL_NAMES: tuple[str, ...] = (
    "stress_capabilities",
    "stress_black_swan_list_windows",
    "stress_black_swan_get_job",
    "stress_monte_carlo_get_job",
    "stress_param_sensitivity_get_job",
    "stress_optimize_get_job",
    "stress_optimize_get_pareto",
)
_STRESS_RESERVED_MUTATION_TOOL_NAMES: tuple[str, ...] = (
    "stress_black_swan_create_job",
    "stress_monte_carlo_create_job",
    "stress_param_sensitivity_create_job",
    "stress_optimize_create_job",
)
_STRESS_RESERVED_DANGEROUS_TOOL_NAMES: tuple[str, ...] = ()

# Reserved: deployment broker discovery stage (future broker-aware deployment UX).
_DEPLOYMENT_BROKER_DISCOVERY_READ_ONLY_TOOL_NAMES: tuple[str, ...] = (
    "trading_capabilities",
)
_DEPLOYMENT_BROKER_DISCOVERY_MUTATION_TOOL_NAMES: tuple[str, ...] = ()
_DEPLOYMENT_BROKER_DISCOVERY_DANGEROUS_TOOL_NAMES: tuple[str, ...] = ()


def _merge_tool_name_groups(*groups: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for group in groups:
        for name in group:
            if name in seen:
                continue
            seen.add(name)
            ordered.append(name)
    return tuple(ordered)


@dataclass(frozen=True, slots=True)
class PhaseToolMatrixRow:
    phase: str
    stage: str
    server_label: str
    allowed_tools: tuple[str, ...]
    mutability: str
    reachable: bool
    preconditions: str = ""


_PHASE_TOOL_MATRIX: tuple[PhaseToolMatrixRow, ...] = (
    PhaseToolMatrixRow(
        phase=Phase.PRE_STRATEGY.value,
        stage=_PRE_STRATEGY_STAGE_INTENT_COLLECTION,
        server_label="market_data",
        allowed_tools=_PRE_STRATEGY_MARKET_DATA_READ_ONLY_TOOL_NAMES,
        mutability=_TOOL_MUTABILITY_READ_ONLY,
        reachable=True,
        preconditions="KYC completed; collect 4 pre-strategy fields.",
    ),
    PhaseToolMatrixRow(
        phase=Phase.PRE_STRATEGY.value,
        stage=_PRE_STRATEGY_STAGE_INTENT_COLLECTION,
        server_label="market_data",
        allowed_tools=_PRE_STRATEGY_MARKET_DATA_MUTATION_TOOL_NAMES,
        mutability=_TOOL_MUTABILITY_MUTATION,
        reachable=True,
        preconditions="Explicit user consent required before fetch.",
    ),
    PhaseToolMatrixRow(
        phase=Phase.PRE_STRATEGY.value,
        stage=_PRE_STRATEGY_STAGE_INTENT_COLLECTION,
        server_label="market_data",
        allowed_tools=_PRE_STRATEGY_MARKET_DATA_DANGEROUS_TOOL_NAMES,
        mutability=_TOOL_MUTABILITY_DANGEROUS,
        reachable=True,
        preconditions="No dangerous tools exposed in this stage.",
    ),
    PhaseToolMatrixRow(
        phase=Phase.STRATEGY.value,
        stage=_STRATEGY_STAGE_SCHEMA_ONLY,
        server_label="strategy",
        allowed_tools=_STRATEGY_SCHEMA_ONLY_READ_ONLY_TOOL_NAMES,
        mutability=_TOOL_MUTABILITY_READ_ONLY,
        reachable=True,
        preconditions="strategy_id is missing.",
    ),
    PhaseToolMatrixRow(
        phase=Phase.STRATEGY.value,
        stage=_STRATEGY_STAGE_SCHEMA_ONLY,
        server_label="strategy",
        allowed_tools=_STRATEGY_SCHEMA_ONLY_MUTATION_TOOL_NAMES,
        mutability=_TOOL_MUTABILITY_MUTATION,
        reachable=True,
        preconditions="Only when user explicitly asks to save/finalize.",
    ),
    PhaseToolMatrixRow(
        phase=Phase.STRATEGY.value,
        stage=_STRATEGY_STAGE_SCHEMA_ONLY,
        server_label="strategy",
        allowed_tools=_STRATEGY_SCHEMA_ONLY_DANGEROUS_TOOL_NAMES,
        mutability=_TOOL_MUTABILITY_DANGEROUS,
        reachable=True,
        preconditions="No dangerous tools exposed in this stage.",
    ),
    PhaseToolMatrixRow(
        phase=Phase.STRATEGY.value,
        stage=_STRATEGY_STAGE_ARTIFACT_OPS,
        server_label="strategy",
        allowed_tools=_STRATEGY_ARTIFACT_OPS_READ_ONLY_TOOL_NAMES,
        mutability=_TOOL_MUTABILITY_READ_ONLY,
        reachable=True,
        preconditions="strategy_id is available.",
    ),
    PhaseToolMatrixRow(
        phase=Phase.STRATEGY.value,
        stage=_STRATEGY_STAGE_ARTIFACT_OPS,
        server_label="strategy",
        allowed_tools=_STRATEGY_ARTIFACT_OPS_MUTATION_TOOL_NAMES,
        mutability=_TOOL_MUTABILITY_MUTATION,
        reachable=True,
        preconditions="Patch first; upsert only as fallback.",
    ),
    PhaseToolMatrixRow(
        phase=Phase.STRATEGY.value,
        stage=_STRATEGY_STAGE_ARTIFACT_OPS,
        server_label="strategy",
        allowed_tools=_STRATEGY_ARTIFACT_OPS_DANGEROUS_TOOL_NAMES,
        mutability=_TOOL_MUTABILITY_DANGEROUS,
        reachable=True,
        preconditions="No dangerous tools exposed in this stage.",
    ),
    PhaseToolMatrixRow(
        phase=Phase.STRATEGY.value,
        stage=_STRATEGY_STAGE_ARTIFACT_OPS,
        server_label="market_data",
        allowed_tools=_STRATEGY_MARKET_DATA_READ_ONLY_TOOL_NAMES,
        mutability=_TOOL_MUTABILITY_READ_ONLY,
        reachable=True,
        preconditions="Coverage check before each backtest create.",
    ),
    PhaseToolMatrixRow(
        phase=Phase.STRATEGY.value,
        stage=_STRATEGY_STAGE_ARTIFACT_OPS,
        server_label="market_data",
        allowed_tools=_STRATEGY_MARKET_DATA_MUTATION_TOOL_NAMES,
        mutability=_TOOL_MUTABILITY_MUTATION,
        reachable=True,
        preconditions="Only when missing-range fetch is confirmed/required.",
    ),
    PhaseToolMatrixRow(
        phase=Phase.STRATEGY.value,
        stage=_STRATEGY_STAGE_ARTIFACT_OPS,
        server_label="market_data",
        allowed_tools=_STRATEGY_MARKET_DATA_DANGEROUS_TOOL_NAMES,
        mutability=_TOOL_MUTABILITY_DANGEROUS,
        reachable=True,
        preconditions="No dangerous tools exposed in this stage.",
    ),
    PhaseToolMatrixRow(
        phase=Phase.STRATEGY.value,
        stage=_STRATEGY_STAGE_ARTIFACT_OPS,
        server_label="backtest",
        allowed_tools=_BACKTEST_FEEDBACK_READ_ONLY_TOOL_NAMES,
        mutability=_TOOL_MUTABILITY_READ_ONLY,
        reachable=True,
        preconditions="Use get_job/chart tools after create_job.",
    ),
    PhaseToolMatrixRow(
        phase=Phase.STRATEGY.value,
        stage=_STRATEGY_STAGE_ARTIFACT_OPS,
        server_label="backtest",
        allowed_tools=_BACKTEST_FEEDBACK_MUTATION_TOOL_NAMES,
        mutability=_TOOL_MUTABILITY_MUTATION,
        reachable=True,
        preconditions="Backtest window must respect coverage bounds and bar cap.",
    ),
    PhaseToolMatrixRow(
        phase=Phase.STRATEGY.value,
        stage=_STRATEGY_STAGE_ARTIFACT_OPS,
        server_label="backtest",
        allowed_tools=_BACKTEST_FEEDBACK_DANGEROUS_TOOL_NAMES,
        mutability=_TOOL_MUTABILITY_DANGEROUS,
        reachable=True,
        preconditions="No dangerous tools exposed in this stage.",
    ),
    PhaseToolMatrixRow(
        phase=Phase.STRESS_TEST.value,
        stage=_STRESS_STAGE_BOOTSTRAP,
        server_label="market_data",
        allowed_tools=_STRATEGY_MARKET_DATA_READ_ONLY_TOOL_NAMES,
        mutability=_TOOL_MUTABILITY_READ_ONLY,
        reachable=False,
        preconditions="Legacy stage only; stress flow disabled by boundary redirect.",
    ),
    PhaseToolMatrixRow(
        phase=Phase.STRESS_TEST.value,
        stage=_STRESS_STAGE_BOOTSTRAP,
        server_label="market_data",
        allowed_tools=_STRATEGY_MARKET_DATA_MUTATION_TOOL_NAMES,
        mutability=_TOOL_MUTABILITY_MUTATION,
        reachable=False,
        preconditions="Legacy stage only; stress flow disabled by boundary redirect.",
    ),
    PhaseToolMatrixRow(
        phase=Phase.STRESS_TEST.value,
        stage=_STRESS_STAGE_BOOTSTRAP,
        server_label="backtest",
        allowed_tools=_BACKTEST_BOOTSTRAP_READ_ONLY_TOOL_NAMES,
        mutability=_TOOL_MUTABILITY_READ_ONLY,
        reachable=False,
        preconditions="Legacy stage only; stress flow disabled by boundary redirect.",
    ),
    PhaseToolMatrixRow(
        phase=Phase.STRESS_TEST.value,
        stage=_STRESS_STAGE_BOOTSTRAP,
        server_label="backtest",
        allowed_tools=_BACKTEST_BOOTSTRAP_MUTATION_TOOL_NAMES,
        mutability=_TOOL_MUTABILITY_MUTATION,
        reachable=False,
        preconditions="Legacy stage only; stress flow disabled by boundary redirect.",
    ),
    PhaseToolMatrixRow(
        phase=Phase.STRESS_TEST.value,
        stage=_STRESS_STAGE_FEEDBACK,
        server_label="market_data",
        allowed_tools=_STRATEGY_MARKET_DATA_READ_ONLY_TOOL_NAMES,
        mutability=_TOOL_MUTABILITY_READ_ONLY,
        reachable=False,
        preconditions="Legacy stage only; stress flow disabled by boundary redirect.",
    ),
    PhaseToolMatrixRow(
        phase=Phase.STRESS_TEST.value,
        stage=_STRESS_STAGE_FEEDBACK,
        server_label="market_data",
        allowed_tools=_STRATEGY_MARKET_DATA_MUTATION_TOOL_NAMES,
        mutability=_TOOL_MUTABILITY_MUTATION,
        reachable=False,
        preconditions="Legacy stage only; stress flow disabled by boundary redirect.",
    ),
    PhaseToolMatrixRow(
        phase=Phase.STRESS_TEST.value,
        stage=_STRESS_STAGE_FEEDBACK,
        server_label="backtest",
        allowed_tools=_BACKTEST_FEEDBACK_READ_ONLY_TOOL_NAMES,
        mutability=_TOOL_MUTABILITY_READ_ONLY,
        reachable=False,
        preconditions="Legacy stage only; stress flow disabled by boundary redirect.",
    ),
    PhaseToolMatrixRow(
        phase=Phase.STRESS_TEST.value,
        stage=_STRESS_STAGE_FEEDBACK,
        server_label="backtest",
        allowed_tools=_BACKTEST_FEEDBACK_MUTATION_TOOL_NAMES,
        mutability=_TOOL_MUTABILITY_MUTATION,
        reachable=False,
        preconditions="Legacy stage only; stress flow disabled by boundary redirect.",
    ),
    PhaseToolMatrixRow(
        phase=Phase.STRESS_TEST.value,
        stage=_STRESS_STAGE_FEEDBACK,
        server_label="strategy",
        allowed_tools=_STRATEGY_ARTIFACT_OPS_READ_ONLY_TOOL_NAMES,
        mutability=_TOOL_MUTABILITY_READ_ONLY,
        reachable=False,
        preconditions="Legacy stage only; stress flow disabled by boundary redirect.",
    ),
    PhaseToolMatrixRow(
        phase=Phase.STRESS_TEST.value,
        stage=_STRESS_STAGE_FEEDBACK,
        server_label="strategy",
        allowed_tools=_STRATEGY_ARTIFACT_OPS_MUTATION_TOOL_NAMES,
        mutability=_TOOL_MUTABILITY_MUTATION,
        reachable=False,
        preconditions="Legacy stage only; stress flow disabled by boundary redirect.",
    ),
    PhaseToolMatrixRow(
        phase=Phase.STRESS_TEST.value,
        stage=_STRESS_STAGE_RESERVED,
        server_label="stress",
        allowed_tools=_STRESS_RESERVED_READ_ONLY_TOOL_NAMES,
        mutability=_TOOL_MUTABILITY_READ_ONLY,
        reachable=False,
        preconditions="Reserved future stress capability exposure.",
    ),
    PhaseToolMatrixRow(
        phase=Phase.STRESS_TEST.value,
        stage=_STRESS_STAGE_RESERVED,
        server_label="stress",
        allowed_tools=_STRESS_RESERVED_MUTATION_TOOL_NAMES,
        mutability=_TOOL_MUTABILITY_MUTATION,
        reachable=False,
        preconditions="Reserved future stress capability exposure.",
    ),
    PhaseToolMatrixRow(
        phase=Phase.STRESS_TEST.value,
        stage=_STRESS_STAGE_RESERVED,
        server_label="stress",
        allowed_tools=_STRESS_RESERVED_DANGEROUS_TOOL_NAMES,
        mutability=_TOOL_MUTABILITY_DANGEROUS,
        reachable=False,
        preconditions="No dangerous stress tools exposed by default.",
    ),
    PhaseToolMatrixRow(
        phase=Phase.DEPLOYMENT.value,
        stage=_DEPLOYMENT_STAGE_READY,
        server_label="trading",
        allowed_tools=_TRADING_DEPLOYMENT_READ_ONLY_TOOL_NAMES,
        mutability=_TOOL_MUTABILITY_READ_ONLY,
        reachable=True,
        preconditions="strategy_confirmed=true; deployment_status=ready.",
    ),
    PhaseToolMatrixRow(
        phase=Phase.DEPLOYMENT.value,
        stage=_DEPLOYMENT_STAGE_READY,
        server_label="trading",
        allowed_tools=_TRADING_DEPLOYMENT_MUTATION_TOOL_NAMES,
        mutability=_TOOL_MUTABILITY_MUTATION,
        reachable=True,
        preconditions="Paper deployment only; user context required.",
    ),
    PhaseToolMatrixRow(
        phase=Phase.DEPLOYMENT.value,
        stage=_DEPLOYMENT_STAGE_READY,
        server_label="trading",
        allowed_tools=_TRADING_DEPLOYMENT_DANGEROUS_TOOL_NAMES,
        mutability=_TOOL_MUTABILITY_DANGEROUS,
        reachable=True,
        preconditions="No dangerous tools exposed in this stage.",
    ),
    PhaseToolMatrixRow(
        phase=Phase.DEPLOYMENT.value,
        stage=_DEPLOYMENT_STAGE_DEPLOYED,
        server_label="trading",
        allowed_tools=_TRADING_DEPLOYMENT_READ_ONLY_TOOL_NAMES,
        mutability=_TOOL_MUTABILITY_READ_ONLY,
        reachable=True,
        preconditions="deployment_status=deployed.",
    ),
    PhaseToolMatrixRow(
        phase=Phase.DEPLOYMENT.value,
        stage=_DEPLOYMENT_STAGE_DEPLOYED,
        server_label="trading",
        allowed_tools=_TRADING_DEPLOYMENT_MUTATION_TOOL_NAMES,
        mutability=_TOOL_MUTABILITY_MUTATION,
        reachable=True,
        preconditions="Paper deployment lifecycle operations only.",
    ),
    PhaseToolMatrixRow(
        phase=Phase.DEPLOYMENT.value,
        stage=_DEPLOYMENT_STAGE_DEPLOYED,
        server_label="trading",
        allowed_tools=_TRADING_DEPLOYMENT_DANGEROUS_TOOL_NAMES,
        mutability=_TOOL_MUTABILITY_DANGEROUS,
        reachable=True,
        preconditions="No dangerous tools exposed in this stage.",
    ),
    PhaseToolMatrixRow(
        phase=Phase.DEPLOYMENT.value,
        stage=_DEPLOYMENT_STAGE_BLOCKED,
        server_label="trading",
        allowed_tools=_TRADING_DEPLOYMENT_READ_ONLY_TOOL_NAMES,
        mutability=_TOOL_MUTABILITY_READ_ONLY,
        reachable=True,
        preconditions="deployment_status=blocked.",
    ),
    PhaseToolMatrixRow(
        phase=Phase.DEPLOYMENT.value,
        stage=_DEPLOYMENT_STAGE_BLOCKED,
        server_label="trading",
        allowed_tools=_TRADING_DEPLOYMENT_MUTATION_TOOL_NAMES,
        mutability=_TOOL_MUTABILITY_MUTATION,
        reachable=True,
        preconditions="Allow pause/stop/recover decisions.",
    ),
    PhaseToolMatrixRow(
        phase=Phase.DEPLOYMENT.value,
        stage=_DEPLOYMENT_STAGE_BLOCKED,
        server_label="trading",
        allowed_tools=_TRADING_DEPLOYMENT_DANGEROUS_TOOL_NAMES,
        mutability=_TOOL_MUTABILITY_DANGEROUS,
        reachable=True,
        preconditions="No dangerous tools exposed in this stage.",
    ),
    PhaseToolMatrixRow(
        phase=Phase.DEPLOYMENT.value,
        stage=_DEPLOYMENT_STAGE_BROKER_DISCOVERY,
        server_label="trading",
        allowed_tools=_DEPLOYMENT_BROKER_DISCOVERY_READ_ONLY_TOOL_NAMES,
        mutability=_TOOL_MUTABILITY_READ_ONLY,
        reachable=False,
        preconditions="Reserved for broker discovery and capability matching.",
    ),
    PhaseToolMatrixRow(
        phase=Phase.DEPLOYMENT.value,
        stage=_DEPLOYMENT_STAGE_BROKER_DISCOVERY,
        server_label="trading",
        allowed_tools=_DEPLOYMENT_BROKER_DISCOVERY_MUTATION_TOOL_NAMES,
        mutability=_TOOL_MUTABILITY_MUTATION,
        reachable=False,
        preconditions="Reserved for broker discovery and capability matching.",
    ),
    PhaseToolMatrixRow(
        phase=Phase.DEPLOYMENT.value,
        stage=_DEPLOYMENT_STAGE_BROKER_DISCOVERY,
        server_label="trading",
        allowed_tools=_DEPLOYMENT_BROKER_DISCOVERY_DANGEROUS_TOOL_NAMES,
        mutability=_TOOL_MUTABILITY_DANGEROUS,
        reachable=False,
        preconditions="Reserved stage: no dangerous tools by default.",
    ),
)


def get_phase_tool_matrix_rows(
    *,
    phase: str | None = None,
    stage: str | None = None,
    reachable_only: bool = False,
) -> tuple[PhaseToolMatrixRow, ...]:
    selected: list[PhaseToolMatrixRow] = []
    phase_key = phase.strip() if isinstance(phase, str) else None
    stage_key = stage.strip() if isinstance(stage, str) else None
    for row in _PHASE_TOOL_MATRIX:
        if phase_key is not None and row.phase != phase_key:
            continue
        if stage_key is not None and row.stage != stage_key:
            continue
        if reachable_only and not row.reachable:
            continue
        selected.append(row)
    return tuple(selected)


def resolve_phase_max_output_tokens(*, phase: str, stage: str | None = None) -> int:
    phase_key = phase.strip() if isinstance(phase, str) else ""
    stage_key = stage.strip() if isinstance(stage, str) else ""
    if phase_key and stage_key:
        staged = _PHASE_MAX_OUTPUT_TOKENS_BY_STAGE.get((phase_key, stage_key))
        if isinstance(staged, int) and staged > 0:
            return staged

    phase_budget = _PHASE_MAX_OUTPUT_TOKENS_BY_PHASE.get(phase_key)
    if isinstance(phase_budget, int) and phase_budget > 0:
        return phase_budget
    return _PHASE_MAX_OUTPUT_TOKENS_DEFAULT


_STRATEGY_SCHEMA_ONLY_TOOL_NAMES: tuple[str, ...] = _merge_tool_name_groups(
    _STRATEGY_SCHEMA_ONLY_READ_ONLY_TOOL_NAMES,
    _STRATEGY_SCHEMA_ONLY_MUTATION_TOOL_NAMES,
)
_STRATEGY_ARTIFACT_OPS_TOOL_NAMES: tuple[str, ...] = _merge_tool_name_groups(
    _STRATEGY_ARTIFACT_OPS_READ_ONLY_TOOL_NAMES,
    _STRATEGY_ARTIFACT_OPS_MUTATION_TOOL_NAMES,
)
_MARKET_DATA_MINIMAL_TOOL_NAMES: tuple[str, ...] = _merge_tool_name_groups(
    _PRE_STRATEGY_MARKET_DATA_READ_ONLY_TOOL_NAMES,
    _PRE_STRATEGY_MARKET_DATA_MUTATION_TOOL_NAMES,
)
_BACKTEST_BOOTSTRAP_TOOL_NAMES: tuple[str, ...] = _merge_tool_name_groups(
    _BACKTEST_BOOTSTRAP_READ_ONLY_TOOL_NAMES,
    _BACKTEST_BOOTSTRAP_MUTATION_TOOL_NAMES,
)
_BACKTEST_FEEDBACK_TOOL_NAMES: tuple[str, ...] = _merge_tool_name_groups(
    _BACKTEST_FEEDBACK_READ_ONLY_TOOL_NAMES,
    _BACKTEST_FEEDBACK_MUTATION_TOOL_NAMES,
)
_TRADING_DEPLOYMENT_TOOL_NAMES: tuple[str, ...] = _merge_tool_name_groups(
    _TRADING_DEPLOYMENT_READ_ONLY_TOOL_NAMES,
    _TRADING_DEPLOYMENT_MUTATION_TOOL_NAMES,
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
    "_INSTRUCTION_CONTEXT_META_KEY",
    "_INSTRUCTION_REFRESH_EVERY_PHASE_TURNS",
    "_OPENAI_STREAM_HARD_TIMEOUT_SECONDS",
    "_UUID_CANDIDATE_PATTERN",
    "_BACKTEST_COMPLETED_TOOL_NAMES",
    "_BACKTEST_DEFAULT_CHARTS",
    "PhaseToolMatrixRow",
    "get_phase_tool_matrix_rows",
    "_TOOL_MUTABILITY_READ_ONLY",
    "_TOOL_MUTABILITY_MUTATION",
    "_TOOL_MUTABILITY_DANGEROUS",
    "_TOOL_MUTABILITY_ORDER",
    "_PRE_STRATEGY_STAGE_INTENT_COLLECTION",
    "_STRATEGY_STAGE_SCHEMA_ONLY",
    "_STRATEGY_STAGE_ARTIFACT_OPS",
    "_STRESS_STAGE_BOOTSTRAP",
    "_STRESS_STAGE_FEEDBACK",
    "_STRESS_STAGE_RESERVED",
    "_DEPLOYMENT_STAGE_READY",
    "_DEPLOYMENT_STAGE_DEPLOYED",
    "_DEPLOYMENT_STAGE_BLOCKED",
    "_DEPLOYMENT_STAGE_BROKER_DISCOVERY",
    "_PHASE_MAX_OUTPUT_TOKENS_DEFAULT",
    "_PHASE_MAX_OUTPUT_TOKENS_BY_PHASE",
    "_PHASE_MAX_OUTPUT_TOKENS_BY_STAGE",
    "resolve_phase_max_output_tokens",
    "_PHASE_TOOL_MATRIX",
    "_STRATEGY_SCHEMA_ONLY_TOOL_NAMES",
    "_STRATEGY_ARTIFACT_OPS_TOOL_NAMES",
    "_MARKET_DATA_MINIMAL_TOOL_NAMES",
    "_BACKTEST_BOOTSTRAP_TOOL_NAMES",
    "_BACKTEST_FEEDBACK_TOOL_NAMES",
    "_TRADING_DEPLOYMENT_TOOL_NAMES",
    "_MCP_CONTEXT_ENABLED_SERVER_LABELS",
    "_kyc_handler",
]
