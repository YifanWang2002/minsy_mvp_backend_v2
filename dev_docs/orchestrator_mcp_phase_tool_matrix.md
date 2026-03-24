# Orchestrator MCP Phase Tool Matrix

Source of truth: `apps/api/orchestration/constants.py` (`_PHASE_TOOL_MATRIX`).

## Scope

- Policy owner: orchestrator runtime policy (`PromptBuilderMixin`).
- Handler-level `tools` are fallback only.
- Matrix row schema:
  - `phase`
  - `stage`
  - `server_label`
  - `allowed_tools`
  - `mutability` (`read-only` / `mutation` / `dangerous`)
  - `reachable`
  - `preconditions`

## Matrix (Audit View)

| phase | stage | server | mutability | reachable | preconditions | allowed_tools |
|---|---|---|---|---|---|---|
| pre_strategy | intent_collection | market_data | read-only | yes | KYC completed; collect 4 pre-strategy fields | check_symbol_available, get_available_symbols, get_symbol_data_coverage, market_data_detect_missing_ranges, market_data_get_sync_job, get_symbol_metadata |
| pre_strategy | intent_collection | market_data | mutation | yes | explicit user consent required before fetch | market_data_fetch_missing_ranges |
| strategy | schema_only | strategy | read-only | yes | strategy_id missing | strategy_validate_dsl |
| strategy | schema_only | strategy | mutation | yes | explicit save/finalize intent | strategy_upsert_dsl |
| strategy | artifact_ops | strategy | read-only | yes | strategy_id available | strategy_validate_dsl, strategy_get_dsl, strategy_list_tunable_params, strategy_list_versions, strategy_get_version_dsl, strategy_diff_versions, get_indicator_catalog |
| strategy | artifact_ops | strategy | mutation | yes | patch first, upsert fallback | strategy_upsert_dsl, strategy_patch_dsl, strategy_rollback_dsl |
| strategy | artifact_ops | market_data | read-only | yes | coverage guardrail before backtest | check_symbol_available, get_available_symbols, get_symbol_data_coverage, get_symbol_metadata, get_symbol_candles, market_data_detect_missing_ranges, market_data_get_sync_job |
| strategy | artifact_ops | market_data | mutation | yes | only when coverage missing and confirmed | market_data_fetch_missing_ranges |
| strategy | artifact_ops | backtest | read-only | yes | use get/analytics after create | backtest_get_job, backtest_entry_hour_pnl_heatmap, backtest_entry_weekday_pnl, backtest_monthly_return_table, backtest_holding_period_pnl_bins, backtest_long_short_breakdown, backtest_exit_reason_breakdown, backtest_underwater_curve, backtest_rolling_metrics |
| strategy | artifact_ops | backtest | mutation | yes | date bounds must fit coverage and bar cap | backtest_create_job |
| stress_test | bootstrap | market_data | read-only | no | legacy only (redirected to strategy) | check_symbol_available, get_available_symbols, get_symbol_data_coverage, get_symbol_metadata, get_symbol_candles, market_data_detect_missing_ranges, market_data_get_sync_job |
| stress_test | bootstrap | market_data | mutation | no | legacy only (redirected to strategy) | market_data_fetch_missing_ranges |
| stress_test | bootstrap | backtest | read-only | no | legacy only (redirected to strategy) | backtest_get_job |
| stress_test | bootstrap | backtest | mutation | no | legacy only (redirected to strategy) | backtest_create_job |
| stress_test | feedback | market_data | read-only | no | legacy only (redirected to strategy) | check_symbol_available, get_available_symbols, get_symbol_data_coverage, get_symbol_metadata, get_symbol_candles, market_data_detect_missing_ranges, market_data_get_sync_job |
| stress_test | feedback | market_data | mutation | no | legacy only (redirected to strategy) | market_data_fetch_missing_ranges |
| stress_test | feedback | backtest | read-only | no | legacy only (redirected to strategy) | backtest_get_job, backtest_entry_hour_pnl_heatmap, backtest_entry_weekday_pnl, backtest_monthly_return_table, backtest_holding_period_pnl_bins, backtest_long_short_breakdown, backtest_exit_reason_breakdown, backtest_underwater_curve, backtest_rolling_metrics |
| stress_test | feedback | backtest | mutation | no | legacy only (redirected to strategy) | backtest_create_job |
| stress_test | feedback | strategy | read-only | no | legacy only (redirected to strategy) | strategy_validate_dsl, strategy_get_dsl, strategy_list_tunable_params, strategy_list_versions, strategy_get_version_dsl, strategy_diff_versions, get_indicator_catalog |
| stress_test | feedback | strategy | mutation | no | legacy only (redirected to strategy) | strategy_upsert_dsl, strategy_patch_dsl, strategy_rollback_dsl |
| stress_test | stress_reserved | stress | read-only | no | reserved future stress exposure | stress_capabilities, stress_black_swan_list_windows, stress_black_swan_get_job, stress_monte_carlo_get_job, stress_param_sensitivity_get_job, stress_optimize_get_job, stress_optimize_get_pareto |
| stress_test | stress_reserved | stress | mutation | no | reserved future stress exposure | stress_black_swan_create_job, stress_monte_carlo_create_job, stress_param_sensitivity_create_job, stress_optimize_create_job |
| deployment | deployment_ready | trading | read-only | yes | strategy confirmed and ready to deploy | trading_capabilities, trading_list_deployments, trading_get_positions, trading_get_orders |
| deployment | deployment_ready | trading | mutation | yes | paper deployment lifecycle ops | trading_create_paper_deployment, trading_start_deployment, trading_pause_deployment, trading_stop_deployment |
| deployment | deployment_deployed | trading | read-only | yes | deployment_status=deployed | trading_capabilities, trading_list_deployments, trading_get_positions, trading_get_orders |
| deployment | deployment_deployed | trading | mutation | yes | runtime lifecycle controls | trading_create_paper_deployment, trading_start_deployment, trading_pause_deployment, trading_stop_deployment |
| deployment | deployment_blocked | trading | read-only | yes | deployment_status=blocked | trading_capabilities, trading_list_deployments, trading_get_positions, trading_get_orders |
| deployment | deployment_blocked | trading | mutation | yes | recover/pause/stop flow | trading_create_paper_deployment, trading_start_deployment, trading_pause_deployment, trading_stop_deployment |
| deployment | deployment_broker_discovery | trading | read-only | no | reserved broker capability discovery stage | trading_capabilities |

## Runtime assembly rules

- Orchestrator computes stage from artifacts, then reads matrix rows.
- Rows are merged by `server_label` in mutability order:
  1. `read-only`
  2. `mutation`
  3. `dangerous`
- Result is emitted as MCP tool defs with `require_approval="never"`.
- Unknown server labels are ignored at assembly time.

## Current product boundary notes

- `stress_test` phase is redirected back to `strategy` by boundary guard before prompt/tool execution.
- `deployment_broker_discovery` and `stress_reserved` are intentionally marked `reachable=no` as future slots.
