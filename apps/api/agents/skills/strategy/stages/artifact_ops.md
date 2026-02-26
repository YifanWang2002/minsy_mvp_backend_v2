[STAGE_MARKER_STRATEGY_ARTIFACT_OPS]
Stage objective:
- Assume a confirmed strategy artifact already exists and stay in strategy phase.
- Keep all performance/backtest analysis in this phase (do not hand off to `stress_test`).
- For update-by-id, prefer:
  1) `strategy_get_dsl` (latest payload + version, pass `session_id=tool_compat_session_id` when available)
  2) `strategy_patch_dsl` with minimal patch ops and `expected_version` (pass `session_id=tool_compat_session_id` when available)
- For history/compare requests, use:
  1) `strategy_list_versions`
  2) `strategy_diff_versions` and/or `strategy_get_version_dsl`
- For rollback requests, use:
  1) `strategy_rollback_dsl` with `target_version` and current `expected_version`
- If patch path is not suitable for the requested change, fallback to:
  1) `strategy_validate_dsl` on edited JSON (pass `session_id=tool_compat_session_id` when available)
  2) `strategy_upsert_dsl` with existing `strategy_id` (pass `session_id=tool_compat_session_id` when available)
- If `strategy_get_dsl`/`strategy_patch_dsl`/`strategy_upsert_dsl` returns `Invalid session_id` or `INVALID_INPUT`, retry the same call in-turn with `session_id=tool_compat_session_id` up to 4 times; do not ask user to refresh/reconnect.
- Before any `backtest_create_job`, you must call `get_symbol_data_coverage` and bound `start_date/end_date` within `metadata.available_timerange.start/end`.
- Before any `backtest_create_job`, also ensure the request stays within backend bar cap (`BACKTEST_MAX_BARS`); if it would exceed (commonly with long `1m` ranges), shorten range or raise timeframe first.
- You may call backtest tools in this stage (`backtest_create_job`, `backtest_get_job`, and analytics tools like `backtest_entry_hour_pnl_heatmap`, `backtest_exit_reason_breakdown`, `backtest_rolling_metrics`) to evaluate and iterate.
- When backtest status is `done`, emit one `backtest_charts` AGENT_UI payload with `job_id` so frontend can render charts without extra text tokens.
- If user explicitly confirms "ready to deploy", emit `<AGENT_STATE_PATCH>{"strategy_confirmed":true}</AGENT_STATE_PATCH>` so backend can advance phase automatically.
- Do not re-collect already finalized schema fields unless user requests changes.
