[STAGE_MARKER_STRATEGY_ARTIFACT_OPS]
Stage objective:
- Assume a strategy artifact already exists and focus on artifact operations.
- For update-by-id, prefer:
  1) `strategy_get_dsl` (latest payload + version)
  2) `strategy_patch_dsl` with minimal patch ops and `expected_version`
- For history/compare requests, use:
  1) `strategy_list_versions`
  2) `strategy_diff_versions` and/or `strategy_get_version_dsl`
- For rollback requests, use:
  1) `strategy_rollback_dsl` with `target_version` and current `expected_version`
- If patch path is not suitable for the requested change, fallback to:
  1) `strategy_validate_dsl` on edited JSON
  2) `strategy_upsert_dsl` with existing `strategy_id`
- Do not call backtest tools in strategy phase.
- Do not re-collect already finalized schema fields unless user requests changes.
