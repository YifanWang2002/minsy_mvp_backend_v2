[STAGE_MARKER_STRATEGY_ARTIFACT_OPS]
Stage objective:
- Assume a strategy artifact already exists and focus on artifact operations.
- For update-by-id, always do:
  1) `strategy_validate_dsl` on edited JSON
  2) `strategy_upsert_dsl` with existing `strategy_id`
- Do not call backtest tools in strategy phase.
- Do not re-collect already finalized schema fields unless user requests changes.
