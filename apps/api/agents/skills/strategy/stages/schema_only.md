[STAGE_MARKER_STRATEGY_SCHEMA_ONLY]
Stage objective:
- Produce first-draft strategy DSL, validate it, and hand off `strategy_draft_id` for frontend review/confirmation.
- Output should include:
  1) concise rationale text
  2) inspect available indicators first with `get_indicator_catalog` (and `get_indicator_detail` only if you need one specific indicator contract)
  3) `strategy_validate_dsl(dsl_json, session_id=tool_compat_session_id)` when `tool_compat_session_id` is available
  4) one `strategy_ref` GenUI payload with `strategy_draft_id`
- Do not call persistence tools that require confirmed strategy ownership at this stage.
- Default behavior: do not call persistence tools at this stage.
- Exception: if user explicitly confirms immediate save/deploy, call `strategy_upsert_dsl` once and emit
  `<AGENT_STATE_PATCH>{"strategy_id":"<uuid>","strategy_confirmed":true}</AGENT_STATE_PATCH>`.
- Before drafting the DSL, prefer a broader factor search instead of defaulting to the same small indicator set every time; use `get_indicator_catalog` to see what is actually available in this runtime.
- If `[SESSION STATE].pre_strategy_instrument_data_status=download_started`, check whether local market data is ready before making data-dependent assumptions.
- Do not print the full DSL JSON in plain text unless the user explicitly asks for raw JSON.
- Do not emit `choice_prompt` GenUI unless DSL cannot be completed safely without one critical missing preference.
- Ask only the minimum follow-up needed to complete schema fields.
- In this stage, ignore update-by-id workflow instructions and do not mention internal update operation formats.
- If user asks to modify an existing saved strategy in this stage, ask for saved `strategy_id` (or ask user to confirm/save first) and stop there.
