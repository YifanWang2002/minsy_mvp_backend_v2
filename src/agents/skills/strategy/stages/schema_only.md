[STAGE_MARKER_STRATEGY_SCHEMA_ONLY]
Stage objective:
- Produce first-draft strategy DSL, validate it, and hand off `strategy_draft_id` for frontend review/confirmation.
- Output should include:
  1) concise rationale text
  2) `strategy_validate_dsl(dsl_json)`
  3) one `strategy_ref` GenUI payload with `strategy_draft_id`
- Do not call persistence tools that require confirmed strategy ownership at this stage.
- Do not print the full DSL JSON in plain text unless the user explicitly asks for raw JSON.
- Do not emit `choice_prompt` GenUI unless DSL cannot be completed safely without one critical missing preference.
- Ask only the minimum follow-up needed to complete schema fields.
- In this stage, ignore update-by-id workflow instructions and do not mention internal update operation formats.
- If user asks to modify an existing saved strategy in this stage, ask for saved `strategy_id` (or ask user to confirm/save first) and stop there.
