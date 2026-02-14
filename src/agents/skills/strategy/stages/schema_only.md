[STAGE_MARKER_STRATEGY_SCHEMA_ONLY]
Stage objective:
- Produce first-draft strategy DSL for frontend review/confirmation.
- Output should include:
  1) concise rationale text
  2) one complete JSON DSL object
- Do not call persistence tools that require confirmed strategy ownership at this stage.
- Do not emit `choice_prompt` GenUI unless DSL cannot be completed safely without one critical missing preference.
- Ask only the minimum follow-up needed to complete schema fields.
