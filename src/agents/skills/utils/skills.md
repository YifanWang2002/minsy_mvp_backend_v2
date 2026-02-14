---
skill: genui_output
description: >
  Emit structured UI payloads via <AGENT_UI_JSON> blocks.
  Used for selectable prompts and strategy draft references.
format_tag: AGENT_UI_JSON
---

# GenUI Output Format

When presenting choices to the user, emit exactly one wrapped JSON block per turn:

```
<AGENT_UI_JSON>{"type":"choice_prompt","choice_id":"<snake_case_id>","question":"<localised question>","subtitle":"<optional localised explanation for the question>","options":[{"id":"<enum_value>","label":"<localised label>","subtitle":"<optional localised explanation for this option>"}]}</AGENT_UI_JSON>
```

When handing off a validated pre-confirm strategy draft, emit:

```
<AGENT_UI_JSON>{"type":"strategy_ref","strategy_draft_id":"<uuid>","display_mode":"draft","source":"strategy_validate_dsl"}</AGENT_UI_JSON>
```

## Supported Fields

### `choice_prompt`

| Field | Required | Description |
|---|---|---|
| `type` | yes | Always `"choice_prompt"` |
| `choice_id` | yes | English snake_case identifier |
| `question` | yes | Main question text (user's language) |
| `subtitle` | optional | Explanatory sentence below the question (user's language) |
| `options[].id` | yes | English snake_case enum value |
| `options[].label` | yes | Display label (user's language) |
| `options[].subtitle` | optional | Short explanation / annotation for this option (user's language) |

### `strategy_ref`

| Field | Required | Description |
|---|---|---|
| `type` | yes | Always `"strategy_ref"` |
| `strategy_draft_id` | yes | Temporary validated strategy draft UUID |
| `display_mode` | optional | Usually `"draft"` |
| `source` | optional | Usually `"strategy_validate_dsl"` |

# Rules

- `choice_id`, option `id` values, and all JSON keys must always be **English snake_case**.
- `question`, `subtitle`, option `label`, and option `subtitle` values must be in the **user's language**.
- Include at least 2 options.
- Do **not** wrap the JSON in markdown code fences.
- Default to at most one `<AGENT_UI_JSON>` block per turn, unless the active phase/system instruction explicitly asks for multiple blocks in the same turn (e.g. chart + choice).
- In strategy pre-confirm turns, when validate returns `strategy_draft_id`, emit `strategy_ref` and avoid repeating full DSL JSON in plain text.
- In KYC phase, if the system prompt says required fields are still missing, emitting one `<AGENT_UI_JSON>` in that turn is mandatory.
- This mandatory rule applies to both the user's first message and all follow-up messages.
