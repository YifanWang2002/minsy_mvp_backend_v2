---
skill: genui_output
description: >
  Emit structured UI choice prompts to the user via <AGENT_UI_JSON> blocks.
  Used whenever the agent needs to present selectable options.
format_tag: AGENT_UI_JSON
---

# GenUI Output Format

When presenting choices to the user, emit exactly one wrapped JSON block per turn:

```
<AGENT_UI_JSON>{"type":"choice_prompt","choice_id":"<snake_case_id>","question":"<localised question>","subtitle":"<optional localised explanation for the question>","options":[{"id":"<enum_value>","label":"<localised label>","subtitle":"<optional localised explanation for this option>"}]}</AGENT_UI_JSON>
```

## Supported Fields

| Field | Required | Description |
|---|---|---|
| `type` | yes | Always `"choice_prompt"` |
| `choice_id` | yes | English snake_case identifier |
| `question` | yes | Main question text (user's language) |
| `subtitle` | optional | Explanatory sentence below the question (user's language) |
| `options[].id` | yes | English snake_case enum value |
| `options[].label` | yes | Display label (user's language) |
| `options[].subtitle` | optional | Short explanation / annotation for this option (user's language) |

# Rules

- `choice_id`, option `id` values, and all JSON keys must always be **English snake_case**.
- `question`, `subtitle`, option `label`, and option `subtitle` values must be in the **user's language**.
- Include at least 2 options.
- Do **not** wrap the JSON in markdown code fences.
- Default to at most one `<AGENT_UI_JSON>` block per turn, unless the active phase/system instruction explicitly asks for multiple blocks in the same turn (e.g. chart + choice).
- In KYC phase, if the system prompt says required fields are still missing, emitting one `<AGENT_UI_JSON>` in that turn is mandatory.
- This mandatory rule applies to both the user's first message and all follow-up messages.
