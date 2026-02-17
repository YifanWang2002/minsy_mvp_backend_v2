---
skill: strategy_patch_workflow
description: >
  Apply minimal field-level updates for existing strategies after a strategy_id is known.
---

# Strategy Patch Workflow

Use this workflow when the user asks to revise an existing strategy and a `strategy_id` is already known.

## User-Facing Guardrail
- This workflow is internal for tool calls.
- Never expose raw operation arrays or JSON paths in normal user replies.
- Never ask the user to manually apply backend update payloads.
- If tool execution fails, ask for retry/context confirmation only; do not ask the user to manually edit strategy values.
- Summarize changes in plain language (what changed, why, expected impact).

## Goal
- Send only minimal changed fields via patch operations.
- Reduce token usage and avoid full-JSON regeneration mistakes.
- Keep updates deterministic and version-safe.

## Tool Sequence
1. `strategy_get_dsl(strategy_id)` to fetch latest `dsl_json` and `metadata.version`.
2. Build minimal update operations only for fields that changed.
3. `strategy_patch_dsl(strategy_id, <update_ops>, expected_version)`.

## Version History Helpers
- `strategy_list_versions(strategy_id, limit)` returns latest revision metadata.
- `strategy_get_version_dsl(strategy_id, version)` returns a specific historical DSL.
- `strategy_diff_versions(strategy_id, from_version, to_version)` returns structured update operations between versions.
- `strategy_rollback_dsl(strategy_id, target_version, expected_version)` restores by creating a new latest version.

## Patch Rules
- Prefer `replace` for value changes.
- Use `add` for new fields or list append (`/path/-`).
- Use `remove` only when user explicitly asks to delete behavior.
- Use `test` before `replace/remove` when correctness matters (guard against stale assumptions).
- Keep operation count as small as possible.
- Patch path root is the DSL object itself. Use `/trade/...`, `/factors/...`, `/timeframe`, etc.
- Never prefix paths with `/dsl_json`.

## Error Handling
- For transient tool failures (`http_error`, status 424/5xx, `Session terminated`), retry the same tool call up to 4 times before reporting failure.
- `STRATEGY_VERSION_CONFLICT`: re-run `strategy_get_dsl`, regenerate patch on latest version, retry.
- `STRATEGY_PATCH_APPLY_FAILED`: fix invalid path/op and retry.
- `STRATEGY_VALIDATION_FAILED`: patch produced invalid DSL; apply minimal corrective patch.
- For `STRATEGY_VALIDATION_FAILED`, read structured `errors[]` and use `code/path/suggestion` to build one minimal corrective patch.
- In one assistant turn, for the same patch intent, run at most 2 patch-validation attempts. If still failing, stop retries and ask one focused clarification.
- Never re-submit an unchanged patch payload after validation failure.
