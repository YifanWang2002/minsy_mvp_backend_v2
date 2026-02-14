---
skill: strategy_patch_workflow
description: >
  Use RFC 6902 JSON Patch for minimal strategy updates after a strategy_id already exists.
---

# Strategy Patch Workflow

Use this workflow when the user asks to revise an existing strategy and a `strategy_id` is already known.

## Goal
- Send only minimal changed fields via patch operations.
- Reduce token usage and avoid full-JSON regeneration mistakes.
- Keep updates deterministic and version-safe.

## Tool Sequence
1. `strategy_get_dsl(session_id, strategy_id)` to fetch latest `dsl_json` and `metadata.version`.
2. Build RFC 6902 patch operations (JSON array of op objects).
3. `strategy_patch_dsl(session_id, strategy_id, patch_json, expected_version)`.

## Version History Helpers
- `strategy_list_versions(session_id, strategy_id, limit)` returns latest revision metadata.
- `strategy_get_version_dsl(session_id, strategy_id, version)` returns a specific historical DSL.
- `strategy_diff_versions(session_id, strategy_id, from_version, to_version)` returns RFC 6902 patch ops.
- `strategy_rollback_dsl(session_id, strategy_id, target_version, expected_version)` restores by creating a new latest version.

## Patch Rules
- Prefer `replace` for value changes.
- Use `add` for new fields or list append (`/path/-`).
- Use `remove` only when user explicitly asks to delete behavior.
- Use `test` before `replace/remove` when correctness matters (guard against stale assumptions).
- Keep operation count as small as possible.
- Patch path root is the DSL object itself. Use `/trade/...`, `/factors/...`, `/timeframe`, etc.
- Never prefix paths with `/dsl_json`.

## Examples

### Replace one parameter
```json
[
  {"op":"test","path":"/trade/long/exits/1/stop/value","value":0.02},
  {"op":"replace","path":"/trade/long/exits/1/stop/value","value":0.015}
]
```

### Append one condition
```json
[
  {"op":"add","path":"/trade/long/entry/condition/all/-","value":{"cmp":{"left":{"ref":"rsi_14"},"op":"lt","right":65}}}
]
```

### Remove one side
```json
[
  {"op":"remove","path":"/trade/short"}
]
```

## Error Handling
- `STRATEGY_VERSION_CONFLICT`: re-run `strategy_get_dsl`, regenerate patch on latest version, retry.
- `STRATEGY_PATCH_APPLY_FAILED`: fix invalid path/op and retry.
- `STRATEGY_VALIDATION_FAILED`: patch produced invalid DSL; apply minimal corrective patch.
