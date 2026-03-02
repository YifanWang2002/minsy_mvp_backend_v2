# Deployment Broker Preflight Implementation Plan

## Goal

Refactor deployment-phase broker handling from a prompt-only, tool-error-driven flow into a backend-enforced preflight flow that:

- exposes broker inventory and compatibility to the AI before deployment
- supports creating the built-in sandbox broker from deployment chat
- requires broker selection and deployment confirmation before deployment tools are available
- emits a structured deployment summary before execution

## Delivery Order

1. Persist shared broker account operations outside `apps.api.routes` so MCP no longer imports API routes.
2. Add broker capability matching helpers and broker preflight MCP tools.
3. Update deployment-phase runtime policy so tools are gated by readiness/confirmation state.
4. Update deployment handler + deployment skill instructions for:
   - broker selection
   - built-in sandbox creation
   - deployment summary
   - confirmation before execute
5. Add API, MCP, and deployment-phase unit tests.

## Key Constraints

- No `apps.mcp -> apps.api.routes` imports.
- No SQLAlchemy usage in `packages/domain`.
- Deployment phase should no longer auto-bounce back to `strategy` only because broker readiness is blocked.
- If multiple compatible brokers exist, AI must ask the user to choose.
- If only one compatible broker exists, backend may auto-select it.

## Primary Files

- `packages/infra/trading/broker_account_store.py`
- `packages/domain/trading/broker_capability_policy.py`
- `apps/mcp/domains/trading/tools.py`
- `apps/api/orchestration/constants.py`
- `apps/api/orchestration/prompt_builder.py`
- `apps/api/orchestration/core.py`
- `apps/api/agents/handlers/deployment_handler.py`
- `apps/api/agents/handlers/strategy_handler.py`
- `apps/api/agents/skills/deployment_skills.py`
- `apps/api/agents/skills/deployment/skills.md`

## Test Coverage Targets

- API:
  - built-in sandbox create
  - built-in sandbox deactivate
  - built-in sandbox re-activate
- MCP:
  - list broker accounts
  - check deployment readiness
  - create built-in sandbox broker
- Deployment phase:
  - preflight tools only before broker/confirmation
  - execute tools only after confirmation
  - fallback choice prompt for multi-broker selection
  - fallback choice prompt for deployment confirmation
