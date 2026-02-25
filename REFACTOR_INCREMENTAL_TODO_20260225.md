# Refactor Incremental TODO (2026-02-25)

## 0. Scope

- Base doc: `ARCHITECTURE_REFACTOR_IMPLEMENTATION_PLAN_20260225.md`
- This round keeps **no Alembic**.
- Objective: incremental refactor with hard test gates after each step.

## 1. Execution Rules

1. One phase at a time, no parallel large refactors.
2. Each phase ends with:
   - scope tests for touched modules
   - cumulative integration tests including previous phases
3. No phase is considered done if any gate fails.
4. Keep task names and API/MCP response schema stable unless explicitly listed.
5. Commit convention:
   - code commit
   - test fix commit (if needed)
   - docs/update commit

## 2. Test Gate Profiles

## 2.1 G0 Smoke

```bash
uv run pytest tests/test_infra/test_config.py -q
uv run pytest tests/test_api/test_status.py tests/test_api/test_health.py -q
uv run pytest tests/test_mcp/test_server_domains.py -q
```

## 2.2 G1 Queue Routing

```bash
uv run pytest tests/test_infra/test_celery_limits.py -q
uv run pytest tests/test_infra/test_celery_paper_trading_queue.py -q
uv run pytest tests/test_infra/test_celery_trade_approval_queue.py -q
uv run pytest tests/test_infra/test_market_data_refresh_dedupe.py -q
```

## 2.3 G2 Domain Jobs

```bash
uv run pytest tests/test_engine/test_backtest_service.py -q
uv run pytest tests/test_engine/test_stress_job_service.py -q
uv run pytest tests/test_engine/test_market_data_sync_service.py -q
```

## 2.4 G3 API/MCP Contract

```bash
uv run pytest tests/test_mcp/test_trading_tools.py -q
uv run pytest tests/test_mcp/test_stress_tools.py -q
uv run pytest tests/test_mcp/test_market_data_sync_tools.py -q
uv run pytest tests/test_api/test_deployments_lifecycle.py -q
uv run pytest tests/test_api/test_market_data_endpoints.py -q
uv run pytest tests/test_api/test_trade_approvals_flow.py -q
```

## 2.5 G4 Cumulative Regression

```bash
uv run pytest tests/test_api -q
uv run pytest tests/test_engine -q
uv run pytest tests/test_mcp -q
uv run pytest tests/test_infra -q
```

## 2.6 G5 Final Full Regression

```bash
uv run pytest -q
uv run python scripts/openai_mcp_smoketest.py
uv run python scripts/e2e_pre_strategy_to_deployment.py
uv run python scripts/openai_mcp_trading_e2e.py
uv run python scripts/live_paper_trading_full_flow.py
```

## 3. Incremental TODO

## T0 Baseline Freeze and Guard Rails

- [ ] Record baseline branch and current test snapshot.
- [ ] Add architecture boundary check script (forbidden imports):
  - forbid `src/engine/** -> src/workers/**`
  - forbid `src/mcp/trading/tools.py -> src/api/routers/deployments.py`
- [ ] Add lightweight CI/local command to run boundary checks.

Phase gate:
- Run `G0`.
- Run boundary check script.

Done criteria:
- baseline snapshot saved
- boundary check command available

## T1 Settings Layer Split (compatibility first)

- [ ] Introduce `packages/shared_settings` skeleton.
- [ ] Keep `src/config.py` as compatibility shim during transition.
- [ ] Migrate first batch of consumers:
  - `src/main.py`
  - `src/workers/celery_app.py`
  - `src/mcp/server.py`
- [ ] Add service env validation entrypoint.

Phase gate:
- Run `G0`.
- Run `tests/test_infra/test_config.py -q`.
- Run `G4`.

Done criteria:
- no behavior change in startup paths
- env loading deterministic across api/mcp/worker

## T2 Celery App Split (cpu/io/beat)

- [ ] Add `apps/worker/cpu/celery_app.py`.
- [ ] Add `apps/worker/io/celery_app.py`.
- [ ] Add `apps/beat/celery_app.py`.
- [ ] Keep task names and queue names unchanged.
- [ ] Update `scripts/dev_up.sh` and `scripts/dev_down.sh` commands to new app entries.

Phase gate:
- Run `G0`.
- Run `G1`.
- Run `G4`.

Done criteria:
- queue ownership verified
- beat only schedules, workers execute

## T3 Publisher Ports (remove domain->worker reverse dependency)

- [ ] Add queue publisher ports in domain layer.
- [ ] Add infra publisher implementations.
- [ ] Refactor:
  - `src/engine/backtest/service.py`
  - `src/engine/stress/service.py`
  - `src/engine/market_data/sync_service.py`
- [ ] Remove direct worker imports in above services.

Phase gate:
- Run `G2`.
- Run boundary check script.
- Run `G4`.

Done criteria:
- no `src.engine.* -> src.workers.*` import in targeted services

## T4 API/MCP Decoupling

- [ ] Replace API direct enqueue with application service calls:
  - `src/api/routers/deployments.py`
  - `src/api/routers/market_data.py`
  - `src/api/routers/trade_approvals.py`
  - `src/services/telegram_service.py`
- [ ] Refactor `src/mcp/trading/tools.py`:
  - remove router internal calls
  - remove direct runtime enqueue imports
- [ ] Remove sync heavy execution paths:
  - stress MCP `run_async=false` path removed
  - market-data sync MCP `run_async=false` path removed

Phase gate:
- Run `G3`.
- Run boundary check script.
- Run `G4`.

Done criteria:
- MCP heavy tasks are always queued
- trading MCP no longer imports API router internals

## T5 Startup Ownership Hardening (no Alembic round)

- [ ] Keep API as schema owner.
- [ ] Enforce worker/mcp always `ensure_schema=False`.
- [ ] Add startup order and readiness checks in local compose/scripts.
- [ ] Add health field to expose fallback mode status for runtime state/signal storage.

Phase gate:
- Run `G0`.
- Run `G1`.
- Run `G4`.

Done criteria:
- startup order deterministic
- no schema side effects in worker/mcp startup

## T6 Containerized Local/Deploy Orchestration

- [ ] Add full-stack compose files:
  - `compose/compose.base.yml`
  - `compose/compose.dev.yml`
  - `compose/compose.prod.yml`
- [ ] Update `.github/workflows/deploy.yml` to container-oriented flow.
- [ ] Add deploy health checks:
  - API health
  - MCP gateway health
  - worker-cpu / worker-io ping

Phase gate:
- Run `G0`.
- Run `G1`.
- Run `G3`.
- Run `G4`.

Done criteria:
- one command up/down in local
- deploy workflow matches service split

## T7 Final Consolidation

- [ ] Update architecture/readme docs with final commands.
- [ ] Run full regression and E2E scripts.
- [ ] Produce migration notes and rollback notes.

Phase gate:
- Run `G5`.

Done criteria:
- all tests and key E2E green
- docs and runbooks aligned with actual code

## 4. Rollback Strategy

1. Every phase must be a separate commit group.
2. If phase gate fails, rollback only current phase commits.
3. Keep compatibility shims until T7 completion.

## 5. Current Status

- [x] TODO defined
- [ ] T0 started
- [ ] T1 done
- [ ] T2 done
- [ ] T3 done
- [ ] T4 done
- [ ] T5 done
- [ ] T6 done
- [ ] T7 done

