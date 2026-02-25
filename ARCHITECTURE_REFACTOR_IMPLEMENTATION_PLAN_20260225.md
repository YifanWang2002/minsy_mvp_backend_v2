# Minsy Backend Refactor Implementation Plan (Single Repo, 6-7 Containers)

## 0) Context and Scope

- Date: 2026-02-25
- Working branch: `refactor/arch-split-plan-20260225`
- Repository: `backend/`
- Goal: reduce coupling and startup complexity, split runtime into medium-grain services, and keep migration feasible for a solo developer.
- Hard constraints:
  - Keep single repository.
  - Target 6-7 runtime containers (exclude optional observability tools).
  - MCP servers remain IO/tool facade; CPU-heavy jobs must execute in Celery workers.
  - Preserve existing product behavior (KYC -> strategy -> backtest -> deployment -> runtime -> approvals/notifications).

## 1) Source-Code Audit Baseline (from current code)

### 1.1 Inventory snapshot

- Python source files under `src/`: 229
- Test files under `tests/`: 130
- `src/config.py`: 1142 LOC
- Direct `from src.config import settings` usage:
  - `src`: 49 files
  - `src + tests`: 76 files
- `.env.example` keys: 152
- `Settings` aliases in `src/config.py`: 173

### 1.2 Confirmed coupling and debt (with exact anchors)

1. MCP trading directly calls API router internals and worker enqueue.
- `src/mcp/trading/tools.py:14` imports `src.api.routers.deployments`.
- `src/mcp/trading/tools.py:407`, `:417`, `:422`, `:510`, `:567`, `:572` call router internal functions.
- `src/mcp/trading/tools.py:30`, `:431`, `:580` enqueue runtime task directly.

2. API/services directly enqueue worker tasks.
- `src/api/routers/deployments.py:54`, `:744-746`
- `src/api/routers/market_data.py:24`, `:104`, `:158`
- `src/api/routers/trade_approvals.py` decision path enqueues task after approve.
- `src/services/telegram_service.py:18`, `:260`

3. Domain/engine layer depends on workers (reverse dependency).
- `src/engine/backtest/service.py:156-159`
- `src/engine/stress/service.py:200-203`
- `src/engine/market_data/sync_service.py:177-180`

4. Celery app is an all-in-one topology (include + routes + beat in one place).
- `src/workers/celery_app.py:71-83`, `:118`, `:128-141`

5. Orchestrator hard-couples MCP URLs and context-token contract.
- MCP URL injection: `src/agents/orchestrator/prompt_builder.py:237`, `:247`, `:257`, `:267`
- Context token injection: `src/agents/orchestrator/prompt_builder.py:303-310`, `:328`
- MCP token decode shared secret: `src/mcp/context_auth.py:83`, `:96`

6. Runtime schema ownership is implicit in application startup.
- `src/models/database.py:212-233` runs `create_all` and runtime DDL patching in process.

7. API startup performs local data bootstrap.
- `src/main.py:30` calls `ensure_market_data()` during app lifespan.

8. In-process state split risk across containers.
- Market runtime ring/cache lives in process memory: `src/engine/market_data/runtime.py:84-87`, `:288-303`
- Redis unavailable fallback to in-memory in runtime state/signal stores:
  - `src/engine/execution/runtime_state_store.py:16-18`, `:82-94`, `:108-114`
  - `src/engine/execution/signal_store.py:21-23`, `:133-137`

9. Additional coupling discovered during audit.
- API has sync backtest execution path in request lifecycle:
  - `src/api/routers/backtests.py:160-169` (`execute_backtest_job` inline)
- MCP stress and market-data tools still allow sync execution branch (`run_async=False`):
  - `src/mcp/stress/tools.py:171-176`
  - `src/mcp/market_data/tools.py:1171-1176`

### 1.3 Env/config debt quantified

- `src/config.py` defines 173 env aliases.
- `.env.example` misses 26 config aliases.
- Split examples (`env/.env.common.example`, `.env.dev.example`, `.env.prod.example`) currently cover only 70 unique keys and miss 103 aliases.
- `.env.example` includes extra keys not in config schema (`MCP_PROXY_UPSTREAM_*`).

## 2) Target Runtime Architecture (6-7 containers)

## 2.1 Recommended target (7 containers)

1. `postgres`
2. `redis`
3. `api-server`
4. `mcp-server`
5. `worker-cpu`
6. `worker-io`
7. `scheduler-beat`

Optional profile only: `flower` (not counted in base topology).

### 2.2 Minimal target (6 containers)

- Same as above, but merge beat into `worker-io` process (`celery worker -B`) for local-only simplicity.
- Production should keep beat separate for resilience and clearer resource control.

### 2.3 Queue and resource mapping

- `worker-cpu` queues:
  - `backtest`
  - `stress`
- `worker-io` queues:
  - `paper_trading`
  - `market_data`
  - `trade_approval`
  - `notifications`
  - `maintenance`
- `scheduler-beat`:
  - emits scheduled ticks only; does not execute business logic.

## 3) Target Codebase Layout (Single Repo)

```text
backend/
  apps/
    api/
      main.py
      router.py
      routes/
      orchestration/
    mcp/
      gateway.py
      domains/
        strategy/
        backtest/
        market_data/
        stress/
        trading/
      auth/
    worker/
      cpu/
        celery_app.py
        tasks/
      io/
        celery_app.py
        tasks/
      common/
        celery_base.py
        task_registry.py
    beat/
      celery_app.py
      schedule.py
  packages/
    core/
      errors/
      result/
      contracts/
      events/
      ids/
    domain/
      strategy/
      backtest/
      stress/
      trading/
      market_data/
      session/
      notification/
      ports/
      services/
    infra/
      db/
        models/
        session.py
        repositories/
      redis/
        client.py
        stores/
      queue/
        publishers.py
        consumers.py
      providers/
        alpaca/
        ccxt/
        yfinance/
        openai/
        telegram/
      observability/
        logger.py
        sentry.py
    shared_settings/
      schema/
      loader/
      service_settings/
      profile/
  docker/
    backend.Dockerfile
    entrypoints/
      api.sh
      mcp.sh
      worker_cpu.sh
      worker_io.sh
      beat.sh
  compose/
    compose.base.yml
    compose.dev.yml
    compose.prod.yml
  alembic.ini
  migrations/
```

## 4) Layer Boundaries and Dependency Rules

## 4.1 Package responsibilities

- `core`
  - Put: domain-neutral types, error codes, result wrappers, contracts.
  - Do not put: FastAPI/Celery/SQLAlchemy/Redis/OpenAI SDK imports.
- `domain`
  - Put: business rules and use-cases for strategy/backtest/stress/trading/market-data.
  - Do not put: HTTP routing, ORM session wiring, queue transport calls.
- `infra`
  - Put: DB models/repositories, Redis stores, queue implementations, external adapters.
  - Do not put: session orchestration state machine logic.
- `shared_settings`
  - Put: settings schema, env/profile loaders, per-service settings objects.
  - Do not put: business logic.

## 4.2 Hard import rule

- `apps/*` can import `packages/*`.
- `packages/domain` can import `packages/core` only.
- `packages/infra` can import `packages/core` and `packages/domain` ports/contracts.
- `packages/domain` must never import `apps/*` or Celery task modules.

## 5) Decoupling Plan for Current Hotspots

## 5.1 Replace direct enqueue with publisher ports

Introduce domain ports:
- `BacktestJobPublisherPort`
- `StressJobPublisherPort`
- `MarketDataSyncJobPublisherPort`
- `RuntimeTaskPublisherPort`
- `TradeApprovalTaskPublisherPort`

Implementation:
- `packages/infra/queue/publishers.py` implements ports using Celery.
- API and MCP call domain services, domain services call publisher ports.
- Remove worker imports from domain/API/MCP modules.

Directly impacted files:
- `src/engine/backtest/service.py`
- `src/engine/stress/service.py`
- `src/engine/market_data/sync_service.py`
- `src/api/routers/deployments.py`
- `src/api/routers/market_data.py`
- `src/api/routers/trade_approvals.py`
- `src/services/telegram_service.py`
- `src/mcp/trading/tools.py`

## 5.2 MCP should be IO facade only

- Keep DB reads/writes and validation in MCP if lightweight.
- Force heavy work to queue:
  - backtest: always queued
  - stress: always queued (remove sync branch)
  - market-data sync: always queued (remove sync branch)
- Remove API router internal calls from MCP trading tools; call domain/application service API instead.

Directly impacted files:
- `src/mcp/trading/tools.py`
- `src/mcp/stress/tools.py`
- `src/mcp/market_data/tools.py`
- `src/mcp/server.py`

## 5.3 Split Celery topology

Current single app split into:
- `apps/worker/cpu/celery_app.py`
- `apps/worker/io/celery_app.py`
- `apps/beat/celery_app.py`

Refactor outputs:
- Shared queue declarations in `apps/worker/common/celery_base.py`.
- Beat schedule in `apps/beat/schedule.py` only.
- Worker apps include only task modules they execute.

Directly impacted files:
- `src/workers/celery_app.py`
- all `src/workers/*_tasks.py`
- scripts and service startup commands.

## 5.4 Settings split and env normalization

Replace monolithic settings with:
- `CommonSettings` (cross-service)
- `ApiSettings`
- `McpSettings`
- `WorkerCpuSettings`
- `WorkerIoSettings`
- `BeatSettings`

Add service env files:
- `env/.env.common`
- `env/.env.api`
- `env/.env.mcp`
- `env/.env.worker_cpu`
- `env/.env.worker_io`
- `env/.env.beat`

Migration behavior:
- keep backward-compatible aliases for 1 release window.
- emit startup warning for deprecated keys.
- fail fast for missing required keys.

Directly impacted files:
- `src/config.py` (to be split)
- all settings consumers currently importing `settings`.

## 5.5 Runtime schema ownership (this refactor round: no Alembic)

- This round defers Alembic to reduce scope and delivery risk.
- Keep current runtime schema path temporarily, but make ownership explicit:
  - API startup is the only schema owner (`init_postgres(ensure_schema=True)`).
  - Worker/MCP startup must only use `ensure_schema=False`.
- Deployment/startup order must ensure API is healthy before starting worker/MCP.
- Add backlog item: migrate to Alembic in a later iteration after service split stabilizes.

Directly impacted files:
- `src/models/database.py`
- `src/main.py`
- worker/MCP startup scripts and compose dependency order.

## 5.6 Data bootstrap decoupling

- Remove `ensure_market_data()` from API lifespan.
- Add explicit data bootstrap job/script:
  - local: `scripts/data_bootstrap.sh`
  - deploy: one-off init job (or manual step)
- API only validates data readiness and returns health warnings.

Directly impacted files:
- `src/main.py`
- `src/util/data_setup.py`
- deployment scripts/compose.

## 5.7 In-memory fallback controls for multi-container reliability

- Default production behavior: disable in-memory fallbacks for runtime state/signal store.
- Keep fallback only for tests/local profile.
- Expose health endpoint fields to surface fallback activation.

Directly impacted files:
- `src/engine/execution/runtime_state_store.py`
- `src/engine/execution/signal_store.py`
- `src/api/routers/health.py`

## 6) Migration Map: Current -> Target Modules

| Current module/file | Target location | Migration action |
|---|---|---|
| `src/config.py` | `packages/shared_settings/*` | Split into schema + loader + service-specific settings |
| `src/models/database.py` | `packages/infra/db/session.py` | Remove runtime DDL; keep connection/session bootstrap |
| `src/models/*` | `packages/infra/db/models/*` | Move ORM models under infra |
| `src/models/redis.py` | `packages/infra/redis/client.py` | Keep pool/client lifecycle |
| `src/workers/celery_app.py` | `apps/worker/*/celery_app.py`, `apps/beat/*` | Split cpu/io/beat apps and routes |
| `src/workers/backtest_tasks.py` | `apps/worker/cpu/tasks/backtest.py` | CPU queue consumer |
| `src/workers/stress_tasks.py` | `apps/worker/cpu/tasks/stress.py` | CPU queue consumer |
| `src/workers/market_data_tasks.py` | `apps/worker/io/tasks/market_data.py` | IO queue consumer (or isolate sync queue if needed) |
| `src/workers/paper_trading_tasks.py` | `apps/worker/io/tasks/paper_trading.py` | IO queue consumer |
| `src/workers/trade_approval_tasks.py` | `apps/worker/io/tasks/trade_approval.py` | IO queue consumer |
| `src/workers/notification_tasks.py` | `apps/worker/io/tasks/notifications.py` | IO queue consumer |
| `src/workers/maintenance_tasks.py` | `apps/worker/io/tasks/maintenance.py` | IO queue consumer |
| `src/engine/backtest/service.py` | `packages/domain/backtest/service.py` | Remove worker imports; use publisher port |
| `src/engine/stress/service.py` | `packages/domain/stress/service.py` | Remove worker imports; use publisher port |
| `src/engine/market_data/sync_service.py` | `packages/domain/market_data/sync_service.py` | Remove worker imports; use publisher port |
| `src/mcp/trading/tools.py` | `apps/mcp/domains/trading/tools.py` | Remove router internal calls; call domain app service |
| `src/mcp/stress/tools.py` | `apps/mcp/domains/stress/tools.py` | Remove sync execution path |
| `src/mcp/market_data/tools.py` | `apps/mcp/domains/market_data/tools.py` | Remove sync execution path |
| `src/mcp/server.py` | `apps/mcp/gateway.py` | One process gateway for domain paths |
| `src/main.py` | `apps/api/main.py` | Remove data bootstrap and schema side effects |
| `src/api/routers/*` | `apps/api/routes/*` | Keep API layer thin; call domain/application services |
| `src/agents/orchestrator/*` | `apps/api/orchestration/*` | Keep in app layer, consume shared settings/contracts |

## 7) Container Build and Runtime Plan

## 7.1 Single runtime image reused by all backend app containers

- Build one backend runtime image with shared packages installed once.
- Containers differ by command only.

Example commands:
- `api-server`: `uv run uvicorn apps.api.main:app --host 0.0.0.0 --port 8000`
- `mcp-server`: `uv run python -m apps.mcp.gateway --host 0.0.0.0 --port 8110`
- `worker-cpu`: `uv run celery -A apps.worker.cpu.celery_app:celery_app worker -Q backtest,stress --concurrency=<cpu>`
- `worker-io`: `uv run celery -A apps.worker.io.celery_app:celery_app worker -Q paper_trading,market_data,trade_approval,notifications,maintenance --concurrency=<io>`
- `scheduler-beat`: `uv run celery -A apps.beat.celery_app:celery_app beat -l info`

## 7.2 Compose files

- `compose/compose.base.yml`
  - postgres, redis, shared networks/volumes.
- `compose/compose.dev.yml`
  - adds api/mcp/worker-cpu/worker-io/beat with bind mounts and optional reload.
- `compose/compose.prod.yml`
  - pinned image tags, restart policies, resource limits, health checks.

## 7.3 K8s-ready conventions (future-proof)

- one Deployment per runtime service (`api`, `mcp`, `worker-cpu`, `worker-io`, `beat`).
- independent HPA for `worker-cpu` and `worker-io`.
- queue names stable and explicit for autoscaling metrics.
- liveness/readiness endpoints for each service.

## 7.4 GitHub Actions `deploy.yml` changes after refactor

Target: align CI/CD with 7-container runtime and keep no-Alembic policy in this round.

Required changes in `.github/workflows/deploy.yml`:

1. Keep pre-deploy backup step (`pg_dump` + user export), then deploy via compose.
2. Replace systemd multi-service restart with compose orchestration:
  - `docker compose -f compose/compose.prod.yml pull`
  - `docker compose -f compose/compose.prod.yml up -d --remove-orphans`
3. Health checks should verify:
  - API: `http://127.0.0.1:8000/api/v1/health`
  - MCP gateway: `http://127.0.0.1:8110/<domain>/mcp` (200/406 accepted)
  - Celery workers: `celery inspect ping` in `worker-cpu` and `worker-io` containers
4. Add explicit startup dependency gate:
  - wait for API ready first
  - then verify worker and mcp health
5. Do not add Alembic/migration command in this round.

Recommended deploy block shape:

```bash
git fetch --prune origin main
git reset --hard origin/main

uv sync --frozen

docker compose -f compose/compose.prod.yml pull
docker compose -f compose/compose.prod.yml up -d --remove-orphans

# health checks
curl -fsS http://127.0.0.1:8000/api/v1/health
curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8110/strategy/mcp
docker compose -f compose/compose.prod.yml exec -T worker-cpu \
  celery -A apps.worker.cpu.celery_app:celery_app inspect ping
docker compose -f compose/compose.prod.yml exec -T worker-io \
  celery -A apps.worker.io.celery_app:celery_app inspect ping
```

## 8) Local Developer Experience

## 8.1 Replace script-driven multi-process startup with compose-first

Current state:
- `compose.dev.yml` starts infra only.
- business processes started by `scripts/dev_up.sh` via `nohup`.

Target state:
- compose starts full stack in one command.
- keep helper script as wrapper only:
  - `scripts/dev_up.sh` -> `docker compose -f compose/compose.dev.yml up -d`
  - `scripts/dev_down.sh` -> `docker compose ... down`

## 8.2 Deterministic env rendering

- Keep `scripts/render_env.sh`, but output per-service env artifacts.
- Add validation command:
  - `uv run python -m packages.shared_settings.validate_env --service api`
  - same for mcp/worker-cpu/worker-io/beat.

## 9) Phased Execution Plan (Solo-dev safe)

## Phase 0 - Baseline freeze and safety rails

- Rotate secrets currently exposed in local `.env` and external providers.
- Add architecture guard tests (forbidden import checks).
- Add queue contract snapshot tests.

Exit criteria:
- baseline tests pass.
- no behavior change yet.

## Phase 1 - Settings split (no behavior change)

- Introduce `packages/shared_settings` and service settings objects.
- Keep compatibility shim `src/config.py` re-exporting new settings during transition.
- Fill missing env keys and deprecate extras.

Exit criteria:
- all services start with new settings loader.
- tests in `tests/test_infra/test_config.py` pass with new layout.

## Phase 2 - Celery topology split

- Split celery apps into cpu/io/beat.
- Keep task names unchanged.
- Update worker startup scripts/compose.

Exit criteria:
- each queue consumed by intended worker only.
- no task loss in smoke tests.

## Phase 3 - Domain enqueue decoupling

- Introduce publisher ports and infra implementations.
- remove worker imports from domain services.

Exit criteria:
- forbidden imports removed.
- domain services unit tests pass with fake publisher.

## Phase 4 - API and MCP decoupling

- API routes call application services, not worker enqueue functions.
- MCP trading tools stop calling API router internals.
- remove sync execution paths in MCP stress/market sync.

Exit criteria:
- MCP tools still return same response schema.
- heavy tasks always queued.

## Phase 5 (deferred) - Alembic migration ownership move

- Out of scope for current refactor round.
- Future work:
  - add Alembic, generate baseline and delta migrations.
  - remove runtime DDL from app startup.

Exit criteria (future phase):
- clean DB bootstrap via migration command only.
- API/worker/mcp startup is schema-side-effect free.

## Phase 6 - Data bootstrap and runtime hardening

- remove `ensure_market_data()` from API startup.
- add explicit data bootstrap workflow.
- production profile disables in-memory fallback paths.

Exit criteria:
- API start time is predictable.
- runtime state consistency no longer depends on single process memory.

## 10) Exact New/Changed Infra Files to Add

## 10.1 New files

- `apps/api/main.py`
- `apps/mcp/gateway.py`
- `apps/worker/common/celery_base.py`
- `apps/worker/cpu/celery_app.py`
- `apps/worker/io/celery_app.py`
- `apps/beat/celery_app.py`
- `packages/shared_settings/schema/*.py`
- `packages/shared_settings/loader/*.py`
- `packages/domain/*/ports/*.py`
- `packages/infra/queue/publishers.py`
- `docker/backend.Dockerfile`
- `docker/entrypoints/{api,mcp,worker_cpu,worker_io,beat}.sh`
- `compose/compose.base.yml`
- `compose/compose.dev.yml`
- `compose/compose.prod.yml`
- `scripts/data_bootstrap.sh`
- `.github/workflows/deploy.yml` (service orchestration and health checks update)

## 10.2 Files to refactor in-place

- `src/config.py` (transition shim then deprecate)
- `src/models/database.py`
- `src/workers/celery_app.py`
- `src/workers/*_tasks.py`
- `src/main.py`
- `src/mcp/trading/tools.py`
- `src/mcp/stress/tools.py`
- `src/mcp/market_data/tools.py`
- `src/engine/backtest/service.py`
- `src/engine/stress/service.py`
- `src/engine/market_data/sync_service.py`
- `src/api/routers/deployments.py`
- `src/api/routers/market_data.py`
- `src/api/routers/trade_approvals.py`
- `src/services/telegram_service.py`

## 11) Acceptance Checklist (feature-complete)

Every item below must pass before refactor is accepted.

### 11.1 Functional acceptance by business flow

1. Auth and account lifecycle
- register/login/refresh/me/password flows pass.

2. Session and KYC flow
- KYC progression and phase transitions preserved.

3. Strategy DSL lifecycle
- generate/revise/confirm strategy.
- strategy versioning/diff/list/detail unchanged.

4. Backtest flow
- create job, queue execution, fetch result/analysis.
- no CPU-heavy backtest execution in API/MCP process.

5. Stress flow
- create stress jobs, queue execution, fetch results/pareto.
- no sync stress execution path exposed.

6. Deployment and runtime
- create/start/pause/stop deployment.
- runtime loop, signals, orders, positions, pnl endpoints all working.

7. Trade approvals
- approve/reject via API and Telegram callback.
- approved request enqueues and executes exactly once.

8. Market data
- quote/bars/subscriptions/checkpoints/health endpoints working.
- missing-range sync always queued.

9. Notifications and social connectors
- outbox dispatch and Telegram notifications continue to work.

10. Chat orchestration and MCP calls
- phase-based tool policy still enforced.
- MCP context token signed/verified across api/mcp boundary.

### 11.2 Test suites required green

Run all:

```bash
uv run pytest tests/test_api -q
uv run pytest tests/test_engine -q
uv run pytest tests/test_mcp -q
uv run pytest tests/test_infra -q
uv run pytest tests/test_models -q
uv run pytest tests/test_agents -q
uv run pytest tests/test_services -q
```

Then run full:

```bash
uv run pytest -q
```

### 11.3 Integration and E2E scripts required

```bash
uv run python scripts/openai_mcp_smoketest.py
uv run python scripts/e2e_pre_strategy_to_deployment.py
uv run python scripts/openai_mcp_trading_e2e.py
uv run python scripts/live_paper_trading_full_flow.py
```

### 11.4 Deployment acceptance

- compose up/down starts/stops all target services cleanly.
- health checks for api/mcp/workers/beat pass.
- queue routing validated:
  - cpu tasks never land in io worker.
  - io tasks never block cpu worker queue.

## 12) Risk Register and Mitigation

1. Risk: migration touches many imports at once.
- Mitigation: compatibility shim phase and forbidden-import CI checks.

2. Risk: queue split causes lost routing.
- Mitigation: task-route snapshot tests + per-queue smoke tests.

3. Risk: keeping runtime DDL (while Alembic deferred) can still create startup-order coupling.
- Mitigation: enforce API-first startup and disallow schema mutation in worker/MCP paths.

4. Risk: MCP behavior drift during decoupling.
- Mitigation: keep tool response schema frozen and add MCP contract tests.

5. Risk: environment drift between services.
- Mitigation: per-service env validation command and startup fail-fast.

## 13) Deliverables at "Done" State

1. New app/package layout in repo with strict import boundaries.
2. Compose-based 6-7 service runtime (single command startup).
3. Settings split with per-service env files and validators.
4. Celery split into `worker-cpu`, `worker-io`, `beat`.
5. MCP heavy operations queue-only, no sync execution for stress/market sync.
6. API/MCP/domain no direct worker or router-internal coupling.
7. CI/CD (`deploy.yml`) switched to container-oriented deployment checks.
8. All existing tests and E2E flows green.
