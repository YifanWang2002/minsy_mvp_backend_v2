# CLAUDE.md — Minsy Backend

## Project Overview

Minsy is an AI-powered quantitative trading platform. The backend is a **Python 3.12+** application composed of four services: a FastAPI REST API, an MCP (Model Context Protocol) tool server, Celery workers (CPU + IO), and a Celery Beat scheduler. It powers the full workflow: KYC → Strategy creation → Backtesting → Deployment → Live monitoring.

## Tech Stack

- **Language:** Python 3.12+
- **Web Framework:** FastAPI 0.115.0+
- **ORM:** SQLAlchemy 2.0.35+ (async, asyncpg)
- **Database:** PostgreSQL
- **Cache/Broker:** Redis 5.2.0
- **Task Queue:** Celery 5.4.0
- **AI Integration:** OpenAI SDK 1.50.0+
- **MCP:** Model Context Protocol 1.0.0+ (tool server)
- **Validation:** Pydantic 2.9.0+
- **Auth:** JWT (PyJWT, HS256)
- **Billing:** Stripe
- **Observability:** Sentry
- **Package Manager:** uv (Astral)
- **Linter/Formatter:** Ruff
- **Containerization:** Docker + Docker Compose

## Project Structure

```
backend/
├── apps/                          # Application entry points
│   ├── api/                       # FastAPI HTTP server (port 8000)
│   │   ├── main.py                # App factory, middleware, lifespan
│   │   ├── routes/                # 19 endpoint modules
│   │   │   ├── auth.py            # Login, registration, token refresh
│   │   │   ├── chat.py            # AI chat with SSE streaming
│   │   │   ├── sessions.py        # Chat session management
│   │   │   ├── strategies.py      # Strategy CRUD, validation, versioning
│   │   │   ├── backtests.py       # Backtest execution and results
│   │   │   ├── deployments.py     # Deployment lifecycle
│   │   │   ├── broker_accounts.py # Broker account management
│   │   │   ├── trading_stream.py  # Real-time trading updates (SSE)
│   │   │   ├── trading_ws.py      # WebSocket trading stream
│   │   │   ├── portfolio.py       # Portfolio analytics
│   │   │   ├── market_data.py     # Market data queries
│   │   │   ├── billing.py         # Subscription management
│   │   │   ├── trade_approvals.py # Trade approval workflow
│   │   │   ├── health.py          # Health check
│   │   │   └── ...                # Social connectors, notifications, issues
│   │   ├── middleware/            # Auth, rate limiting, Sentry
│   │   ├── orchestration/         # AI chat orchestration
│   │   │   ├── agents/            # Skills-based agents
│   │   │   │   ├── kyc/           # KYC agent
│   │   │   │   ├── pre_strategy/  # Pre-strategy agent
│   │   │   │   ├── strategy/      # Strategy agent
│   │   │   │   ├── deployment/    # Deployment agent
│   │   │   │   └── stress_test/   # Stress test agent
│   │   │   └── ...                # Orchestration logic
│   │   └── services/              # Billing webhooks, Telegram, trading queue
│   ├── mcp/                       # MCP tool server (port 8110)
│   │   ├── main.py                # MCP server entry point
│   │   ├── router.py              # Request multiplexer
│   │   └── domains/               # 5 tool domains
│   │       ├── strategy/          # Strategy CRUD + DSL validation
│   │       ├── backtest/          # Backtest execution
│   │       ├── market_data/       # Market data queries
│   │       ├── stress/            # Stress testing
│   │       └── trading/           # Live trading operations
│   ├── worker/                    # Celery async workers
│   │   ├── cpu/                   # CPU-intensive (backtest, stress testing)
│   │   └── io/                    # I/O tasks (market data sync, paper trading, notifications)
│   └── beat/                      # Celery Beat scheduler
├── packages/                      # Reusable domain and infrastructure code
│   ├── domain/                    # Pure business logic (no infra deps)
│   │   ├── strategy/              # Strategy DSL, models, semantic validation
│   │   ├── backtest/              # Backtest engine, performance analytics
│   │   ├── trading/               # Trading runtime, broker ops, PnL
│   │   ├── stress/                # Stress testing (Monte Carlo, black swan)
│   │   ├── market_data/           # Data aggregation, catalog, sync
│   │   ├── session/               # Session management
│   │   ├── user/                  # User domain logic
│   │   ├── billing/               # Subscription, quota, usage tracking
│   │   ├── notification/          # Notification services
│   │   └── ports/                 # Abstract interfaces for queue operations
│   ├── infra/                     # Infrastructure adapters
│   │   ├── db/                    # SQLAlchemy session, 40 database models
│   │   ├── redis/                 # Redis client, locks, stores
│   │   ├── queue/                 # Celery app config, publishers
│   │   ├── auth/                  # JWT, MCP context auth
│   │   └── providers/             # External integrations
│   │       ├── trading/           # Broker adapters (CCXT, Alpaca, Sandbox)
│   │       ├── market_data/       # Data providers
│   │       ├── telegram/          # Telegram bot
│   │       ├── stripe/            # Stripe billing
│   │       └── im/               # Instant messaging
│   ├── core/                      # Cross-module contracts (events)
│   ├── shared_settings/           # Unified configuration (layered env loading)
│   └── logs/                      # Logging + Sentry integration
├── env/                           # Environment configuration files
│   ├── .env.secrets               # API keys, secrets (gitignored)
│   ├── .env.common                # Shared across all services
│   ├── .env.dev                   # Dev profile
│   ├── .env.prod                  # Prod profile
│   └── .env.{profile}.{service}   # Service-specific overrides
├── test/                          # Integration tests
├── tests/                         # Unit tests
├── scripts/                       # Utility scripts
├── data/                          # Market data (crypto, forex, futures, us_stocks)
├── runtime/                       # Runtime artifacts (backups, exports, reports)
├── docs/ & dev_docs/              # Documentation
├── pyproject.toml                 # Project config, dependencies, Ruff, pytest
├── uv.lock                        # Locked dependencies
├── Dockerfile                     # Multi-stage Docker build
├── compose.dev.yml                # Docker Compose (7 services)
└── Caddyfile                      # Reverse proxy config
```

## Architecture

### Service Architecture

The backend runs as 4 separate services:

1. **API** (`apps/api`) — FastAPI HTTP server on port 8000. Handles REST endpoints, SSE streaming, WebSocket connections, and AI chat orchestration.
2. **MCP** (`apps/mcp`) — MCP tool server on port 8110. Provides tool endpoints for strategy, backtest, market data, stress testing, and trading operations. Called by the AI orchestration layer.
3. **Worker CPU** (`apps/worker/cpu`) — Celery worker for CPU-intensive tasks (backtest execution, stress testing).
4. **Worker IO** (`apps/worker/io`) — Celery worker for I/O-bound tasks (market data sync, paper trading, notifications, trade approvals).
5. **Beat** (`apps/beat`) — Celery Beat scheduler for periodic tasks.

### Layered Architecture

```
Presentation    →  FastAPI routes + MCP domains
Orchestration   →  AI chat orchestration with skills-based agents
Domain          →  Pure business logic (packages/domain/)
Infrastructure  →  DB, Redis, Queue, Auth, Providers (packages/infra/)
Cross-cutting   →  Logging, Sentry, Events (packages/core/, packages/logs/)
```

### Import Boundary Rules (Enforced)

- `packages/` CANNOT import from `apps.*`
- `packages/domain/` has NO infrastructure dependencies
- Cross-app imports are restricted (allowlist-based)
- Enforced by `scripts/check_import_boundaries.py`

### Data Flow

```
Frontend → API (FastAPI) → Orchestration (AI Chat) → MCP Tools
                                                       ↓
                                               Domain Logic
                                                       ↓
                                       Infrastructure (DB/Redis/Queue)
                                                       ↓
                                       Celery Workers (CPU/IO)
```

### AI Chat Orchestration

The chat system uses a skills-based agent architecture:
- **KYC Agent** — User onboarding and risk profiling
- **Pre-Strategy Agent** — Strategy ideation and refinement
- **Strategy Agent** — Strategy creation and DSL validation
- **Deployment Agent** — Deployment lifecycle management
- **Stress Test Agent** — Stress testing orchestration

Agents communicate with the MCP server to execute tools (strategy CRUD, backtest, market data, trading).

## Common Commands

```bash
# Install dependencies
uv sync --frozen

# Run the API server (development)
uv run uvicorn apps.api.main:app --host 0.0.0.0 --port 8000 --reload

# Run the MCP server
uv run uvicorn apps.mcp.main:app --host 0.0.0.0 --port 8110 --reload

# Run Celery workers
uv run celery -A apps.worker.cpu worker --loglevel=info --pool=prefork
uv run celery -A apps.worker.io worker --loglevel=info --pool=gevent

# Run Celery Beat
uv run celery -A apps.beat beat --loglevel=info

# Run all services via Docker Compose
docker compose -f compose.dev.yml up -d --build

# Run infra only (for local uvicorn/celery debug)
docker compose -f compose.dev.yml up -d postgres redis

# Run all tests
uv run pytest -q

# Run integration tests only
uv run pytest test/integration/

# Run unit tests only
uv run pytest tests/

# Run a specific test file
uv run pytest test/integration/test_chat_api.py -v

# Lint and format
uv run ruff check .
uv run ruff format .

# Check import boundaries
uv run python scripts/check_import_boundaries.py

# Database migrations (Alembic)
uv run alembic upgrade head
uv run alembic revision --autogenerate -m "description"
```

## Environment Configuration

### Layered Loading Order (later overrides earlier)

1. `env/.env.secrets` — API keys, secrets
2. `env/.env.common` — Shared across all services
3. `env/.env.{profile}` — Profile-specific (`dev` or `prod`)
4. `env/.env.{profile}.{service}` — Service-specific (`api`, `mcp`, `worker_cpu`, `worker_io`, `beat`)

### Key Environment Variables

| Variable | Purpose |
|----------|---------|
| `MINSY_ENV_PROFILE` | `dev` or `prod` (default: `dev`) |
| `MINSY_SERVICE` | Service identifier |
| `OPENAI_API_KEY` | OpenAI API key |
| `OPENAI_RESPONSE_MODEL` | AI model to use |
| `POSTGRES_HOST` | PostgreSQL host |
| `POSTGRES_DB` | Database name |
| `REDIS_HOST` | Redis host |
| `STRIPE_PUBLISHABLE_KEY` | Stripe publishable key |
| `STRIPE_SECRET_KEY` | Stripe billing key |
| `STRIPE_WEBHOOK_SECRET` | Stripe webhook verification |
| `STRIPE_PRICE_PLUS_MONTHLY` | Stripe price id for Plus plan |
| `STRIPE_PRICE_PRO_MONTHLY` | Stripe price id for Pro plan |
| `BILLING_PRICING_JSON` | Unified billing config (pricing/cost model/tier limits) |
| `BILLING_FRONTEND_BASE_URL` | Frontend return URL base for billing flow |
| `BILLING_CHECKOUT_SUCCESS_URL` | Override Stripe checkout success return URL |
| `BILLING_CHECKOUT_CANCEL_URL` | Override Stripe checkout cancel return URL |
| `BILLING_PORTAL_RETURN_URL` | Override Stripe portal return URL |
| `MCP_SERVER_URL_*` | MCP endpoint URLs |
| `SENTRY_DSN` | Sentry error tracking |
| `JWT_SECRET_KEY` | JWT signing secret |

Configuration is managed via Pydantic settings in `packages/shared_settings/`.

## Database

- **ORM:** SQLAlchemy 2.0+ with async support (asyncpg driver)
- **Migrations:** Alembic (in `packages/infra/db/`)
- **40 models** across domains:
  - User: `user`, `user_settings`, `social_connector`
  - Strategy: `strategy`, `strategy_revision`
  - Trading: `deployment`, `deployment_run`, `order`, `fill`, `position`, `signal_event`, `manual_trade_action`
  - Broker: `broker_account_audit_log`
  - Market data: `market_data_catalog`, `market_data_sync_chunk`, `market_data_error_event`
  - Backtest/Stress: `stress_job`
  - PnL: `pnl_snapshot`, `sandbox_ledger_entry`
  - Billing: `billing_customer`, `billing_subscription`, `billing_usage_monthly`, `billing_webhook_event`
  - Notifications: `notification_delivery_attempt`
  - Trading events: `trade_approval_request`, `order_state_transition`, `trading_event_outbox`, `phase_transition`

## API Endpoints

All endpoints are prefixed with `/api/v1/`.

| Endpoint | Method(s) | Purpose |
|----------|-----------|---------|
| `/auth/*` | POST | Login, registration, token refresh |
| `/chat/*` | POST, GET | AI chat with SSE streaming |
| `/sessions/*` | GET, POST, PATCH, DELETE | Chat session management |
| `/strategies/*` | GET, POST, PUT, DELETE | Strategy CRUD, validation |
| `/backtests/*` | POST, GET | Backtest execution and results |
| `/deployments/*` | GET, POST, PATCH, DELETE | Deployment lifecycle |
| `/broker-accounts/*` | GET, POST, DELETE | Broker account management |
| `/trading-stream/*` | GET (SSE) | Real-time trading updates |
| `/trading-ws/*` | WebSocket | Bidirectional trading stream |
| `/portfolio/*` | GET | Portfolio analytics |
| `/market-data/*` | GET | Market data queries |
| `/billing/*` | GET, POST | Subscription management |
| `/trade-approvals/*` | GET, POST | Trade approval workflow |
| `/health` | GET | Health check |

### Billing + Quota Notes (2026-03-06)

- Billing endpoints are implemented in `apps/api/routes/billing.py` and include plans, checkout, portal, overview, webhook sync, and hosted return page.
- Quota guard is enforced in chat, strategy creation, backtest creation, deployment create/start, and stress job scheduling.
- Usage metrics persisted by `packages/domain/billing/usage_service.py`:
  - `ai_tokens_monthly_total`
  - `cpu_tokens_monthly_total`
  - `strategies_current_count`
  - `deployments_running_count`

## Authentication

- **Method:** JWT (HS256)
- **Access token expiry:** 1440 minutes (24 hours)
- **Refresh token expiry:** 7 days
- **Rate limiting:** 30 requests per 60 seconds on auth endpoints
- **MCP auth:** Context-based authentication via `packages/infra/auth/mcp_context.py`
- **Middleware:** `apps/api/middleware/auth.py`

## Code Style & Conventions

- **Linter/Formatter:** Ruff (configured in `pyproject.toml`)
- **Target:** Python 3.12
- **Line length:** 88
- **Rules:** E, F, I, N, W, UP (E501 ignored)
- **Naming:** snake_case for functions/variables, PascalCase for classes
- **Type hints:** Required throughout, Pydantic models for validation
- **Async-first:** Use `async def` for all I/O-bound operations
- **Import style:** Absolute imports, enforced boundaries between layers

## Docker & Deployment

### Docker Build

```bash
docker build -t minsy-backend:dev .
```

Multi-stage build: Python 3.12-slim base, TA-Lib C library, uv-based dependency installation.

### Docker Compose (`compose.dev.yml`)

7 services:
- `postgres` — PostgreSQL database
- `redis` — Redis cache/broker
- `api` — FastAPI server (port 8000)
- `mcp` — MCP tool server (port 8110)
- `worker-cpu` — Celery CPU worker
- `worker-io` — Celery IO worker
- `beat` — Celery Beat scheduler

### Caddy Reverse Proxy (`Caddyfile`)

- `api.minsyai.com` → API (port 8000)
- `mcp.minsyai.com/{strategy|backtest|market|stress|trading}` → MCP (port 8110)
- `flower.minsyai.com` → Celery Flower (port 5555)
- SSE endpoints have compression disabled (`/api/v1/chat/*`)

## Testing

- **Framework:** pytest with pytest-asyncio
- **Config:** `pyproject.toml` (`asyncio_mode = "auto"`)
- **Integration tests:** `test/` — API, domain, MCP, Celery, container tests
- **Unit tests:** `tests/` — agents, API, domain logic
- **Fixtures:** Docker Compose stack, seeded credentials, auth headers, test client
- **Test helpers:** `test/_support/live_helpers.py` for Docker Compose orchestration

Always run `uv run pytest -q` before committing.

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/check_import_boundaries.py` | Enforce architectural import boundaries |
| `scripts/migrate_legacy_docker_data.py` | PostgreSQL/Redis data migration |
| `scripts/generate_demo_full_chat_session.py` | Generate demo data |
| `scripts/cleanup_test_user_data.py` | Clean up test data |
| `scripts/verify_strategy_validate_dsl_routing.py` | Strategy validation testing |
| `scripts/run_integration_tests.sh` | Integration test runner |

## Important Notes

- The AI orchestration layer (`apps/api/orchestration/`) is the most complex part. It manages multi-turn conversations with skills-based agents that call MCP tools.
- The MCP server (`apps/mcp/`) is a separate service — changes to tool definitions affect AI behavior.
- Domain logic in `packages/domain/` must remain free of infrastructure dependencies. Use ports/adapters pattern.
- Celery workers are split by workload type: CPU-intensive (backtest, stress) vs I/O-bound (market data, notifications). Route tasks to the correct worker.
- The strategy DSL (`packages/domain/strategy/`) has its own validation pipeline — changes require careful testing.
- Broker adapters (`packages/infra/providers/trading/`) support CCXT, Alpaca, and Sandbox — each has different capabilities.
- Environment configuration uses layered loading — service-specific overrides take precedence. Check `packages/shared_settings/` for the schema.
- Import boundaries are enforced — run `scripts/check_import_boundaries.py` to verify before committing.
