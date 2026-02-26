# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Minsy backend — Python 3.12+ FastAPI microservices for an AI-powered quantitative trading strategy platform. Uses MCP (Model Context Protocol) orchestration, Celery workers, and async PostgreSQL/Redis.

## Common Commands

```bash
# Dependencies (uses uv package manager)
uv sync --frozen               # Install from lock file
uv sync                        # Update and install

# Run API server
uv run uvicorn apps.api.main:app --reload --port 8000

# Run tests
uv run pytest -q                                    # All tests
uv run pytest -q tests/test_api/                    # API tests only
uv run pytest -q tests/test_mcp/                    # MCP tests only
uv run pytest -q tests/test_engine/                 # Engine tests only
uv run pytest -q tests/test_infra/                  # Infra tests only
uv run pytest tests/test_infra/test_import_boundaries.py -q  # Single test file

# Lint and format
uv run ruff check --select E,F,I,N,W,UP src        # Lint (E501 ignored)
uv run ruff format --line-length 88 src             # Format

# Architecture validation
uv run python scripts/check_import_boundaries.py

# Local services (postgres + redis)
docker-compose -f compose.dev.yml up -d
```

## Architecture

### Layered Structure

The backend follows a strict layered architecture with import boundaries enforced by `scripts/check_import_boundaries.py`:

```
apps/           → Application entry points (no business logic here)
  api/          → FastAPI app, routes, schemas, services, middleware, orchestration
  mcp/          → MCP servers per domain (strategy, backtest, market_data, stress, trading)
  worker/       → Celery tasks split by profile (cpu/ and io/)
  beat/         → Celery Beat scheduler

packages/       → Reusable domain libraries
  core/         → Shared contracts, enums, types
  domain/       → Pure business logic (strategy DSL, backtest engine, trading, stress, market_data)
  infra/        → Infrastructure adapters (db/SQLAlchemy, redis, queue/Celery, auth/JWT, providers/yfinance)
  shared_settings/ → Configuration management

src/            → LEGACY — being migrated to apps/ + packages/ (do not add new code here)
```

**Key rule:** Domain logic (`packages/domain/`) must not import from FastAPI, Celery, or SQLAlchemy directly. Infrastructure concerns live in `packages/infra/`.

### Key Patterns

- **Async everywhere**: asyncpg + SQLAlchemy async ORM, asyncio-native
- **MCP servers**: Each domain (strategy, backtest, market_data, stress, trading) runs on ports 8111-8115
- **Agent orchestration**: Multi-phase flow — KYC → Pre-Strategy → Strategy → Confirmation → Deployment (see `apps/api/orchestration/`)
- **Real-time**: SSE for chat streams, WebSocket for bidirectional, polling for status
- **pytest config**: `asyncio_mode = "auto"` in pyproject.toml — async tests work without explicit markers

### Refactoring Status

The backend is actively migrating from `src/` (legacy) to the `apps/ + packages/` structure. Some files in `src/` have compatibility shims. Always add new code to `apps/` or `packages/`, not `src/`.

## Infrastructure

- **Database**: PostgreSQL 16 (no Alembic — uses SQLAlchemy `create_all`)
- **Cache/Queue**: Redis 7
- **Task Queue**: Celery with cpu/io worker profiles + Flower monitoring
- **Reverse Proxy**: Caddy (production)
- **Observability**: Sentry
- **Deployment**: SSH-based GCP VM deploy via `.github/workflows/deploy.yml`, merges to `main` trigger deploy
