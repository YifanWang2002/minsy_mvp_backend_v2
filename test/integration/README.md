# Integration Test Suite

This directory contains comprehensive integration tests for the Minsy paper trading system.

## Test Files Overview

| File | Description | Focus |
|------|-------------|-------|
| `test_paper_trading_live.py` | Paper trading deployment lifecycle, multi-symbol strategies | 60% |
| `test_pnl_realtime_updates_live.py` | PnL snapshots, position tracking, runtime state | 60% |
| `test_multi_strategy_tracking_live.py` | Multi-strategy creation and tracking | 60% |
| `test_celery_beat_scheduling_live.py` | Celery beat scheduling, worker coordination | 60% |
| `test_openai_real_response_live.py` | Real OpenAI API response streaming | 10% |
| `test_alpaca_market_data_live.py` | Real Alpaca market data fetching | 10% |
| `test_mcp_endpoints_live.py` | MCP router and tool endpoints | 10% |
| `test_multi_container_communication_live.py` | Container health and networking | 5% |
| `test_stress_capacity_live.py` | Stress testing and capacity limits | 5% |
| `test_parquet_data_loading_live.py` | Local parquet data loading | 5% |

## Prerequisites

1. Docker Compose stack running:
   ```bash
   docker compose -f compose.dev.yml up -d --build
   ```

2. Environment variables configured in `env/.env.secrets`:
   - `OPENAI_API_KEY`
   - `ALPACA_API_KEY`
   - `ALPACA_API_SECRET`

3. Test user exists in PostgreSQL:
   - Email: `2@test.com`
   - Password: `123456`

4. For MCP tests with OpenAI, cloudflared must be installed:
   ```bash
   brew install cloudflared
   ```

## Running Tests

### Using the Script

```bash
# Run all integration tests
./scripts/run_integration_tests.sh --all

# Run paper trading tests only (60% focus)
./scripts/run_integration_tests.sh --paper

# Run quick smoke tests
./scripts/run_integration_tests.sh --quick

# Run with verbose output
./scripts/run_integration_tests.sh --all -v

# Run specific pattern
./scripts/run_integration_tests.sh --all -k "test_000"
```

### Using pytest Directly

```bash
# All integration tests
uv run pytest test/integration/ -v

# Paper trading tests
uv run pytest test/integration/test_paper_trading_live.py -v

# Specific test class
uv run pytest test/integration/test_paper_trading_live.py::TestPaperTradingDeploymentLifecycle -v

# Specific test
uv run pytest test/integration/test_paper_trading_live.py::TestPaperTradingDeploymentLifecycle::test_000_create_btc_1min_deployment -v
```

## Test Categories

### Paper Trading System (60% Focus)

These tests verify the core paper trading functionality:

1. **Deployment Lifecycle**
   - Create deployments with real Alpaca credentials
   - Start/pause/resume/stop deployments
   - Multi-symbol deployments

2. **Runtime Execution**
   - Scheduler tick execution
   - Market data refresh
   - Signal generation

3. **Multi-Strategy**
   - Multiple strategies with different timeframes (1m, 5m)
   - Multiple symbols (BTC/USD, ETH/USD, SOL/USD, DOGE/USD)
   - Concurrent deployment tracking

4. **PnL and Positions**
   - Runtime state updates
   - Position tracking
   - Order tracking

### OpenAI Integration (10%)

- Real SSE streaming
- Response quality
- Tool call integration
- Multi-language support

### Alpaca Market Data (10%)

- Real-time quotes
- Historical bars
- Multiple timeframes
- Data quality validation

### MCP Endpoints (10%)

- Router health
- Domain accessibility
- Tool registration
- Cloudflared tunnel integration

### Infrastructure (10%)

- Container health
- Inter-container networking
- Service discovery
- Celery task queues

## Stress Testing

The stress tests evaluate system capacity:

- Maximum concurrent deployments (10, 20)
- Multi-symbol deployments (5 symbols)
- Concurrent API requests (50)
- Resource utilization monitoring
- Queue backpressure handling

## Notes

1. **Real Data Sources**: All tests use real data sources (OpenAI, Alpaca) - no mocks.

2. **Crypto Market**: Tests use crypto symbols (BTC/USD, ETH/USD) because the crypto market is 24/7.

3. **Cleanup**: Tests clean up created resources after completion.

4. **Timeouts**: Some tests may take longer due to real API calls.

5. **Rate Limits**: Be aware of API rate limits when running tests frequently.
