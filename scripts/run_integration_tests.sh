#!/usr/bin/env bash
# Run integration tests for the paper trading system
# Usage: ./scripts/run_integration_tests.sh [options]
#
# Options:
#   --all           Run all integration tests
#   --paper         Run paper trading tests only (60% focus)
#   --openai        Run OpenAI response tests only
#   --alpaca        Run Alpaca market data tests only
#   --mcp           Run MCP endpoint tests only
#   --celery        Run Celery beat scheduling tests only
#   --containers    Run multi-container communication tests only
#   --stress        Run stress/capacity tests only
#   --quick         Run quick smoke tests only
#   -v, --verbose   Verbose output
#   -k PATTERN      Run tests matching pattern

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(dirname "$SCRIPT_DIR")"

cd "$BACKEND_DIR"

# Default options
VERBOSE=""
PATTERN=""
TEST_PATHS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            TEST_PATHS="test/integration/"
            shift
            ;;
        --paper)
            TEST_PATHS="test/integration/test_paper_trading_live.py test/integration/test_pnl_realtime_updates_live.py test/integration/test_multi_strategy_tracking_live.py"
            shift
            ;;
        --openai)
            TEST_PATHS="test/integration/test_openai_real_response_live.py"
            shift
            ;;
        --alpaca)
            TEST_PATHS="test/integration/test_alpaca_market_data_live.py"
            shift
            ;;
        --mcp)
            TEST_PATHS="test/integration/test_mcp_endpoints_live.py"
            shift
            ;;
        --celery)
            TEST_PATHS="test/integration/test_celery_beat_scheduling_live.py"
            shift
            ;;
        --containers)
            TEST_PATHS="test/integration/test_multi_container_communication_live.py"
            shift
            ;;
        --stress)
            TEST_PATHS="test/integration/test_stress_capacity_live.py"
            shift
            ;;
        --quick)
            PATTERN="-k 'test_000'"
            TEST_PATHS="test/integration/"
            shift
            ;;
        -v|--verbose)
            VERBOSE="-v"
            shift
            ;;
        -k)
            PATTERN="-k '$2'"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Default to all tests if no path specified
if [[ -z "$TEST_PATHS" ]]; then
    TEST_PATHS="test/integration/"
fi

echo "=========================================="
echo "Running Integration Tests"
echo "=========================================="
echo "Test paths: $TEST_PATHS"
echo "Verbose: ${VERBOSE:-no}"
echo "Pattern: ${PATTERN:-all}"
echo ""

# Ensure compose stack is up
echo "Ensuring Docker Compose stack is running..."
docker compose -f compose.dev.yml up -d --build

# Wait for services to be healthy
echo "Waiting for services to be healthy..."
sleep 10

# Run tests
echo ""
echo "Running tests..."
eval "uv run pytest $VERBOSE $PATTERN $TEST_PATHS --tb=short -x"

echo ""
echo "=========================================="
echo "Integration Tests Complete"
echo "=========================================="
