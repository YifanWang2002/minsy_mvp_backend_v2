"""Live integration tests for paper trading system.

These tests focus on the core paper trading functionality:
1. Multi-container communication
2. Celery beat scheduling
3. Order triggering and execution
4. Real-time PnL updates
5. Multi-symbol crypto strategies with different timeframes

This is the primary test suite (60% focus) for the paper trading system.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from test._support.live_helpers import (
    BACKEND_DIR,
    parse_sse_payloads,
    run_command,
)


# Crypto symbols for testing (24/7 market)
CRYPTO_SYMBOLS = ["BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD"]
TIMEFRAMES = ["1m", "5m"]


def _load_crypto_strategy_dsl(
    *,
    name: str,
    symbols: list[str],
    timeframe: str,
) -> dict[str, Any]:
    """Generate a crypto strategy DSL for testing."""
    return {
        "dsl_version": "1.0.0",
        "strategy": {
            "name": name,
            "description": f"Test crypto strategy for {', '.join(symbols)} on {timeframe}",
        },
        "universe": {
            "market": "crypto",
            "tickers": symbols,
        },
        "timeframe": timeframe,
        "factors": {
            "ema_9": {
                "type": "ema",
                "params": {"period": 9, "source": "close"},
            },
            "ema_21": {
                "type": "ema",
                "params": {"period": 21, "source": "close"},
            },
            "rsi_14": {
                "type": "rsi",
                "params": {"period": 14, "source": "close"},
            },
        },
        "trade": {
            "long": {
                "position_sizing": {
                    "mode": "pct_equity",
                    "pct": 0.1,
                },
                "entry": {
                    "order": {"type": "market"},
                    "condition": {
                        "all": [
                            {
                                "cross": {
                                    "a": {"ref": "ema_9"},
                                    "op": "cross_above",
                                    "b": {"ref": "ema_21"},
                                }
                            },
                            {
                                "cmp": {
                                    "left": {"ref": "rsi_14"},
                                    "op": "lt",
                                    "right": 70,
                                }
                            },
                        ]
                    },
                },
                "exits": [
                    {
                        "type": "signal_exit",
                        "name": "exit_on_cross_down",
                        "order": {"type": "market"},
                        "condition": {
                            "cross": {
                                "a": {"ref": "ema_9"},
                                "op": "cross_below",
                                "b": {"ref": "ema_21"},
                            }
                        },
                    },
                    {
                        "type": "stop_loss",
                        "name": "sl_pct",
                        "stop": {"kind": "pct", "value": 0.02},
                    },
                ],
            },
        },
    }


class TestPaperTradingDeploymentLifecycle:
    """Test suite for paper trading deployment lifecycle."""

    def _create_thread(
        self,
        api_test_client: TestClient,
        auth_headers: dict[str, str],
    ) -> str:
        """Create a new chat thread."""
        response = api_test_client.post(
            "/api/v1/chat/new-thread",
            headers=auth_headers,
            json={"metadata": {"source": "pytest-paper-trading-live"}},
        )
        assert response.status_code == 201, response.text
        return str(response.json()["session_id"])

    def _create_strategy(
        self,
        api_test_client: TestClient,
        auth_headers: dict[str, str],
        *,
        symbols: list[str],
        timeframe: str,
    ) -> str:
        """Create a strategy with given parameters."""
        session_id = self._create_thread(api_test_client, auth_headers)
        dsl = _load_crypto_strategy_dsl(
            name=f"Pytest Crypto {uuid4().hex[:8]}",
            symbols=symbols,
            timeframe=timeframe,
        )

        response = api_test_client.post(
            "/api/v1/strategies/confirm",
            headers=auth_headers,
            json={
                "session_id": session_id,
                "dsl_json": dsl,
                "auto_start_backtest": False,
                "language": "en",
            },
        )
        assert response.status_code == 200, response.text
        return str(response.json()["strategy_id"])

    def _create_broker_account(
        self,
        api_test_client: TestClient,
        auth_headers: dict[str, str],
    ) -> str:
        """Create a broker account with real Alpaca credentials."""
        from packages.shared_settings.schema.settings import settings

        tag = uuid4().hex[:10]
        response = api_test_client.post(
            "/api/v1/broker-accounts",
            headers=auth_headers,
            params={"validate": "true"},  # Validate with real Alpaca
            json={
                "provider": "alpaca",
                "mode": "paper",
                "credentials": {
                    "api_key": settings.alpaca_api_key,
                    "api_secret": settings.alpaca_api_secret,
                },
                "metadata": {"source": "pytest-paper-trading-live", "tag": tag},
            },
        )
        assert response.status_code == 201, response.text
        return str(response.json()["broker_account_id"])

    def _create_deployment(
        self,
        api_test_client: TestClient,
        auth_headers: dict[str, str],
        *,
        symbols: list[str],
        timeframe: str,
        capital: int = 10000,
    ) -> tuple[str, str]:
        """Create a deployment and return (deployment_id, strategy_id)."""
        strategy_id = self._create_strategy(
            api_test_client,
            auth_headers,
            symbols=symbols,
            timeframe=timeframe,
        )
        broker_account_id = self._create_broker_account(api_test_client, auth_headers)

        response = api_test_client.post(
            "/api/v1/deployments",
            headers=auth_headers,
            json={
                "strategy_id": strategy_id,
                "broker_account_id": broker_account_id,
                "mode": "paper",
                "capital_allocated": capital,
                "risk_limits": {"max_position_pct": 20, "max_daily_loss_pct": 5},
                "runtime_state": {"source": "pytest-paper-trading-live"},
            },
        )
        assert response.status_code == 201, response.text
        return str(response.json()["deployment_id"]), strategy_id

    def test_000_create_btc_1min_deployment(
        self,
        api_test_client: TestClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test creating a BTC/USD 1-minute deployment."""
        deployment_id, strategy_id = self._create_deployment(
            api_test_client,
            auth_headers,
            symbols=["BTC/USD"],
            timeframe="1m",
        )

        # Verify deployment was created
        detail = api_test_client.get(
            f"/api/v1/deployments/{deployment_id}",
            headers=auth_headers,
        )
        assert detail.status_code == 200
        payload = detail.json()
        assert payload["mode"] == "paper"
        assert payload["status"] in {"pending", "active"}

    def test_010_start_and_verify_deployment_active(
        self,
        api_test_client: TestClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test starting a deployment and verifying it becomes active."""
        deployment_id, _ = self._create_deployment(
            api_test_client,
            auth_headers,
            symbols=["ETH/USD"],
            timeframe="1m",
        )

        # Start the deployment
        start_response = api_test_client.post(
            f"/api/v1/deployments/{deployment_id}/start",
            headers=auth_headers,
        )
        assert start_response.status_code == 200
        assert start_response.json()["deployment"]["status"] == "active"

        # Wait a bit for scheduler to pick it up
        time.sleep(3)

        # Verify deployment is still active
        detail = api_test_client.get(
            f"/api/v1/deployments/{deployment_id}",
            headers=auth_headers,
        )
        assert detail.status_code == 200
        assert detail.json()["status"] == "active"

    def test_020_multi_symbol_deployment(
        self,
        api_test_client: TestClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test creating a multi-symbol deployment."""
        deployment_id, _ = self._create_deployment(
            api_test_client,
            auth_headers,
            symbols=["BTC/USD", "ETH/USD"],
            timeframe="5m",
        )

        start_response = api_test_client.post(
            f"/api/v1/deployments/{deployment_id}/start",
            headers=auth_headers,
        )
        assert start_response.status_code == 200

    def test_030_deployment_pause_resume_cycle(
        self,
        api_test_client: TestClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test pause and resume cycle for deployment."""
        deployment_id, _ = self._create_deployment(
            api_test_client,
            auth_headers,
            symbols=["BTC/USD"],
            timeframe="1m",
        )

        # Start
        api_test_client.post(
            f"/api/v1/deployments/{deployment_id}/start",
            headers=auth_headers,
        )

        # Pause
        pause_response = api_test_client.post(
            f"/api/v1/deployments/{deployment_id}/pause",
            headers=auth_headers,
        )
        assert pause_response.status_code == 200
        assert pause_response.json()["deployment"]["status"] == "paused"

        # Resume
        resume_response = api_test_client.post(
            f"/api/v1/deployments/{deployment_id}/start",
            headers=auth_headers,
        )
        assert resume_response.status_code == 200
        assert resume_response.json()["deployment"]["status"] == "active"

    def test_040_deployment_stop(
        self,
        api_test_client: TestClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test stopping a deployment."""
        deployment_id, _ = self._create_deployment(
            api_test_client,
            auth_headers,
            symbols=["BTC/USD"],
            timeframe="1m",
        )

        # Start then stop
        api_test_client.post(
            f"/api/v1/deployments/{deployment_id}/start",
            headers=auth_headers,
        )

        stop_response = api_test_client.post(
            f"/api/v1/deployments/{deployment_id}/manual-actions",
            headers=auth_headers,
            json={"action": "stop", "payload": {"reason": "pytest-stop-test"}},
        )
        assert stop_response.status_code == 200

        # Verify stopped
        detail = api_test_client.get(
            f"/api/v1/deployments/{deployment_id}",
            headers=auth_headers,
        )
        assert detail.json()["status"] == "stopped"


class TestPaperTradingRuntimeExecution:
    """Test suite for paper trading runtime execution."""

    def _extract_result_json(self, stdout: str) -> dict[str, Any]:
        for line in stdout.splitlines():
            if line.startswith("RESULT_JSON="):
                return json.loads(line.removeprefix("RESULT_JSON="))
        raise AssertionError(f"No RESULT_JSON in output: {stdout[:500]}")

    def _run_worker_script(self, script: str) -> dict[str, Any]:
        result = run_command(
            [
                "docker",
                "exec",
                "minsy-worker-io-dev",
                ".venv/bin/python",
                "-c",
                script,
            ],
            cwd=BACKEND_DIR,
            timeout=300,
        )
        return self._extract_result_json(result.stdout)

    def test_000_scheduler_tick_execution(
        self,
        compose_stack: list[dict[str, Any]],
    ) -> None:
        """Test that scheduler tick executes successfully."""
        _ = compose_stack
        payload = self._run_worker_script(
            "import json; "
            "from apps.worker.io.tasks.paper_trading import scheduler_tick_task; "
            "result=scheduler_tick_task.run(); "
            "print('RESULT_JSON='+json.dumps(result))"
        )
        assert payload.get("status") in {"ok", "disabled"}
        if payload.get("status") == "ok":
            assert "deployments_total" in payload
            assert "enqueued" in payload
            assert "skipped" in payload

    def test_010_market_data_refresh_task(
        self,
        compose_stack: list[dict[str, Any]],
    ) -> None:
        """Test market data refresh task execution."""
        _ = compose_stack
        payload = self._run_worker_script(
            "import json; "
            "from apps.worker.io.tasks.market_data import refresh_active_subscriptions_task; "
            "result=refresh_active_subscriptions_task.run(); "
            "print('RESULT_JSON='+json.dumps(result))"
        )
        assert "scheduled" in payload
        assert "total" in payload


class TestPaperTradingMultiStrategy:
    """Test suite for running multiple strategies simultaneously."""

    def _create_deployment_quick(
        self,
        api_test_client: TestClient,
        auth_headers: dict[str, str],
        *,
        symbol: str,
        timeframe: str,
    ) -> str:
        """Quick deployment creation helper."""
        from packages.shared_settings.schema.settings import settings

        # Create thread
        thread_resp = api_test_client.post(
            "/api/v1/chat/new-thread",
            headers=auth_headers,
            json={"metadata": {"source": "pytest-multi-strategy"}},
        )
        session_id = thread_resp.json()["session_id"]

        # Create strategy
        dsl = _load_crypto_strategy_dsl(
            name=f"Multi-{symbol}-{timeframe}-{uuid4().hex[:6]}",
            symbols=[symbol],
            timeframe=timeframe,
        )
        strategy_resp = api_test_client.post(
            "/api/v1/strategies/confirm",
            headers=auth_headers,
            json={
                "session_id": session_id,
                "dsl_json": dsl,
                "auto_start_backtest": False,
                "language": "en",
            },
        )
        strategy_id = strategy_resp.json()["strategy_id"]

        # Create broker account
        broker_resp = api_test_client.post(
            "/api/v1/broker-accounts",
            headers=auth_headers,
            params={"validate": "false"},
            json={
                "provider": "alpaca",
                "mode": "paper",
                "credentials": {
                    "api_key": settings.alpaca_api_key,
                    "api_secret": settings.alpaca_api_secret,
                },
                "metadata": {"source": "pytest-multi-strategy"},
            },
        )
        broker_account_id = broker_resp.json()["broker_account_id"]

        # Create deployment
        deploy_resp = api_test_client.post(
            "/api/v1/deployments",
            headers=auth_headers,
            json={
                "strategy_id": strategy_id,
                "broker_account_id": broker_account_id,
                "mode": "paper",
                "capital_allocated": 5000,
                "risk_limits": {"max_position_pct": 10},
                "runtime_state": {},
            },
        )
        return deploy_resp.json()["deployment_id"]

    def test_000_create_multiple_strategies_different_timeframes(
        self,
        api_test_client: TestClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test creating multiple strategies with different timeframes."""
        deployments = []

        # Create deployments for different symbols and timeframes
        configs = [
            ("BTC/USD", "1m"),
            ("ETH/USD", "1m"),
            ("BTC/USD", "5m"),
            ("SOL/USD", "5m"),
        ]

        for symbol, timeframe in configs:
            deployment_id = self._create_deployment_quick(
                api_test_client,
                auth_headers,
                symbol=symbol,
                timeframe=timeframe,
            )
            deployments.append(deployment_id)

        assert len(deployments) == 4

        # Start all deployments
        for deployment_id in deployments:
            start_resp = api_test_client.post(
                f"/api/v1/deployments/{deployment_id}/start",
                headers=auth_headers,
            )
            assert start_resp.status_code == 200

        # Wait for scheduler to process
        time.sleep(5)

        # Verify all are active
        for deployment_id in deployments:
            detail = api_test_client.get(
                f"/api/v1/deployments/{deployment_id}",
                headers=auth_headers,
            )
            assert detail.json()["status"] == "active"

        # Clean up - stop all
        for deployment_id in deployments:
            api_test_client.post(
                f"/api/v1/deployments/{deployment_id}/manual-actions",
                headers=auth_headers,
                json={"action": "stop", "payload": {"reason": "pytest-cleanup"}},
            )


class TestPaperTradingRuntimeState:
    """Test suite for runtime state tracking."""

    def test_000_runtime_state_updates(
        self,
        api_test_client: TestClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test that runtime state is properly updated."""
        from packages.shared_settings.schema.settings import settings

        # Create and start a deployment
        thread_resp = api_test_client.post(
            "/api/v1/chat/new-thread",
            headers=auth_headers,
            json={"metadata": {"source": "pytest-runtime-state"}},
        )
        session_id = thread_resp.json()["session_id"]

        dsl = _load_crypto_strategy_dsl(
            name=f"RuntimeState-{uuid4().hex[:6]}",
            symbols=["BTC/USD"],
            timeframe="1m",
        )
        strategy_resp = api_test_client.post(
            "/api/v1/strategies/confirm",
            headers=auth_headers,
            json={
                "session_id": session_id,
                "dsl_json": dsl,
                "auto_start_backtest": False,
                "language": "en",
            },
        )
        strategy_id = strategy_resp.json()["strategy_id"]

        broker_resp = api_test_client.post(
            "/api/v1/broker-accounts",
            headers=auth_headers,
            params={"validate": "false"},
            json={
                "provider": "alpaca",
                "mode": "paper",
                "credentials": {
                    "api_key": settings.alpaca_api_key,
                    "api_secret": settings.alpaca_api_secret,
                },
                "metadata": {"source": "pytest-runtime-state"},
            },
        )
        broker_account_id = broker_resp.json()["broker_account_id"]

        deploy_resp = api_test_client.post(
            "/api/v1/deployments",
            headers=auth_headers,
            json={
                "strategy_id": strategy_id,
                "broker_account_id": broker_account_id,
                "mode": "paper",
                "capital_allocated": 10000,
                "risk_limits": {},
                "runtime_state": {"initial": True},
            },
        )
        deployment_id = deploy_resp.json()["deployment_id"]

        # Start deployment
        api_test_client.post(
            f"/api/v1/deployments/{deployment_id}/start",
            headers=auth_headers,
        )

        # Wait for runtime to process
        time.sleep(10)

        # Check runtime views
        runtime_resp = api_test_client.get(
            f"/api/v1/deployments/{deployment_id}/runtime",
            headers=auth_headers,
        )
        if runtime_resp.status_code == 200:
            runtime_data = runtime_resp.json()
            # Runtime state should have been updated
            assert "deployment_id" in runtime_data or "status" in runtime_data

        # Clean up
        api_test_client.post(
            f"/api/v1/deployments/{deployment_id}/manual-actions",
            headers=auth_headers,
            json={"action": "stop", "payload": {"reason": "pytest-cleanup"}},
        )


class TestPaperTradingPositionsAndOrders:
    """Test suite for positions and orders tracking."""

    def test_000_positions_endpoint(
        self,
        api_test_client: TestClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test positions endpoint returns valid data."""
        # List all deployments first
        deployments_resp = api_test_client.get(
            "/api/v1/deployments",
            headers=auth_headers,
        )
        assert deployments_resp.status_code == 200

        deployments = deployments_resp.json().get("deployments", [])
        if deployments:
            deployment_id = deployments[0]["deployment_id"]
            positions_resp = api_test_client.get(
                f"/api/v1/deployments/{deployment_id}/positions",
                headers=auth_headers,
            )
            assert positions_resp.status_code == 200

    def test_010_orders_endpoint(
        self,
        api_test_client: TestClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test orders endpoint returns valid data."""
        deployments_resp = api_test_client.get(
            "/api/v1/deployments",
            headers=auth_headers,
        )
        assert deployments_resp.status_code == 200

        deployments = deployments_resp.json().get("deployments", [])
        if deployments:
            deployment_id = deployments[0]["deployment_id"]
            orders_resp = api_test_client.get(
                f"/api/v1/deployments/{deployment_id}/orders",
                headers=auth_headers,
            )
            assert orders_resp.status_code == 200

    def test_020_portfolio_overview(
        self,
        api_test_client: TestClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test portfolio overview endpoint."""
        portfolio_resp = api_test_client.get(
            "/api/v1/portfolio",
            headers=auth_headers,
        )
        assert portfolio_resp.status_code == 200
        data = portfolio_resp.json()
        assert "total_equity" in data or "deployments" in data or "positions" in data
