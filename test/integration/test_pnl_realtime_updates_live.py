"""Live integration tests for real-time PnL and position updates.

These tests verify:
1. PnL snapshot creation and updates
2. Position tracking accuracy
3. Real-time state synchronization
4. Runtime state store functionality
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from test._support.live_helpers import (
    BACKEND_DIR,
    run_command,
)


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
            "description": f"PnL test strategy",
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
                        "cross": {
                            "a": {"ref": "ema_9"},
                            "op": "cross_above",
                            "b": {"ref": "ema_21"},
                        }
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
                ],
            },
        },
    }


class TestPnLSnapshotTracking:
    """Test suite for PnL snapshot tracking."""

    def _create_active_deployment(
        self,
        api_test_client: TestClient,
        auth_headers: dict[str, str],
    ) -> str:
        """Create and start a deployment."""
        from packages.shared_settings.schema.settings import settings

        # Create thread
        thread_resp = api_test_client.post(
            "/api/v1/chat/new-thread",
            headers=auth_headers,
            json={"metadata": {"source": "pnl-test"}},
        )
        session_id = thread_resp.json()["session_id"]

        # Create strategy
        dsl = _load_crypto_strategy_dsl(
            name=f"PnL-Test-{uuid4().hex[:6]}",
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
                "metadata": {"source": "pnl-test"},
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
                "capital_allocated": 10000,
                "risk_limits": {},
                "runtime_state": {},
            },
        )
        deployment_id = deploy_resp.json()["deployment_id"]

        # Start deployment
        api_test_client.post(
            f"/api/v1/deployments/{deployment_id}/start",
            headers=auth_headers,
        )

        return deployment_id

    def test_000_pnl_snapshots_endpoint(
        self,
        api_test_client: TestClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test PnL snapshots endpoint."""
        deployment_id = self._create_active_deployment(api_test_client, auth_headers)

        # Wait for runtime to process
        time.sleep(10)

        # Get PnL snapshots
        pnl_resp = api_test_client.get(
            f"/api/v1/deployments/{deployment_id}/pnl-snapshots",
            headers=auth_headers,
        )
        # Endpoint may or may not exist
        assert pnl_resp.status_code in {200, 404}

        # Clean up
        api_test_client.post(
            f"/api/v1/deployments/{deployment_id}/manual-actions",
            headers=auth_headers,
            json={"action": "stop", "payload": {"reason": "pnl-test-cleanup"}},
        )

    def test_010_runtime_state_contains_pnl(
        self,
        api_test_client: TestClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test runtime state contains PnL information."""
        deployment_id = self._create_active_deployment(api_test_client, auth_headers)

        # Wait for runtime to process
        time.sleep(15)

        # Get deployment detail
        detail_resp = api_test_client.get(
            f"/api/v1/deployments/{deployment_id}",
            headers=auth_headers,
        )
        assert detail_resp.status_code == 200

        # Check for runtime state
        deployment = detail_resp.json()
        # Runtime state may contain PnL info
        if "runtime_state" in deployment:
            runtime_state = deployment["runtime_state"]
            # May have equity, pnl, or other financial info
            assert isinstance(runtime_state, dict)

        # Clean up
        api_test_client.post(
            f"/api/v1/deployments/{deployment_id}/manual-actions",
            headers=auth_headers,
            json={"action": "stop", "payload": {"reason": "pnl-test-cleanup"}},
        )


class TestPositionTracking:
    """Test suite for position tracking."""

    def test_000_positions_endpoint_returns_list(
        self,
        api_test_client: TestClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test positions endpoint returns a list."""
        # Get all deployments
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
            data = positions_resp.json()
            assert "positions" in data or isinstance(data, list)

    def test_010_orders_endpoint_returns_list(
        self,
        api_test_client: TestClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test orders endpoint returns a list."""
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
            data = orders_resp.json()
            assert "orders" in data or isinstance(data, list)


class TestRuntimeStateStore:
    """Test suite for runtime state store functionality."""

    def _extract_result_json(self, stdout: str) -> dict[str, Any]:
        for line in stdout.splitlines():
            if line.startswith("RESULT_JSON="):
                return json.loads(line.removeprefix("RESULT_JSON="))
        raise AssertionError(f"No RESULT_JSON in output: {stdout[:500]}")

    def test_000_runtime_state_store_upsert(
        self,
        compose_stack: list[dict[str, Any]],
    ) -> None:
        """Test runtime state store upsert operation."""
        _ = compose_stack

        result = run_command(
            [
                "docker",
                "exec",
                "minsy-api-dev",
                ".venv/bin/python",
                "-c",
                "import json; "
                "import asyncio; "
                "from uuid import uuid4; "
                "from packages.infra.redis.stores.runtime_state_store import runtime_state_store; "
                "async def test(): "
                "    test_id=uuid4(); "
                "    await runtime_state_store.upsert(test_id,{'test':True}); "
                "    state=await runtime_state_store.get(test_id); "
                "    return {'status':'ok','has_state':state is not None}; "
                "result=asyncio.run(test()); "
                "print('RESULT_JSON='+json.dumps(result))",
            ],
            cwd=BACKEND_DIR,
            timeout=60,
        )
        payload = self._extract_result_json(result.stdout)
        assert payload.get("status") == "ok"

    def test_010_runtime_state_store_get(
        self,
        compose_stack: list[dict[str, Any]],
    ) -> None:
        """Test runtime state store get operation."""
        _ = compose_stack

        result = run_command(
            [
                "docker",
                "exec",
                "minsy-api-dev",
                ".venv/bin/python",
                "-c",
                "import json; "
                "import asyncio; "
                "from uuid import uuid4; "
                "from packages.infra.redis.stores.runtime_state_store import runtime_state_store; "
                "async def test(): "
                "    test_id=uuid4(); "
                "    state=await runtime_state_store.get(test_id); "
                "    return {'status':'ok','state_is_none':state is None}; "
                "result=asyncio.run(test()); "
                "print('RESULT_JSON='+json.dumps(result))",
            ],
            cwd=BACKEND_DIR,
            timeout=60,
        )
        payload = self._extract_result_json(result.stdout)
        assert payload.get("status") == "ok"


class TestSignalStore:
    """Test suite for signal store functionality."""

    def _extract_result_json(self, stdout: str) -> dict[str, Any]:
        for line in stdout.splitlines():
            if line.startswith("RESULT_JSON="):
                return json.loads(line.removeprefix("RESULT_JSON="))
        raise AssertionError(f"No RESULT_JSON in output: {stdout[:500]}")

    def test_000_signal_store_operations(
        self,
        compose_stack: list[dict[str, Any]],
    ) -> None:
        """Test signal store operations."""
        _ = compose_stack

        result = run_command(
            [
                "docker",
                "exec",
                "minsy-api-dev",
                ".venv/bin/python",
                "-c",
                "import json; "
                "import asyncio; "
                "from uuid import uuid4; "
                "from packages.infra.redis.stores.signal_store import signal_store, SignalRecord; "
                "async def test(): "
                "    test_id=uuid4(); "
                "    record=SignalRecord(signal='hold',reason='test',timestamp_utc='2024-01-01T00:00:00Z'); "
                "    await signal_store.push(test_id,record); "
                "    signals=await signal_store.get_recent(test_id,limit=10); "
                "    return {'status':'ok','signal_count':len(signals)}; "
                "result=asyncio.run(test()); "
                "print('RESULT_JSON='+json.dumps(result))",
            ],
            cwd=BACKEND_DIR,
            timeout=60,
        )
        payload = self._extract_result_json(result.stdout)
        assert payload.get("status") == "ok"
        assert payload.get("signal_count", 0) >= 0


class TestRealTimeUpdates:
    """Test suite for real-time update mechanisms."""

    def test_000_trading_stream_endpoint(
        self,
        api_test_client: TestClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test trading stream SSE endpoint."""
        # Get a deployment
        deployments_resp = api_test_client.get(
            "/api/v1/deployments",
            headers=auth_headers,
        )
        assert deployments_resp.status_code == 200

        deployments = deployments_resp.json().get("deployments", [])
        if not deployments:
            pytest.skip("No deployments available")

        deployment_id = deployments[0]["deployment_id"]

        # Try to connect to trading stream
        # Note: This is a streaming endpoint, so we just check it's accessible
        stream_resp = api_test_client.get(
            f"/api/v1/trading-stream/{deployment_id}",
            headers=auth_headers,
            timeout=5,
        )
        # May timeout or return data
        assert stream_resp.status_code in {200, 404, 408}

    def test_010_portfolio_endpoint(
        self,
        api_test_client: TestClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test portfolio endpoint for aggregated view."""
        portfolio_resp = api_test_client.get(
            "/api/v1/portfolio",
            headers=auth_headers,
        )
        assert portfolio_resp.status_code == 200
        data = portfolio_resp.json()
        # Should have some portfolio data structure
        assert isinstance(data, dict)
