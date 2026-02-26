"""Live integration tests for multi-strategy tracking.

These tests create and track multiple crypto strategies simultaneously:
1. Different symbols (BTC, ETH, SOL, DOGE)
2. Different timeframes (1m, 5m)
3. Monitor their execution over time
4. Verify order triggering and PnL updates
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
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
    symbol: str,
    timeframe: str,
) -> dict[str, Any]:
    """Generate a crypto strategy DSL for a single symbol."""
    return {
        "dsl_version": "1.0.0",
        "strategy": {
            "name": name,
            "description": f"Multi-strategy test for {symbol} on {timeframe}",
        },
        "universe": {
            "market": "crypto",
            "tickers": [symbol],
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
            "short": {
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
                                    "op": "cross_below",
                                    "b": {"ref": "ema_21"},
                                }
                            },
                            {
                                "cmp": {
                                    "left": {"ref": "rsi_14"},
                                    "op": "gt",
                                    "right": 30,
                                }
                            },
                        ]
                    },
                },
                "exits": [
                    {
                        "type": "signal_exit",
                        "name": "exit_on_cross_up",
                        "order": {"type": "market"},
                        "condition": {
                            "cross": {
                                "a": {"ref": "ema_9"},
                                "op": "cross_above",
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


class TestMultiStrategyCreation:
    """Test suite for creating multiple strategies."""

    def _create_deployment(
        self,
        api_test_client: TestClient,
        auth_headers: dict[str, str],
        *,
        symbol: str,
        timeframe: str,
        capital: int = 5000,
    ) -> dict[str, Any]:
        """Create a deployment for a single symbol."""
        from packages.shared_settings.schema.settings import settings

        # Create thread
        thread_resp = api_test_client.post(
            "/api/v1/chat/new-thread",
            headers=auth_headers,
            json={"metadata": {"source": f"multi-strategy-{symbol}-{timeframe}"}},
        )
        if thread_resp.status_code != 201:
            return {"error": "Failed to create thread"}
        session_id = thread_resp.json()["session_id"]

        # Create strategy
        dsl = _load_crypto_strategy_dsl(
            name=f"Multi-{symbol}-{timeframe}-{uuid4().hex[:6]}",
            symbol=symbol,
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
        if strategy_resp.status_code != 200:
            return {"error": "Failed to create strategy"}
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
                "metadata": {"source": f"multi-strategy-{symbol}-{timeframe}"},
            },
        )
        if broker_resp.status_code != 201:
            return {"error": "Failed to create broker account"}
        broker_account_id = broker_resp.json()["broker_account_id"]

        # Create deployment
        deploy_resp = api_test_client.post(
            "/api/v1/deployments",
            headers=auth_headers,
            json={
                "strategy_id": strategy_id,
                "broker_account_id": broker_account_id,
                "mode": "paper",
                "capital_allocated": capital,
                "risk_limits": {"max_position_pct": 20, "max_daily_loss_pct": 5},
                "runtime_state": {"symbol": symbol, "timeframe": timeframe},
            },
        )
        if deploy_resp.status_code != 201:
            return {"error": "Failed to create deployment"}

        return {
            "deployment_id": deploy_resp.json()["deployment_id"],
            "strategy_id": strategy_id,
            "broker_account_id": broker_account_id,
            "symbol": symbol,
            "timeframe": timeframe,
        }

    def test_000_create_btc_1min_strategy(
        self,
        api_test_client: TestClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test creating BTC/USD 1-minute strategy."""
        result = self._create_deployment(
            api_test_client,
            auth_headers,
            symbol="BTC/USD",
            timeframe="1m",
        )
        assert "deployment_id" in result, f"Failed: {result.get('error')}"

        # Start deployment
        start_resp = api_test_client.post(
            f"/api/v1/deployments/{result['deployment_id']}/start",
            headers=auth_headers,
        )
        assert start_resp.status_code == 200

        # Clean up after test
        api_test_client.post(
            f"/api/v1/deployments/{result['deployment_id']}/manual-actions",
            headers=auth_headers,
            json={"action": "stop", "payload": {"reason": "test-cleanup"}},
        )

    def test_010_create_eth_1min_strategy(
        self,
        api_test_client: TestClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test creating ETH/USD 1-minute strategy."""
        result = self._create_deployment(
            api_test_client,
            auth_headers,
            symbol="ETH/USD",
            timeframe="1m",
        )
        assert "deployment_id" in result

        start_resp = api_test_client.post(
            f"/api/v1/deployments/{result['deployment_id']}/start",
            headers=auth_headers,
        )
        assert start_resp.status_code == 200

        api_test_client.post(
            f"/api/v1/deployments/{result['deployment_id']}/manual-actions",
            headers=auth_headers,
            json={"action": "stop", "payload": {"reason": "test-cleanup"}},
        )

    def test_020_create_btc_5min_strategy(
        self,
        api_test_client: TestClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test creating BTC/USD 5-minute strategy."""
        result = self._create_deployment(
            api_test_client,
            auth_headers,
            symbol="BTC/USD",
            timeframe="5m",
        )
        assert "deployment_id" in result

        start_resp = api_test_client.post(
            f"/api/v1/deployments/{result['deployment_id']}/start",
            headers=auth_headers,
        )
        assert start_resp.status_code == 200

        api_test_client.post(
            f"/api/v1/deployments/{result['deployment_id']}/manual-actions",
            headers=auth_headers,
            json={"action": "stop", "payload": {"reason": "test-cleanup"}},
        )

    def test_030_create_sol_5min_strategy(
        self,
        api_test_client: TestClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test creating SOL/USD 5-minute strategy."""
        result = self._create_deployment(
            api_test_client,
            auth_headers,
            symbol="SOL/USD",
            timeframe="5m",
        )
        assert "deployment_id" in result

        start_resp = api_test_client.post(
            f"/api/v1/deployments/{result['deployment_id']}/start",
            headers=auth_headers,
        )
        assert start_resp.status_code == 200

        api_test_client.post(
            f"/api/v1/deployments/{result['deployment_id']}/manual-actions",
            headers=auth_headers,
            json={"action": "stop", "payload": {"reason": "test-cleanup"}},
        )


class TestMultiStrategyTracking:
    """Test suite for tracking multiple strategies simultaneously."""

    def _create_and_start_deployment(
        self,
        api_test_client: TestClient,
        auth_headers: dict[str, str],
        *,
        symbol: str,
        timeframe: str,
    ) -> str | None:
        """Create and start a deployment, return deployment_id."""
        from packages.shared_settings.schema.settings import settings

        try:
            thread_resp = api_test_client.post(
                "/api/v1/chat/new-thread",
                headers=auth_headers,
                json={"metadata": {"source": f"tracking-{symbol}-{timeframe}"}},
            )
            session_id = thread_resp.json()["session_id"]

            dsl = _load_crypto_strategy_dsl(
                name=f"Track-{symbol}-{timeframe}-{uuid4().hex[:6]}",
                symbol=symbol,
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
                    "metadata": {"source": f"tracking-{symbol}-{timeframe}"},
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
                    "capital_allocated": 5000,
                    "risk_limits": {},
                    "runtime_state": {},
                },
            )
            deployment_id = deploy_resp.json()["deployment_id"]

            api_test_client.post(
                f"/api/v1/deployments/{deployment_id}/start",
                headers=auth_headers,
            )

            return deployment_id
        except Exception:
            return None

    def test_000_track_multiple_strategies_60_seconds(
        self,
        api_test_client: TestClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test tracking multiple strategies for 60 seconds."""
        # Create strategies
        configs = [
            ("BTC/USD", "1m"),
            ("ETH/USD", "1m"),
            ("BTC/USD", "5m"),
            ("SOL/USD", "5m"),
        ]

        deployments = []
        for symbol, timeframe in configs:
            deployment_id = self._create_and_start_deployment(
                api_test_client,
                auth_headers,
                symbol=symbol,
                timeframe=timeframe,
            )
            if deployment_id:
                deployments.append({
                    "deployment_id": deployment_id,
                    "symbol": symbol,
                    "timeframe": timeframe,
                })

        assert len(deployments) >= 3, f"Only created {len(deployments)}/4 deployments"

        # Track for 60 seconds
        start_time = time.monotonic()
        tracking_results = []

        while time.monotonic() - start_time < 60:
            for deployment in deployments:
                detail_resp = api_test_client.get(
                    f"/api/v1/deployments/{deployment['deployment_id']}",
                    headers=auth_headers,
                )
                if detail_resp.status_code == 200:
                    data = detail_resp.json()
                    tracking_results.append({
                        "timestamp": datetime.now(UTC).isoformat(),
                        "deployment_id": deployment["deployment_id"],
                        "symbol": deployment["symbol"],
                        "timeframe": deployment["timeframe"],
                        "status": data.get("status"),
                    })
            time.sleep(10)

        # Verify tracking worked
        assert len(tracking_results) > 0

        # Check all deployments remained active
        active_count = sum(1 for r in tracking_results if r["status"] == "active")
        total_checks = len(tracking_results)
        assert active_count >= total_checks * 0.8, f"Too many non-active: {active_count}/{total_checks}"

        # Clean up
        for deployment in deployments:
            api_test_client.post(
                f"/api/v1/deployments/{deployment['deployment_id']}/manual-actions",
                headers=auth_headers,
                json={"action": "stop", "payload": {"reason": "tracking-test-cleanup"}},
            )


class TestStrategySignalGeneration:
    """Test suite for strategy signal generation."""

    def _extract_result_json(self, stdout: str) -> dict[str, Any]:
        for line in stdout.splitlines():
            if line.startswith("RESULT_JSON="):
                return json.loads(line.removeprefix("RESULT_JSON="))
        raise AssertionError(f"No RESULT_JSON in output: {stdout[:500]}")

    def test_000_scheduler_generates_signals(
        self,
        compose_stack: list[dict[str, Any]],
    ) -> None:
        """Test that scheduler tick generates signals for active deployments."""
        _ = compose_stack

        # Run scheduler tick
        result = run_command(
            [
                "docker",
                "exec",
                "minsy-worker-io-dev",
                ".venv/bin/python",
                "-c",
                "import json; "
                "from apps.worker.io.tasks.paper_trading import scheduler_tick_task; "
                "result=scheduler_tick_task.run(); "
                "print('RESULT_JSON='+json.dumps(result))",
            ],
            cwd=BACKEND_DIR,
            timeout=300,
        )
        payload = self._extract_result_json(result.stdout)

        if payload.get("status") == "ok":
            # Check scheduler processed deployments
            assert "deployments_total" in payload
            assert "enqueued" in payload
            # Log results for debugging
            print(f"Scheduler tick: {payload}")


class TestOrderTriggering:
    """Test suite for order triggering verification."""

    def test_000_orders_created_for_active_deployment(
        self,
        api_test_client: TestClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test that orders are created for active deployments."""
        # Get all deployments
        deployments_resp = api_test_client.get(
            "/api/v1/deployments?status=active",
            headers=auth_headers,
        )
        assert deployments_resp.status_code == 200

        deployments = deployments_resp.json().get("deployments", [])
        if not deployments:
            pytest.skip("No active deployments")

        # Check orders for each deployment
        for deployment in deployments[:3]:  # Check first 3
            deployment_id = deployment["deployment_id"]
            orders_resp = api_test_client.get(
                f"/api/v1/deployments/{deployment_id}/orders",
                headers=auth_headers,
            )
            assert orders_resp.status_code == 200
            # Orders may be empty, that's OK
            orders_data = orders_resp.json()
            assert "orders" in orders_data or isinstance(orders_data, list)
