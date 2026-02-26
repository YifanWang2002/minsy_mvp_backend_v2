"""Stress tests for architecture capacity.

These tests evaluate:
1. Maximum number of concurrent deployments
2. Maximum number of symbols per deployment
3. Maximum number of concurrent users
4. System behavior under load
5. Resource utilization limits
"""

from __future__ import annotations

import asyncio
import concurrent.futures
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
            "description": f"Stress test strategy for {len(symbols)} symbols",
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
                    "pct": 0.05,
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


class TestDeploymentScaling:
    """Test suite for deployment scaling limits."""

    def _create_deployment(
        self,
        api_test_client: TestClient,
        auth_headers: dict[str, str],
        *,
        symbols: list[str],
        timeframe: str,
        index: int,
    ) -> str | None:
        """Create a single deployment."""
        from packages.shared_settings.schema.settings import settings

        try:
            # Create thread
            thread_resp = api_test_client.post(
                "/api/v1/chat/new-thread",
                headers=auth_headers,
                json={"metadata": {"source": f"stress-test-{index}"}},
            )
            if thread_resp.status_code != 201:
                return None
            session_id = thread_resp.json()["session_id"]

            # Create strategy
            dsl = _load_crypto_strategy_dsl(
                name=f"Stress-{index}-{uuid4().hex[:6]}",
                symbols=symbols,
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
                return None
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
                    "metadata": {"source": f"stress-test-{index}"},
                },
            )
            if broker_resp.status_code != 201:
                return None
            broker_account_id = broker_resp.json()["broker_account_id"]

            # Create deployment
            deploy_resp = api_test_client.post(
                "/api/v1/deployments",
                headers=auth_headers,
                json={
                    "strategy_id": strategy_id,
                    "broker_account_id": broker_account_id,
                    "mode": "paper",
                    "capital_allocated": 1000,
                    "risk_limits": {"max_position_pct": 5},
                    "runtime_state": {"stress_test_index": index},
                },
            )
            if deploy_resp.status_code != 201:
                return None
            return deploy_resp.json()["deployment_id"]
        except Exception:
            return None

    def test_000_create_10_deployments(
        self,
        api_test_client: TestClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test creating 10 concurrent deployments."""
        deployments = []
        symbols = ["BTC/USD"]

        for i in range(10):
            deployment_id = self._create_deployment(
                api_test_client,
                auth_headers,
                symbols=symbols,
                timeframe="1m",
                index=i,
            )
            if deployment_id:
                deployments.append(deployment_id)

        assert len(deployments) >= 8, f"Only created {len(deployments)}/10 deployments"

        # Start all deployments
        started = 0
        for deployment_id in deployments:
            resp = api_test_client.post(
                f"/api/v1/deployments/{deployment_id}/start",
                headers=auth_headers,
            )
            if resp.status_code == 200:
                started += 1

        assert started >= 8, f"Only started {started}/{len(deployments)} deployments"

        # Wait for scheduler
        time.sleep(5)

        # Verify all are active
        active = 0
        for deployment_id in deployments:
            detail = api_test_client.get(
                f"/api/v1/deployments/{deployment_id}",
                headers=auth_headers,
            )
            if detail.status_code == 200 and detail.json()["status"] == "active":
                active += 1

        assert active >= 8, f"Only {active}/{len(deployments)} deployments active"

        # Clean up
        for deployment_id in deployments:
            api_test_client.post(
                f"/api/v1/deployments/{deployment_id}/manual-actions",
                headers=auth_headers,
                json={"action": "stop", "payload": {"reason": "stress-test-cleanup"}},
            )

    def test_010_create_20_deployments(
        self,
        api_test_client: TestClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test creating 20 concurrent deployments."""
        deployments = []
        symbols = ["BTC/USD", "ETH/USD"]

        for i in range(20):
            deployment_id = self._create_deployment(
                api_test_client,
                auth_headers,
                symbols=symbols,
                timeframe="5m",
                index=i,
            )
            if deployment_id:
                deployments.append(deployment_id)

        assert len(deployments) >= 15, f"Only created {len(deployments)}/20 deployments"

        # Start all
        for deployment_id in deployments:
            api_test_client.post(
                f"/api/v1/deployments/{deployment_id}/start",
                headers=auth_headers,
            )

        time.sleep(10)

        # Clean up
        for deployment_id in deployments:
            api_test_client.post(
                f"/api/v1/deployments/{deployment_id}/manual-actions",
                headers=auth_headers,
                json={"action": "stop", "payload": {"reason": "stress-test-cleanup"}},
            )


class TestSymbolScaling:
    """Test suite for symbol scaling limits."""

    def test_000_multi_symbol_deployment_5_symbols(
        self,
        api_test_client: TestClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test deployment with 5 symbols."""
        from packages.shared_settings.schema.settings import settings

        symbols = ["BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD", "AVAX/USD"]

        # Create thread
        thread_resp = api_test_client.post(
            "/api/v1/chat/new-thread",
            headers=auth_headers,
            json={"metadata": {"source": "stress-test-5-symbols"}},
        )
        session_id = thread_resp.json()["session_id"]

        # Create strategy
        dsl = _load_crypto_strategy_dsl(
            name=f"Stress-5sym-{uuid4().hex[:6]}",
            symbols=symbols,
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
        assert strategy_resp.status_code == 200
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
                "metadata": {"source": "stress-test-5-symbols"},
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
        assert deploy_resp.status_code == 201
        deployment_id = deploy_resp.json()["deployment_id"]

        # Start and verify
        start_resp = api_test_client.post(
            f"/api/v1/deployments/{deployment_id}/start",
            headers=auth_headers,
        )
        assert start_resp.status_code == 200

        time.sleep(5)

        detail = api_test_client.get(
            f"/api/v1/deployments/{deployment_id}",
            headers=auth_headers,
        )
        assert detail.json()["status"] == "active"

        # Clean up
        api_test_client.post(
            f"/api/v1/deployments/{deployment_id}/manual-actions",
            headers=auth_headers,
            json={"action": "stop", "payload": {"reason": "stress-test-cleanup"}},
        )


class TestConcurrentRequests:
    """Test suite for concurrent request handling."""

    def test_000_concurrent_api_requests(
        self,
        api_test_client: TestClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test handling 50 concurrent API requests."""
        import threading

        results = []
        errors = []

        def make_request(index: int):
            try:
                resp = api_test_client.get(
                    "/api/v1/health",
                    headers=auth_headers,
                )
                results.append(resp.status_code)
            except Exception as e:
                errors.append(str(e))

        threads = []
        for i in range(50):
            t = threading.Thread(target=make_request, args=(i,))
            threads.append(t)

        # Start all threads
        for t in threads:
            t.start()

        # Wait for completion
        for t in threads:
            t.join(timeout=30)

        # Verify results
        success_count = sum(1 for r in results if r == 200)
        assert success_count >= 45, f"Only {success_count}/50 requests succeeded"
        assert len(errors) < 5, f"Too many errors: {errors}"

    def test_010_concurrent_deployment_listing(
        self,
        api_test_client: TestClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test concurrent deployment listing requests."""
        import threading

        results = []

        def list_deployments(index: int):
            try:
                resp = api_test_client.get(
                    "/api/v1/deployments",
                    headers=auth_headers,
                )
                results.append(resp.status_code)
            except Exception:
                results.append(500)

        threads = []
        for i in range(30):
            t = threading.Thread(target=list_deployments, args=(i,))
            threads.append(t)

        for t in threads:
            t.start()

        for t in threads:
            t.join(timeout=30)

        success_count = sum(1 for r in results if r == 200)
        assert success_count >= 25, f"Only {success_count}/30 requests succeeded"


class TestResourceUtilization:
    """Test suite for resource utilization monitoring."""

    def test_000_container_memory_usage(
        self,
        compose_stack: list[dict[str, Any]],
    ) -> None:
        """Test container memory usage is within limits."""
        _ = compose_stack

        result = run_command(
            [
                "docker",
                "stats",
                "--no-stream",
                "--format",
                "{{.Name}}\t{{.MemUsage}}\t{{.MemPerc}}",
            ],
            cwd=BACKEND_DIR,
            timeout=30,
        )

        lines = result.stdout.strip().split("\n")
        for line in lines:
            if "minsy" in line.lower():
                parts = line.split("\t")
                if len(parts) >= 3:
                    mem_pct = parts[2].replace("%", "").strip()
                    try:
                        pct = float(mem_pct)
                        # Memory usage should be under 80%
                        assert pct < 80, f"Container {parts[0]} using {pct}% memory"
                    except ValueError:
                        pass

    def test_010_container_cpu_usage(
        self,
        compose_stack: list[dict[str, Any]],
    ) -> None:
        """Test container CPU usage is reasonable."""
        _ = compose_stack

        result = run_command(
            [
                "docker",
                "stats",
                "--no-stream",
                "--format",
                "{{.Name}}\t{{.CPUPerc}}",
            ],
            cwd=BACKEND_DIR,
            timeout=30,
        )

        lines = result.stdout.strip().split("\n")
        for line in lines:
            if "minsy" in line.lower():
                parts = line.split("\t")
                if len(parts) >= 2:
                    cpu_pct = parts[1].replace("%", "").strip()
                    try:
                        pct = float(cpu_pct)
                        # CPU usage should be under 90% at idle
                        assert pct < 90, f"Container {parts[0]} using {pct}% CPU"
                    except ValueError:
                        pass


class TestQueueBackpressure:
    """Test suite for queue backpressure handling."""

    def _extract_result_json(self, stdout: str) -> dict[str, Any]:
        for line in stdout.splitlines():
            if line.startswith("RESULT_JSON="):
                return json.loads(line.removeprefix("RESULT_JSON="))
        raise AssertionError(f"No RESULT_JSON in output: {stdout[:500]}")

    def test_000_queue_backlog_monitoring(
        self,
        compose_stack: list[dict[str, Any]],
    ) -> None:
        """Test queue backlog is monitored."""
        _ = compose_stack

        result = run_command(
            [
                "docker",
                "exec",
                "minsy-redis-dev",
                "redis-cli",
                "LLEN",
                "paper_trading",
            ],
            cwd=BACKEND_DIR,
            timeout=30,
        )
        queue_len = int(result.stdout.strip())
        # Queue should not have excessive backlog
        assert queue_len < 1000, f"Paper trading queue backlog: {queue_len}"

    def test_010_scheduler_handles_backlog(
        self,
        compose_stack: list[dict[str, Any]],
    ) -> None:
        """Test scheduler handles queue backlog gracefully."""
        _ = compose_stack

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
            # Check backlog handling
            backlog = payload.get("queue_backlog", 0)
            overloaded = payload.get("backlog_overloaded", False)
            if backlog > 100:
                # Should be handling backlog
                assert overloaded or payload.get("skipped", 0) > 0


class TestLongRunningStability:
    """Test suite for long-running stability."""

    def test_000_sustained_deployment_operation(
        self,
        api_test_client: TestClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test sustained deployment operation over time."""
        from packages.shared_settings.schema.settings import settings

        # Create a deployment
        thread_resp = api_test_client.post(
            "/api/v1/chat/new-thread",
            headers=auth_headers,
            json={"metadata": {"source": "stress-test-sustained"}},
        )
        session_id = thread_resp.json()["session_id"]

        dsl = _load_crypto_strategy_dsl(
            name=f"Sustained-{uuid4().hex[:6]}",
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
                "metadata": {"source": "stress-test-sustained"},
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
                "runtime_state": {},
            },
        )
        deployment_id = deploy_resp.json()["deployment_id"]

        # Start deployment
        api_test_client.post(
            f"/api/v1/deployments/{deployment_id}/start",
            headers=auth_headers,
        )

        # Monitor for 30 seconds
        start_time = time.monotonic()
        checks = 0
        failures = 0

        while time.monotonic() - start_time < 30:
            detail = api_test_client.get(
                f"/api/v1/deployments/{deployment_id}",
                headers=auth_headers,
            )
            checks += 1
            if detail.status_code != 200 or detail.json()["status"] not in {"active", "paused"}:
                failures += 1
            time.sleep(2)

        # Should have minimal failures
        assert failures < checks * 0.1, f"Too many failures: {failures}/{checks}"

        # Clean up
        api_test_client.post(
            f"/api/v1/deployments/{deployment_id}/manual-actions",
            headers=auth_headers,
            json={"action": "stop", "payload": {"reason": "stress-test-cleanup"}},
        )
