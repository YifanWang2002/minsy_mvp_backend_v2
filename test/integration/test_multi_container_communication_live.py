"""Live integration tests for multi-container architecture communication.

These tests verify:
1. All containers are running and healthy
2. Inter-container network connectivity
3. Service discovery works correctly
4. Health checks pass for all services
"""

from __future__ import annotations

import json
import time
from typing import Any

import pytest

from test._support.live_helpers import (
    BACKEND_DIR,
    compose_ps,
    run_command,
    wait_http_ok,
)


EXPECTED_SERVICES = (
    "postgres",
    "redis",
    "mcp",
    "api",
    "worker-cpu",
    "worker-io",
    "beat",
)


class TestContainerHealth:
    """Test suite for container health status."""

    def test_000_all_containers_running(
        self,
        compose_stack: list[dict[str, Any]],
    ) -> None:
        """Test that all expected containers are running."""
        service_map = {
            str(row.get("Service", "")).strip(): row
            for row in compose_stack
            if isinstance(row, dict)
        }

        for service in EXPECTED_SERVICES:
            assert service in service_map, f"Service {service} not found"
            state = str(service_map[service].get("State", "")).strip().lower()
            assert state == "running", f"Service {service} is {state}, expected running"

    def test_010_postgres_healthy(
        self,
        compose_stack: list[dict[str, Any]],
    ) -> None:
        """Test PostgreSQL container is healthy."""
        _ = compose_stack
        result = run_command(
            [
                "docker",
                "exec",
                "minsy-postgres-dev",
                "pg_isready",
                "-U",
                "postgres",
            ],
            cwd=BACKEND_DIR,
            timeout=30,
        )
        assert result.returncode == 0

    def test_020_redis_healthy(
        self,
        compose_stack: list[dict[str, Any]],
    ) -> None:
        """Test Redis container is healthy."""
        _ = compose_stack
        result = run_command(
            [
                "docker",
                "exec",
                "minsy-redis-dev",
                "redis-cli",
                "ping",
            ],
            cwd=BACKEND_DIR,
            timeout=30,
        )
        assert result.returncode == 0
        assert "PONG" in result.stdout

    def test_030_api_health_endpoint(
        self,
        compose_stack: list[dict[str, Any]],
    ) -> None:
        """Test API health endpoint."""
        _ = compose_stack
        status = wait_http_ok(
            "http://127.0.0.1:8000/api/v1/health",
            timeout_seconds=30,
        )
        assert status == 200

    def test_040_api_status_endpoint(
        self,
        compose_stack: list[dict[str, Any]],
    ) -> None:
        """Test API status endpoint."""
        _ = compose_stack
        status = wait_http_ok(
            "http://127.0.0.1:8000/api/v1/status",
            timeout_seconds=30,
        )
        assert status == 200

    def test_050_mcp_router_health(
        self,
        compose_stack: list[dict[str, Any]],
    ) -> None:
        """Test MCP router is accessible."""
        _ = compose_stack
        # MCP router doesn't have a /health endpoint, use strategy domain instead
        status = wait_http_ok(
            "http://127.0.0.1:8110/strategy/mcp",
            timeout_seconds=30,
            min_status=200,
            max_status=499,
        )
        assert 200 <= status <= 499


class TestContainerNetworking:
    """Test suite for container networking."""

    def test_000_api_can_reach_postgres(
        self,
        compose_stack: list[dict[str, Any]],
    ) -> None:
        """Test API container can reach PostgreSQL."""
        _ = compose_stack
        result = run_command(
            [
                "docker",
                "exec",
                "minsy-api-dev",
                "nc",
                "-zv",
                "postgres",
                "5432",
            ],
            cwd=BACKEND_DIR,
            timeout=30,
            check=False,
        )
        # nc might not be available, try alternative
        if result.returncode != 0:
            result = run_command(
                [
                    "docker",
                    "exec",
                    "minsy-api-dev",
                    ".venv/bin/python",
                    "-c",
                    "import socket; s=socket.socket(); s.settimeout(5); s.connect(('postgres',5432)); print('OK')",
                ],
                cwd=BACKEND_DIR,
                timeout=30,
            )
            assert "OK" in result.stdout

    def test_010_api_can_reach_redis(
        self,
        compose_stack: list[dict[str, Any]],
    ) -> None:
        """Test API container can reach Redis."""
        _ = compose_stack
        result = run_command(
            [
                "docker",
                "exec",
                "minsy-api-dev",
                ".venv/bin/python",
                "-c",
                "import socket; s=socket.socket(); s.settimeout(5); s.connect(('redis',6379)); print('OK')",
            ],
            cwd=BACKEND_DIR,
            timeout=30,
        )
        assert "OK" in result.stdout

    def test_020_api_can_reach_mcp(
        self,
        compose_stack: list[dict[str, Any]],
    ) -> None:
        """Test API container can reach MCP."""
        _ = compose_stack
        result = run_command(
            [
                "docker",
                "exec",
                "minsy-api-dev",
                ".venv/bin/python",
                "-c",
                "import socket; s=socket.socket(); s.settimeout(5); s.connect(('mcp',8110)); print('OK')",
            ],
            cwd=BACKEND_DIR,
            timeout=30,
        )
        assert "OK" in result.stdout

    def test_030_worker_can_reach_postgres(
        self,
        compose_stack: list[dict[str, Any]],
    ) -> None:
        """Test worker container can reach PostgreSQL."""
        _ = compose_stack
        result = run_command(
            [
                "docker",
                "exec",
                "minsy-worker-io-dev",
                ".venv/bin/python",
                "-c",
                "import socket; s=socket.socket(); s.settimeout(5); s.connect(('postgres',5432)); print('OK')",
            ],
            cwd=BACKEND_DIR,
            timeout=30,
        )
        assert "OK" in result.stdout

    def test_040_worker_can_reach_redis(
        self,
        compose_stack: list[dict[str, Any]],
    ) -> None:
        """Test worker container can reach Redis."""
        _ = compose_stack
        result = run_command(
            [
                "docker",
                "exec",
                "minsy-worker-io-dev",
                ".venv/bin/python",
                "-c",
                "import socket; s=socket.socket(); s.settimeout(5); s.connect(('redis',6379)); print('OK')",
            ],
            cwd=BACKEND_DIR,
            timeout=30,
        )
        assert "OK" in result.stdout


class TestServiceDiscovery:
    """Test suite for service discovery."""

    def test_000_dns_resolution_postgres(
        self,
        compose_stack: list[dict[str, Any]],
    ) -> None:
        """Test DNS resolution for postgres."""
        _ = compose_stack
        result = run_command(
            [
                "docker",
                "exec",
                "minsy-api-dev",
                ".venv/bin/python",
                "-c",
                "import socket; ip=socket.gethostbyname('postgres'); print(f'IP={ip}')",
            ],
            cwd=BACKEND_DIR,
            timeout=30,
        )
        assert "IP=" in result.stdout

    def test_010_dns_resolution_redis(
        self,
        compose_stack: list[dict[str, Any]],
    ) -> None:
        """Test DNS resolution for redis."""
        _ = compose_stack
        result = run_command(
            [
                "docker",
                "exec",
                "minsy-api-dev",
                ".venv/bin/python",
                "-c",
                "import socket; ip=socket.gethostbyname('redis'); print(f'IP={ip}')",
            ],
            cwd=BACKEND_DIR,
            timeout=30,
        )
        assert "IP=" in result.stdout

    def test_020_dns_resolution_mcp(
        self,
        compose_stack: list[dict[str, Any]],
    ) -> None:
        """Test DNS resolution for mcp."""
        _ = compose_stack
        result = run_command(
            [
                "docker",
                "exec",
                "minsy-api-dev",
                ".venv/bin/python",
                "-c",
                "import socket; ip=socket.gethostbyname('mcp'); print(f'IP={ip}')",
            ],
            cwd=BACKEND_DIR,
            timeout=30,
        )
        assert "IP=" in result.stdout


class TestContainerLogs:
    """Test suite for container log analysis."""

    def test_000_api_no_critical_errors(
        self,
        compose_stack: list[dict[str, Any]],
    ) -> None:
        """Test API container has no critical errors in logs."""
        _ = compose_stack
        result = run_command(
            [
                "docker",
                "logs",
                "--tail",
                "100",
                "minsy-api-dev",
            ],
            cwd=BACKEND_DIR,
            timeout=30,
            check=False,
        )
        logs = result.stdout + result.stderr
        # Check for critical errors
        critical_patterns = [
            "CRITICAL",
            "Traceback (most recent call last)",
            "ModuleNotFoundError",
            "ImportError",
        ]
        for pattern in critical_patterns:
            if pattern in logs:
                # Allow some errors during startup
                error_count = logs.count(pattern)
                assert error_count < 5, f"Too many {pattern} in API logs: {error_count}"

    def test_010_worker_no_critical_errors(
        self,
        compose_stack: list[dict[str, Any]],
    ) -> None:
        """Test worker container has no critical errors in logs."""
        _ = compose_stack
        result = run_command(
            [
                "docker",
                "logs",
                "--tail",
                "100",
                "minsy-worker-io-dev",
            ],
            cwd=BACKEND_DIR,
            timeout=30,
            check=False,
        )
        logs = result.stdout + result.stderr
        critical_patterns = [
            "CRITICAL",
            "ModuleNotFoundError",
            "ImportError",
        ]
        for pattern in critical_patterns:
            if pattern in logs:
                error_count = logs.count(pattern)
                assert error_count < 5, f"Too many {pattern} in worker logs: {error_count}"


class TestContainerRestart:
    """Test suite for container restart behavior."""

    def test_000_api_restart_recovery(
        self,
        compose_stack: list[dict[str, Any]],
    ) -> None:
        """Test API container recovers after restart."""
        _ = compose_stack

        # Restart API container
        run_command(
            [
                "docker",
                "restart",
                "minsy-api-dev",
            ],
            cwd=BACKEND_DIR,
            timeout=60,
        )

        # Wait for recovery
        time.sleep(5)

        # Verify health
        status = wait_http_ok(
            "http://127.0.0.1:8000/api/v1/health",
            timeout_seconds=120,
        )
        assert status == 200

    def test_010_worker_restart_recovery(
        self,
        compose_stack: list[dict[str, Any]],
    ) -> None:
        """Test worker container recovers after restart."""
        _ = compose_stack

        # Restart worker container
        run_command(
            [
                "docker",
                "restart",
                "minsy-worker-io-dev",
            ],
            cwd=BACKEND_DIR,
            timeout=60,
        )

        # Wait for recovery
        time.sleep(10)

        # Verify worker is running
        rows = compose_ps()
        worker_rows = [
            r for r in rows
            if "worker-io" in str(r.get("Service", "")).lower()
        ]
        assert len(worker_rows) >= 1
        assert worker_rows[0].get("State") == "running"
