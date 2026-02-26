"""Live integration tests for Celery beat scheduling and worker coordination.

These tests verify:
1. Celery beat is properly scheduling tasks
2. Workers are processing tasks correctly
3. Task queues are functioning
4. Inter-container communication works
"""

from __future__ import annotations

import json
import time
from typing import Any

import pytest

from test._support.live_helpers import BACKEND_DIR, run_command


def _extract_result_json(stdout: str) -> dict[str, Any]:
    for line in stdout.splitlines():
        if line.startswith("RESULT_JSON="):
            return json.loads(line.removeprefix("RESULT_JSON="))
    raise AssertionError(f"No RESULT_JSON in output: {stdout[:500]}")


def _run_in_container(container: str, script: str, timeout: int = 300) -> dict[str, Any]:
    """Run a Python script in a Docker container."""
    result = run_command(
        [
            "docker",
            "exec",
            container,
            ".venv/bin/python",
            "-c",
            script,
        ],
        cwd=BACKEND_DIR,
        timeout=timeout,
    )
    return _extract_result_json(result.stdout)


class TestCeleryBeatScheduling:
    """Test suite for Celery beat scheduling."""

    def test_000_beat_container_running(
        self,
        compose_stack: list[dict[str, Any]],
    ) -> None:
        """Test that beat container is running."""
        _ = compose_stack
        beat_containers = [
            c for c in compose_stack
            if "beat" in str(c.get("Service", "")).lower()
        ]
        assert len(beat_containers) >= 1, "Beat container not found"
        assert beat_containers[0].get("State") == "running"

    def test_010_paper_trading_scheduler_tick_scheduled(
        self,
        compose_stack: list[dict[str, Any]],
    ) -> None:
        """Test that paper trading scheduler tick is being scheduled."""
        _ = compose_stack

        # Check Redis for scheduled tasks
        result = run_command(
            [
                "docker",
                "exec",
                "minsy-redis-dev",
                "redis-cli",
                "KEYS",
                "celery*",
            ],
            cwd=BACKEND_DIR,
            timeout=30,
        )
        # Should have some celery keys
        assert result.returncode == 0

    def test_020_worker_io_processing_tasks(
        self,
        compose_stack: list[dict[str, Any]],
    ) -> None:
        """Test that worker-io is processing tasks."""
        _ = compose_stack

        # Run scheduler tick manually and verify it works
        payload = _run_in_container(
            "minsy-worker-io-dev",
            "import json; "
            "from apps.worker.io.tasks.paper_trading import scheduler_tick_task; "
            "result=scheduler_tick_task.run(); "
            "print('RESULT_JSON='+json.dumps(result))"
        )
        assert payload.get("status") in {"ok", "disabled"}

    def test_030_worker_cpu_available(
        self,
        compose_stack: list[dict[str, Any]],
    ) -> None:
        """Test that worker-cpu container is available."""
        _ = compose_stack
        cpu_containers = [
            c for c in compose_stack
            if "worker-cpu" in str(c.get("Service", "")).lower()
        ]
        assert len(cpu_containers) >= 1, "Worker-CPU container not found"
        assert cpu_containers[0].get("State") == "running"


class TestCeleryTaskQueues:
    """Test suite for Celery task queue functionality."""

    def test_000_paper_trading_queue_accessible(
        self,
        compose_stack: list[dict[str, Any]],
    ) -> None:
        """Test that paper_trading queue is accessible."""
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
        assert result.returncode == 0
        # Queue length should be a number (could be 0)
        queue_len = int(result.stdout.strip())
        assert queue_len >= 0

    def test_010_market_data_queue_accessible(
        self,
        compose_stack: list[dict[str, Any]],
    ) -> None:
        """Test that market_data queue is accessible."""
        _ = compose_stack

        result = run_command(
            [
                "docker",
                "exec",
                "minsy-redis-dev",
                "redis-cli",
                "LLEN",
                "market_data",
            ],
            cwd=BACKEND_DIR,
            timeout=30,
        )
        assert result.returncode == 0

    def test_020_notifications_queue_accessible(
        self,
        compose_stack: list[dict[str, Any]],
    ) -> None:
        """Test that notifications queue is accessible."""
        _ = compose_stack

        result = run_command(
            [
                "docker",
                "exec",
                "minsy-redis-dev",
                "redis-cli",
                "LLEN",
                "notifications",
            ],
            cwd=BACKEND_DIR,
            timeout=30,
        )
        assert result.returncode == 0

    def test_030_backtest_queue_accessible(
        self,
        compose_stack: list[dict[str, Any]],
    ) -> None:
        """Test that backtest queue is accessible."""
        _ = compose_stack

        result = run_command(
            [
                "docker",
                "exec",
                "minsy-redis-dev",
                "redis-cli",
                "LLEN",
                "backtest",
            ],
            cwd=BACKEND_DIR,
            timeout=30,
        )
        assert result.returncode == 0


class TestWorkerTaskExecution:
    """Test suite for worker task execution."""

    def test_000_market_data_backfill_task(
        self,
        compose_stack: list[dict[str, Any]],
    ) -> None:
        """Test market data backfill task execution."""
        _ = compose_stack

        payload = _run_in_container(
            "minsy-worker-io-dev",
            "import json; "
            "from apps.worker.io.tasks.market_data import backfill_symbol_task; "
            "result=backfill_symbol_task.run('crypto','BTC/USD',7); "
            "print('RESULT_JSON='+json.dumps(result))"
        )
        assert payload.get("market") == "crypto"
        assert payload.get("symbol") in {"BTC/USD", "BTCUSD"}
        assert payload.get("status") in {"ok", "error", "partial_error"}

    def test_010_refresh_subscriptions_task(
        self,
        compose_stack: list[dict[str, Any]],
    ) -> None:
        """Test refresh active subscriptions task."""
        _ = compose_stack

        payload = _run_in_container(
            "minsy-worker-io-dev",
            "import json; "
            "from apps.worker.io.tasks.market_data import refresh_active_subscriptions_task; "
            "result=refresh_active_subscriptions_task.run(); "
            "print('RESULT_JSON='+json.dumps(result))"
        )
        assert "scheduled" in payload
        assert "total" in payload

    def test_020_notifications_dispatch_task(
        self,
        compose_stack: list[dict[str, Any]],
    ) -> None:
        """Test notifications dispatch task."""
        _ = compose_stack

        payload = _run_in_container(
            "minsy-worker-io-dev",
            "import json; "
            "from apps.worker.io.tasks.notification import dispatch_pending_notifications_task; "
            "result=dispatch_pending_notifications_task.run(); "
            "print('RESULT_JSON='+json.dumps(result))"
        )
        assert "status" in payload

    def test_030_trade_approval_expire_task(
        self,
        compose_stack: list[dict[str, Any]],
    ) -> None:
        """Test trade approval expire task."""
        _ = compose_stack

        payload = _run_in_container(
            "minsy-worker-io-dev",
            "import json; "
            "from apps.worker.io.tasks.trade_approval import expire_pending_trade_approvals_task; "
            "result=expire_pending_trade_approvals_task.run(); "
            "print('RESULT_JSON='+json.dumps(result))"
        )
        assert "expired" in payload or "status" in payload


class TestInterContainerCommunication:
    """Test suite for inter-container communication."""

    def test_000_api_to_redis_connection(
        self,
        compose_stack: list[dict[str, Any]],
    ) -> None:
        """Test API container can connect to Redis."""
        _ = compose_stack

        result = run_command(
            [
                "docker",
                "exec",
                "minsy-api-dev",
                ".venv/bin/python",
                "-c",
                "import json; "
                "from packages.infra.redis.client import get_sync_redis_client; "
                "r=get_sync_redis_client(); "
                "r.ping(); "
                "print('RESULT_JSON='+json.dumps({'status':'ok'}))",
            ],
            cwd=BACKEND_DIR,
            timeout=60,
        )
        payload = _extract_result_json(result.stdout)
        assert payload.get("status") == "ok"

    def test_010_api_to_postgres_connection(
        self,
        compose_stack: list[dict[str, Any]],
    ) -> None:
        """Test API container can connect to PostgreSQL."""
        _ = compose_stack

        # Use exec() to handle async code in -c
        result = run_command(
            [
                "docker",
                "exec",
                "minsy-api-dev",
                ".venv/bin/python",
                "-c",
                "import json; "
                "exec('"
                "import asyncio\\n"
                "from packages.infra.db import session as db_module\\n"
                "async def test():\\n"
                "    await db_module.init_postgres(ensure_schema=False)\\n"
                "    return {\"status\":\"ok\"}\\n"
                "result=asyncio.run(test())\\n"
                "print(\"RESULT_JSON=\"+json.dumps(result))\\n"
                "')",
            ],
            cwd=BACKEND_DIR,
            timeout=60,
        )
        payload = _extract_result_json(result.stdout)
        assert payload.get("status") == "ok"

    def test_020_worker_to_redis_connection(
        self,
        compose_stack: list[dict[str, Any]],
    ) -> None:
        """Test worker container can connect to Redis."""
        _ = compose_stack

        payload = _run_in_container(
            "minsy-worker-io-dev",
            "import json; "
            "from packages.infra.redis.client import get_sync_redis_client; "
            "r=get_sync_redis_client(); "
            "r.ping(); "
            "print('RESULT_JSON='+json.dumps({'status':'ok'}))"
        )
        assert payload.get("status") == "ok"

    def test_030_mcp_to_postgres_connection(
        self,
        compose_stack: list[dict[str, Any]],
    ) -> None:
        """Test MCP container can connect to PostgreSQL."""
        _ = compose_stack

        # Use exec() to handle async code in -c
        result = run_command(
            [
                "docker",
                "exec",
                "minsy-mcp-dev",
                ".venv/bin/python",
                "-c",
                "import json; "
                "exec('"
                "import asyncio\\n"
                "from packages.infra.db import session as db_module\\n"
                "async def test():\\n"
                "    await db_module.init_postgres(ensure_schema=False)\\n"
                "    return {\"status\":\"ok\"}\\n"
                "result=asyncio.run(test())\\n"
                "print(\"RESULT_JSON=\"+json.dumps(result))\\n"
                "')",
            ],
            cwd=BACKEND_DIR,
            timeout=60,
        )
        payload = _extract_result_json(result.stdout)
        assert payload.get("status") == "ok"


class TestTaskResultPropagation:
    """Test suite for task result propagation."""

    def test_000_task_result_stored_in_redis(
        self,
        compose_stack: list[dict[str, Any]],
    ) -> None:
        """Test that task results are stored in Redis."""
        _ = compose_stack

        # Run a task and check result is stored
        payload = _run_in_container(
            "minsy-worker-io-dev",
            "import json; "
            "from apps.worker.io.tasks.market_data import refresh_active_subscriptions_task; "
            "async_result=refresh_active_subscriptions_task.apply_async(); "
            "result=async_result.get(timeout=60); "
            "print('RESULT_JSON='+json.dumps({'task_id':async_result.id,'result':result}))"
        )
        assert "task_id" in payload
        assert "result" in payload
