from __future__ import annotations

import asyncio
import json
import time
from decimal import Decimal
from uuid import uuid4

from packages.infra.providers.trading.adapters.alpaca_trading import AlpacaTradingAdapter
from packages.infra.providers.trading.adapters.base import AdapterError, OrderIntent
from test._support.live_helpers import BACKEND_DIR, run_command


def _extract_result_json(stdout: str) -> dict[str, object]:
    for line in stdout.splitlines():
        if line.startswith("RESULT_JSON="):
            return json.loads(line.removeprefix("RESULT_JSON="))
    raise AssertionError(f"No RESULT_JSON line in stdout: {stdout[:400]}")


def _inspect_active_queues_stdout() -> str:
    deadline = time.monotonic() + 90.0
    last_output = ""
    while time.monotonic() < deadline:
        result = run_command(
            [
                "docker",
                "exec",
                "minsy-worker-io-dev",
                ".venv/bin/celery",
                "-A",
                "apps.worker.io.celery_app:celery_app",
                "inspect",
                "active_queues",
            ],
            cwd=BACKEND_DIR,
            timeout=120,
            check=False,
        )
        combined = (result.stdout + "\n" + result.stderr).strip()
        last_output = combined
        if result.returncode == 0 and "worker-io@" in combined:
            return combined
        time.sleep(3)
    raise AssertionError(f"Celery inspect active_queues not ready: {last_output}")


def test_000_accessibility_worker_queues(compose_stack: list[dict[str, object]]) -> None:
    _ = compose_stack
    stdout = _inspect_active_queues_stdout()
    assert "worker-io@" in stdout
    assert "worker-cpu@" in stdout
    assert "market_data" in stdout
    assert "paper_trading" in stdout
    assert "backtest" in stdout


def test_010_worker_io_market_data_refresh_task_live(
    compose_stack: list[dict[str, object]],
) -> None:
    _ = compose_stack
    script = (
        "import json; "
        "from apps.worker.io.tasks.market_data import refresh_symbol_task; "
        "result = refresh_symbol_task.run('stocks', 'SPY'); "
        "print('RESULT_JSON=' + json.dumps(result))"
    )
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
        timeout=180,
    )
    payload = _extract_result_json(result.stdout)
    assert payload["market"] == "stocks"
    assert payload["symbol"] == "SPY"
    assert payload["status"] in {"ok", "partial_error"}


async def _submit_and_cleanup_paper_order() -> tuple[str, str]:
    adapter = AlpacaTradingAdapter()
    try:
        intent = OrderIntent(
            client_order_id=f"codex-test-{uuid4().hex[:20]}",
            symbol="SPY",
            side="buy",
            qty=Decimal("1"),
            order_type="market",
            time_in_force="day",
        )
        order = await adapter.submit_order(intent)
        provider_order_id = order.provider_order_id
        status = order.status

        fetched = await adapter.fetch_order(provider_order_id)
        if fetched is not None:
            status = fetched.status

        if fetched is not None and status not in {
            "filled",
            "canceled",
            "cancelled",
            "rejected",
            "expired",
        }:
            try:
                await adapter.cancel_order(provider_order_id)
            except AdapterError:
                # Order can fill before cancellation is processed.
                pass

        return provider_order_id, status
    finally:
        await adapter.aclose()


def test_020_alpaca_paper_trade_submission_live() -> None:
    provider_order_id, status = asyncio.run(_submit_and_cleanup_paper_order())
    assert provider_order_id
    assert status
