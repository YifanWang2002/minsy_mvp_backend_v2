from __future__ import annotations

import json

from test._support.live_helpers import BACKEND_DIR, run_command


def _extract_result_json(stdout: str) -> dict[str, object]:
    for line in stdout.splitlines():
        if line.startswith("RESULT_JSON="):
            return json.loads(line.removeprefix("RESULT_JSON="))
    raise AssertionError(f"No RESULT_JSON in output: {stdout[:500]}")


def _run_worker_script(script: str) -> dict[str, object]:
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
        timeout=240,
    )
    return _extract_result_json(result.stdout)


def test_000_accessibility_market_data_backfill_task_live(
    compose_stack: list[dict[str, object]],
) -> None:
    _ = compose_stack
    payload = _run_worker_script(
        "import json; "
        "from apps.worker.io.tasks.market_data import backfill_symbol_task; "
        "result=backfill_symbol_task.run('stocks','SPY',30); "
        "print('RESULT_JSON='+json.dumps(result))"
    )
    assert payload["market"] == "stocks"
    assert payload["symbol"] == "SPY"
    assert payload["status"] in {"ok", "error", "partial_error"}


def test_010_refresh_active_subscriptions_task_live(
    compose_stack: list[dict[str, object]],
) -> None:
    _ = compose_stack
    payload = _run_worker_script(
        "import json; "
        "from apps.worker.io.tasks.market_data import refresh_active_subscriptions_task; "
        "result=refresh_active_subscriptions_task.run(); "
        "print('RESULT_JSON='+json.dumps(result))"
    )
    assert set(payload.keys()) >= {"scheduled", "deduped", "total"}
    assert int(payload["scheduled"]) >= 0
    assert int(payload["deduped"]) >= 0
    assert int(payload["total"]) >= 0
