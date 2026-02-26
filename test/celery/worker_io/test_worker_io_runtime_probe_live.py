from __future__ import annotations

import json
import time

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


def _inspect_stdout(command: str) -> str:
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
                command,
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
    raise AssertionError(f"Celery inspect {command} not ready: {last_output}")


def test_000_accessibility_celery_ping_and_stats(
    compose_stack: list[dict[str, object]],
) -> None:
    _ = compose_stack
    ping_output = _inspect_stdout("ping")
    assert "worker-io@" in ping_output

    stats_output = _inspect_stdout("stats")
    assert "worker-io@" in stats_output


def test_010_scheduler_and_notification_tasks_live(
    compose_stack: list[dict[str, object]],
) -> None:
    _ = compose_stack
    scheduler_payload = _run_worker_script(
        "import json; "
        "from apps.worker.io.tasks.paper_trading import scheduler_tick_task; "
        "result=scheduler_tick_task.run(); "
        "print('RESULT_JSON='+json.dumps(result))"
    )
    assert scheduler_payload.get("status") == "ok"
    assert int(scheduler_payload.get("deployments_total", 0)) >= 0

    notification_payload = _run_worker_script(
        "import json; "
        "from apps.worker.io.tasks.notification import dispatch_pending_notifications_task; "
        "result=dispatch_pending_notifications_task.run(); "
        "print('RESULT_JSON='+json.dumps(result))"
    )
    assert notification_payload.get("status") == "ok"
    assert int(notification_payload.get("picked", 0)) >= 0
