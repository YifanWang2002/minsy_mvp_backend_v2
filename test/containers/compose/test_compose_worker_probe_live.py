from __future__ import annotations

from test._support.live_helpers import BACKEND_DIR, run_command


def test_000_accessibility_worker_io_and_cpu_online(
    compose_stack: list[dict[str, object]],
) -> None:
    _ = compose_stack
    io_ping = run_command(
        [
            "docker",
            "exec",
            "minsy-worker-io-dev",
            ".venv/bin/celery",
            "-A",
            "apps.worker.io.celery_app:celery_app",
            "inspect",
            "ping",
        ],
        cwd=BACKEND_DIR,
        timeout=120,
    )
    assert "worker-io@" in io_ping.stdout

    cpu_ping = run_command(
        [
            "docker",
            "exec",
            "minsy-worker-cpu-dev",
            ".venv/bin/celery",
            "-A",
            "apps.worker.cpu.celery_app:celery_app",
            "inspect",
            "ping",
        ],
        cwd=BACKEND_DIR,
        timeout=120,
    )
    assert "worker-cpu@" in cpu_ping.stdout


def test_010_worker_queue_bindings_visible(
    compose_stack: list[dict[str, object]],
) -> None:
    _ = compose_stack
    queues = run_command(
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
    )
    stdout = queues.stdout
    assert "market_data" in stdout
    assert "paper_trading" in stdout
    assert "notifications" in stdout
    assert "trade_approval" in stdout
