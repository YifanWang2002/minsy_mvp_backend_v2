from __future__ import annotations

from test._support.live_helpers import (
    BACKEND_DIR,
    compose_ps,
    run_command,
    wait_http_ok,
)


def test_000_accessibility_compose_services_running(
    compose_stack: list[dict[str, object]],
) -> None:
    _ = compose_stack
    rows = compose_ps()
    by_service = {
        str(row.get("Service", "")): row
        for row in rows
        if isinstance(row, dict)
    }

    expected_services = {
        "postgres",
        "redis",
        "mcp",
        "api",
        "worker-cpu",
        "worker-io",
        "beat",
    }
    assert expected_services.issubset(set(by_service.keys()))

    for service_name in expected_services:
        state = str(by_service[service_name].get("State", "")).lower()
        assert state == "running", (service_name, by_service[service_name])


def test_010_compose_endpoints_are_reachable(
    compose_stack: list[dict[str, object]],
) -> None:
    _ = compose_stack
    health_code = wait_http_ok("http://127.0.0.1:8000/api/v1/health", timeout_seconds=120)
    assert health_code == 200

    status_code = wait_http_ok("http://127.0.0.1:8000/api/v1/status", timeout_seconds=120)
    assert status_code == 200

    for domain in ("strategy", "backtest", "market", "stress", "trading"):
        code = wait_http_ok(
            f"http://127.0.0.1:8110/{domain}/mcp",
            timeout_seconds=120,
            min_status=200,
            max_status=499,
        )
        assert code < 500, (domain, code)


def test_020_compose_logs_no_fatal_runtime_errors(
    compose_stack: list[dict[str, object]],
) -> None:
    _ = compose_stack
    result = run_command(
        [
            "docker",
            "compose",
            "-f",
            "compose.dev.yml",
            "logs",
            "--no-color",
            "--tail",
            "200",
            "api",
            "mcp",
            "worker-cpu",
            "worker-io",
            "beat",
        ],
        cwd=BACKEND_DIR,
    )
    logs = result.stdout.lower()
    fatal_markers = (
        "traceback (most recent call last)",
        "child exited unexpectedly",
        "error: unable to",
        "fatal:",
    )
    for marker in fatal_markers:
        assert marker not in logs, marker
