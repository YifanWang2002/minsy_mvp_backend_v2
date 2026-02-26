from __future__ import annotations

from test._support.live_helpers import BACKEND_DIR, run_command, wait_http_ok


def test_000_accessibility_restart_mcp_service_recovers(
    compose_stack: list[dict[str, object]],
) -> None:
    _ = compose_stack

    run_command(
        ["docker", "compose", "-f", "compose.dev.yml", "restart", "mcp"],
        cwd=BACKEND_DIR,
        timeout=300,
    )

    # Wait until MCP router is reachable again after restart.
    for domain in ("strategy", "backtest", "market", "stress", "trading"):
        code = wait_http_ok(
            f"http://127.0.0.1:8110/{domain}/mcp",
            timeout_seconds=180,
            min_status=200,
            max_status=499,
        )
        assert code < 500


def test_010_post_restart_api_health_still_ok(
    compose_stack: list[dict[str, object]],
) -> None:
    _ = compose_stack
    health_code = wait_http_ok("http://127.0.0.1:8000/api/v1/health", timeout_seconds=180)
    status_code = wait_http_ok("http://127.0.0.1:8000/api/v1/status", timeout_seconds=180)
    assert health_code == 200
    assert status_code == 200
