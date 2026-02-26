from __future__ import annotations

import os
import sys
from typing import Any

import pytest
from fastapi.testclient import TestClient

from test._support.live_helpers import (
    BACKEND_DIR,
    ensure_compose_stack_up,
    wait_http_ok,
)

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


@pytest.fixture(scope="session")
def compose_stack() -> list[dict[str, Any]]:
    rows = ensure_compose_stack_up()
    wait_http_ok("http://127.0.0.1:8000/api/v1/health", timeout_seconds=120)
    wait_http_ok("http://127.0.0.1:8000/api/v1/status", timeout_seconds=120)
    for domain in ("strategy", "backtest", "market", "stress", "trading"):
        wait_http_ok(
            f"http://127.0.0.1:8110/{domain}/mcp",
            timeout_seconds=120,
            min_status=200,
            max_status=499,
        )
    return rows


@pytest.fixture(scope="session")
def seeded_user_credentials() -> tuple[str, str]:
    return ("2@test.com", "123456")


@pytest.fixture(scope="function")
def api_test_client(compose_stack: list[dict[str, Any]]) -> TestClient:
    _ = compose_stack

    # Resolve local env files explicitly for deterministic local test runtime.
    os.environ["MINSY_ENV_FILES"] = ",".join(
        [
            "env/.env.secrets",
            "env/.env.common",
            "env/.env.dev",
            "env/.env.dev.api",
        ]
    )
    os.environ["MINSY_SERVICE"] = "api"
    # Stabilize local integration tests by removing shell-level proxy variables.
    # Some SDK clients (for example OpenAI httpx transport) auto-read these vars.
    for key in (
        "ALL_PROXY",
        "all_proxy",
        "HTTP_PROXY",
        "http_proxy",
        "HTTPS_PROXY",
        "https_proxy",
    ):
        os.environ.pop(key, None)

    from apps.api.main import app  # Imported lazily after env initialization.

    with TestClient(app) as client:
        yield client


@pytest.fixture(scope="function")
def auth_headers(
    api_test_client: TestClient,
    seeded_user_credentials: tuple[str, str],
) -> dict[str, str]:
    email, password = seeded_user_credentials
    response = api_test_client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": password},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    access_token = str(payload["access_token"])
    return {"Authorization": f"Bearer {access_token}"}
