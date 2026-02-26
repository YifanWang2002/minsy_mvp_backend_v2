from __future__ import annotations

from fastapi.testclient import TestClient


def test_000_accessibility_backtests_demo_ensure(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    response = api_test_client.post(
        "/api/v1/backtests/jobs/demo/ensure",
        headers=auth_headers,
    )
    assert response.status_code in {200, 404, 409}, response.text


def test_010_backtests_demo_ensure_success_payload_contract(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    response = api_test_client.post(
        "/api/v1/backtests/jobs/demo/ensure",
        headers=auth_headers,
    )
    if response.status_code != 200:
        assert response.status_code in {404, 409}, response.text
        return

    payload = response.json()
    assert payload["job_id"]
    assert payload["strategy_id"]
    assert payload["status"] in {"pending", "running", "done", "failed"}
