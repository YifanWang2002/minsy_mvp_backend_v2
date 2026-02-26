from __future__ import annotations

from fastapi.testclient import TestClient


def test_000_accessibility_trade_approvals_list(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    response = api_test_client.get(
        "/api/v1/trade-approvals",
        headers=auth_headers,
        params={"limit": 50},
    )
    assert response.status_code == 200, response.text
    assert isinstance(response.json(), list)


def test_010_trade_approvals_limit_filter(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    limit = 3
    response = api_test_client.get(
        "/api/v1/trade-approvals",
        headers=auth_headers,
        params={"limit": limit},
    )
    assert response.status_code == 200, response.text
    rows = response.json()
    assert isinstance(rows, list)
    assert len(rows) <= limit
