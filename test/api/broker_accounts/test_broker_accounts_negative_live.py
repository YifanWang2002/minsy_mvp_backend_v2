from __future__ import annotations

from uuid import uuid4

from fastapi.testclient import TestClient


def test_000_accessibility_broker_account_requires_auth(
    api_test_client: TestClient,
) -> None:
    response = api_test_client.get("/api/v1/broker-accounts")
    assert response.status_code == 401


def test_010_create_broker_account_empty_credentials_rejected(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    response = api_test_client.post(
        "/api/v1/broker-accounts?validate=false",
        headers=auth_headers,
        json={
            "provider": "alpaca",
            "mode": "paper",
            "credentials": {},
            "metadata": {"source": "pytest-live"},
        },
    )
    assert response.status_code == 422, response.text


def test_020_missing_broker_account_validate_returns_404(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    response = api_test_client.post(
        f"/api/v1/broker-accounts/{uuid4()}/validate",
        headers=auth_headers,
    )
    assert response.status_code == 404
