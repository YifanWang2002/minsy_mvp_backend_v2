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
        "/api/v1/broker-accounts",
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


def test_030_duplicate_identity_upserts_existing_account(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    identity = f"dup-{uuid4().hex[:12]}"
    first = api_test_client.post(
        "/api/v1/broker-accounts",
        headers=auth_headers,
        json={
            "provider": "sandbox",
            "mode": "paper",
            "exchange_id": "sandbox",
            "account_uid": identity,
            "credentials": {"sandbox_token": f"k-{uuid4().hex[:8]}"},
            "metadata": {"source": "pytest-live", "case": "duplicate-active-identity"},
        },
    )
    assert first.status_code == 201, first.text
    first_id = first.json()["broker_account_id"]

    duplicate = api_test_client.post(
        "/api/v1/broker-accounts",
        headers=auth_headers,
        json={
            "provider": "sandbox",
            "mode": "paper",
            "exchange_id": "sandbox",
            "account_uid": identity,
            "credentials": {"sandbox_token": f"s-{uuid4().hex[:8]}"},
            "metadata": {"source": "pytest-live", "case": "duplicate-active-identity"},
        },
    )
    assert duplicate.status_code == 201, duplicate.text
    payload = duplicate.json()
    assert payload["broker_account_id"] == first_id
    assert payload["status"] == "active"

    api_test_client.post(
        f"/api/v1/broker-accounts/{first_id}/deactivate",
        headers=auth_headers,
    )


def test_040_failed_validation_does_not_persist_account(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    before = api_test_client.get("/api/v1/broker-accounts", headers=auth_headers)
    assert before.status_code == 200, before.text
    before_ids = {
        str(item["broker_account_id"])
        for item in before.json()
        if isinstance(item, dict) and item.get("broker_account_id")
    }

    response = api_test_client.post(
        "/api/v1/broker-accounts",
        headers=auth_headers,
        json={
            "provider": "ccxt",
            "mode": "paper",
            "credentials": {
                "exchange_id": "okx",
                "api_key": "pytest-key-only",
            },
            "metadata": {"source": "pytest-live", "case": "validation-failed-no-persist"},
        },
    )
    assert response.status_code == 422, response.text
    detail = response.json().get("detail", {})
    assert detail.get("code") == "BROKER_ACCOUNT_VALIDATION_FAILED"
    assert detail.get("validation_status") == "credentials_missing"

    after = api_test_client.get("/api/v1/broker-accounts", headers=auth_headers)
    assert after.status_code == 200, after.text
    after_ids = {
        str(item["broker_account_id"])
        for item in after.json()
        if isinstance(item, dict) and item.get("broker_account_id")
    }
    assert after_ids == before_ids
