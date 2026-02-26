from __future__ import annotations

from uuid import uuid4

from fastapi.testclient import TestClient


def _create_broker_account(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
) -> dict[str, object]:
    tag = uuid4().hex[:10]
    response = api_test_client.post(
        "/api/v1/broker-accounts",
        headers=auth_headers,
        params={"validate": "false"},
        json={
            "provider": "alpaca",
            "mode": "paper",
            "credentials": {
                "api_key": f"pytest-key-{tag}",
                "api_secret": f"pytest-secret-{tag}",
            },
            "metadata": {
                "source": "pytest-live",
                "tag": tag,
            },
        },
    )
    assert response.status_code == 201, response.text
    payload = response.json()
    assert payload["broker_account_id"]
    return payload


def test_000_accessibility_create_broker_account_without_validation(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    payload = _create_broker_account(api_test_client, auth_headers)
    assert payload["provider"] == "alpaca"
    assert payload["mode"] == "paper"
    assert payload["last_validated_status"] == "validation_skipped"
    assert payload["status"] in {"active", "pending"}


def test_010_rotate_credentials_and_deactivate(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    created = _create_broker_account(api_test_client, auth_headers)
    broker_account_id = str(created["broker_account_id"])

    rotated = api_test_client.patch(
        f"/api/v1/broker-accounts/{broker_account_id}/credentials",
        headers=auth_headers,
        params={"validate": "false"},
        json={
            "credentials": {
                "api_key": f"pytest-rotated-key-{uuid4().hex[:8]}",
                "api_secret": f"pytest-rotated-secret-{uuid4().hex[:8]}",
            }
        },
    )
    assert rotated.status_code == 200, rotated.text
    rotated_payload = rotated.json()
    assert str(rotated_payload["broker_account_id"]) == broker_account_id
    assert rotated_payload["updated_source"] == "api"
    assert rotated_payload["last_validated_status"] == "validation_skipped"
    assert rotated_payload["key_fingerprint"]

    deactivated = api_test_client.post(
        f"/api/v1/broker-accounts/{broker_account_id}/deactivate",
        headers=auth_headers,
    )
    assert deactivated.status_code == 200, deactivated.text
    assert deactivated.json()["status"] == "inactive"
