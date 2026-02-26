from __future__ import annotations

from uuid import uuid4

from fastapi.testclient import TestClient


def test_000_accessibility_trade_approvals_requires_auth(
    api_test_client: TestClient,
) -> None:
    response = api_test_client.get("/api/v1/trade-approvals")
    assert response.status_code == 401


def test_010_trade_approvals_invalid_status_filter_rejected(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    response = api_test_client.get(
        "/api/v1/trade-approvals",
        headers=auth_headers,
        params={"status": "unknown_status"},
    )
    assert response.status_code == 422, response.text


def test_020_trade_approval_approve_and_reject_missing_request_404(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    missing_id = uuid4()
    approve = api_test_client.post(
        f"/api/v1/trade-approvals/{missing_id}/approve",
        headers=auth_headers,
        json={"note": "pytest-live approve"},
    )
    assert approve.status_code == 404, approve.text

    reject = api_test_client.post(
        f"/api/v1/trade-approvals/{missing_id}/reject",
        headers=auth_headers,
        json={"note": "pytest-live reject"},
    )
    assert reject.status_code == 404, reject.text
