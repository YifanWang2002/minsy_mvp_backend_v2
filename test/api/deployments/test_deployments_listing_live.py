from __future__ import annotations

from uuid import uuid4

from fastapi.testclient import TestClient


def test_000_accessibility_deployments_list(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    response = api_test_client.get("/api/v1/deployments", headers=auth_headers)
    assert response.status_code == 200, response.text
    assert isinstance(response.json(), list)


def test_010_deployment_detail_or_not_found(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    listing = api_test_client.get("/api/v1/deployments", headers=auth_headers)
    assert listing.status_code == 200, listing.text
    rows = listing.json()

    if rows:
        deployment_id = rows[0]["deployment_id"]
        detail = api_test_client.get(
            f"/api/v1/deployments/{deployment_id}",
            headers=auth_headers,
        )
        assert detail.status_code == 200, detail.text
        payload = detail.json()
        assert str(payload["deployment_id"]) == str(deployment_id)
        assert payload["mode"] in {"paper", "live"}
        return

    missing = api_test_client.get(
        f"/api/v1/deployments/{uuid4()}",
        headers=auth_headers,
    )
    assert missing.status_code == 404
