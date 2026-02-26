from __future__ import annotations

from uuid import uuid4

from fastapi.testclient import TestClient


def _first_deployment_id(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
) -> str | None:
    listing = api_test_client.get("/api/v1/deployments", headers=auth_headers)
    assert listing.status_code == 200, listing.text
    rows = listing.json()
    if not rows:
        return None
    return str(rows[0]["deployment_id"])


def test_000_accessibility_deployment_runtime_views(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    deployment_id = _first_deployment_id(api_test_client, auth_headers)
    if deployment_id is None:
        deployment_id = str(uuid4())

    for suffix in ("orders", "positions", "pnl", "signals"):
        response = api_test_client.get(
            f"/api/v1/deployments/{deployment_id}/{suffix}",
            headers=auth_headers,
        )
        if deployment_id and response.status_code == 200:
            assert isinstance(response.json(), list)
        else:
            assert response.status_code == 404, response.text


def test_010_process_now_for_missing_deployment_404(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    response = api_test_client.post(
        f"/api/v1/deployments/{uuid4()}/process-now",
        headers=auth_headers,
    )
    assert response.status_code == 404
