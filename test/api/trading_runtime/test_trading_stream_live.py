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


def test_000_accessibility_trading_stream_or_not_found(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    deployment_id = _first_deployment_id(api_test_client, auth_headers)
    if deployment_id is None:
        deployment_id = str(uuid4())

    response = api_test_client.get(
        f"/api/v1/stream/deployments/{deployment_id}",
        headers=auth_headers,
        params={"max_events": 1, "poll_seconds": 0.2, "heartbeat_seconds": 0.2},
    )

    if response.status_code == 200:
        text = response.text
        assert "event:" in text
        assert "data:" in text
    else:
        assert response.status_code == 404, response.text


def test_010_trading_stream_missing_deployment_404(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    response = api_test_client.get(
        f"/api/v1/stream/deployments/{uuid4()}",
        headers=auth_headers,
        params={"max_events": 1, "poll_seconds": 0.2, "heartbeat_seconds": 0.2},
    )
    assert response.status_code == 404
