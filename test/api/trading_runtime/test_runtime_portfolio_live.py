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


def test_000_accessibility_runtime_portfolio_and_fills(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    deployment_id = _first_deployment_id(api_test_client, auth_headers)
    if deployment_id is None:
        deployment_id = str(uuid4())

    portfolio = api_test_client.get(
        f"/api/v1/deployments/{deployment_id}/portfolio",
        headers=auth_headers,
    )
    fills = api_test_client.get(
        f"/api/v1/deployments/{deployment_id}/fills",
        headers=auth_headers,
    )

    if portfolio.status_code == 200:
        payload = portfolio.json()
        assert str(payload["deployment_id"]) == deployment_id
        assert isinstance(payload.get("positions"), list)
    else:
        assert portfolio.status_code == 404, portfolio.text

    if fills.status_code == 200:
        assert isinstance(fills.json(), list)
    else:
        assert fills.status_code == 404, fills.text


def test_010_runtime_portfolio_missing_deployment_404(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    response = api_test_client.get(
        f"/api/v1/deployments/{uuid4()}/portfolio",
        headers=auth_headers,
    )
    assert response.status_code == 404
