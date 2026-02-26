from __future__ import annotations

from uuid import uuid4

from fastapi.testclient import TestClient


def test_000_accessibility_strategy_versions_or_missing(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    list_response = api_test_client.get("/api/v1/strategies", headers=auth_headers)
    assert list_response.status_code == 200, list_response.text
    rows = list_response.json()

    if rows:
        strategy_id = rows[0]["strategy_id"]
        versions = api_test_client.get(
            f"/api/v1/strategies/{strategy_id}/versions",
            headers=auth_headers,
            params={"limit": 20},
        )
        assert versions.status_code == 200, versions.text
        assert isinstance(versions.json(), list)
        return

    missing = api_test_client.get(
        f"/api/v1/strategies/{uuid4()}/versions",
        headers=auth_headers,
        params={"limit": 20},
    )
    assert missing.status_code == 404


def test_010_strategy_diff_missing_strategy_404(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    response = api_test_client.get(
        f"/api/v1/strategies/{uuid4()}/diff",
        headers=auth_headers,
        params={"from_version": 1, "to_version": 2},
    )
    assert response.status_code == 404
