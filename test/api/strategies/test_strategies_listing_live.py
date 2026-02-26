from __future__ import annotations

from uuid import uuid4

from fastapi.testclient import TestClient


def test_000_accessibility_strategies_list(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    response = api_test_client.get("/api/v1/strategies", headers=auth_headers)
    assert response.status_code == 200, response.text
    assert isinstance(response.json(), list)


def test_010_strategy_detail_or_not_found(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    listing = api_test_client.get("/api/v1/strategies", headers=auth_headers)
    assert listing.status_code == 200, listing.text
    rows = listing.json()

    if rows:
        strategy_id = rows[0]["strategy_id"]
        detail = api_test_client.get(
            f"/api/v1/strategies/{strategy_id}",
            headers=auth_headers,
        )
        assert detail.status_code == 200, detail.text
        payload = detail.json()
        assert str(payload["strategy_id"]) == str(strategy_id)
        assert isinstance(payload.get("dsl_json"), dict)
        return

    missing = api_test_client.get(
        f"/api/v1/strategies/{uuid4()}",
        headers=auth_headers,
    )
    assert missing.status_code == 404
