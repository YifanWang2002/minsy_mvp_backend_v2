from __future__ import annotations

from fastapi.testclient import TestClient


def test_000_accessibility_health_endpoint(api_test_client: TestClient) -> None:
    response = api_test_client.get("/api/v1/health")
    assert response.status_code == 200, response.text


def test_010_health_payload_contract(api_test_client: TestClient) -> None:
    response = api_test_client.get("/api/v1/health")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] in {"ok", "degraded"}
    assert isinstance(payload.get("db"), bool)
    assert isinstance(payload.get("redis"), bool)
    assert isinstance(payload.get("state_stores"), dict)
    services = payload.get("services")
    assert isinstance(services, dict)
    assert services.get("api") == "ok"
