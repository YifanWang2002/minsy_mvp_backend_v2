from __future__ import annotations

from fastapi.testclient import TestClient


def test_000_accessibility_status_endpoint(api_test_client: TestClient) -> None:
    response = api_test_client.get("/api/v1/status")
    assert response.status_code == 200, response.text


def test_010_status_payload_contract(api_test_client: TestClient) -> None:
    response = api_test_client.get("/api/v1/status")
    assert response.status_code == 200, response.text
    payload = response.json()

    assert payload["status"] in {"healthy", "degraded", "down"}
    services = payload.get("services")
    assert isinstance(services, dict)
    for key in ("api_endpoint", "chat", "backtest", "flower", "live_trading"):
        assert key in services
        service_payload = services[key]
        assert service_payload["status"] in {"healthy", "degraded", "down"}
        assert isinstance(service_payload.get("details"), str)

    system = payload.get("system")
    assert isinstance(system, dict)
    assert "cpu_usage_percent" in system
    assert "memory_usage_percent" in system
    assert "memory_available_gb" in system
    assert isinstance(payload.get("response_time_ms"), (int, float))
    assert isinstance(payload.get("timestamp"), str)
