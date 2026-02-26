from __future__ import annotations

from fastapi.testclient import TestClient


def test_000_accessibility_market_data_quote_from_real_provider(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    response = api_test_client.get(
        "/api/v1/market-data/quote",
        params={"symbol": "SPY", "market": "stocks", "refresh_if_missing": "true"},
        headers=auth_headers,
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["symbol"] == "SPY"
    assert payload["market"] == "stocks"
    assert any(payload.get(field) is not None for field in ("bid", "ask", "last"))


def test_010_market_data_subscriptions_and_health(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    subscribe = api_test_client.post(
        "/api/v1/market-data/subscriptions",
        params={"market": "stocks"},
        json={"symbols": ["SPY"]},
        headers=auth_headers,
    )
    assert subscribe.status_code == 200, subscribe.text

    subscriptions = api_test_client.get(
        "/api/v1/market-data/subscriptions",
        headers=auth_headers,
    )
    assert subscriptions.status_code == 200, subscriptions.text
    active = subscriptions.json().get("active_symbols", [])
    assert "SPY" in active

    health = api_test_client.get(
        "/api/v1/market-data/health",
        params={"window_minutes": 60, "max_events": 20},
        headers=auth_headers,
    )
    assert health.status_code == 200, health.text
    health_payload = health.json()
    assert "runtime_metrics" in health_payload
    assert "active_subscriptions" in health_payload

    cleanup = api_test_client.delete(
        "/api/v1/market-data/subscriptions",
        headers=auth_headers,
    )
    assert cleanup.status_code == 200, cleanup.text
