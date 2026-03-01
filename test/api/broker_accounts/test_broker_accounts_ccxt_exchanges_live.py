from __future__ import annotations

from fastapi.testclient import TestClient


def test_000_accessibility_ccxt_exchanges_requires_auth(
    api_test_client: TestClient,
) -> None:
    response = api_test_client.get("/api/v1/broker-accounts/ccxt/exchanges")
    assert response.status_code == 401


def test_010_accessibility_ccxt_exchanges_contract(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    response = api_test_client.get(
        "/api/v1/broker-accounts/ccxt/exchanges",
        headers=auth_headers,
    )
    assert response.status_code == 200, response.text

    payload = response.json()
    assert isinstance(payload, list)
    assert payload

    exchange_ids = {str(item.get("exchange_id", "")).strip() for item in payload}
    assert "binance" in exchange_ids
    assert "okx" in exchange_ids

    first = payload[0]
    assert isinstance(first.get("required_fields"), list)
    assert isinstance(first.get("optional_fields"), list)
    assert isinstance(first.get("supports_paper"), bool)
    assert isinstance(first.get("supports_live"), bool)
    assert isinstance(first.get("paper_trading_status"), str)
    assert isinstance(first.get("paper_trading_message"), str)
    assert isinstance(first.get("live_trading_status"), str)
    assert isinstance(first.get("live_trading_message"), str)
    assert first.get("supports_live") is False
