from __future__ import annotations

from fastapi.testclient import TestClient


def test_000_accessibility_market_data_checkpoints(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    response = api_test_client.get(
        "/api/v1/market-data/checkpoints",
        headers=auth_headers,
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert "updated_at" in payload
    assert "checkpoints" in payload


def test_010_market_data_bars_shape(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    response = api_test_client.get(
        "/api/v1/market-data/bars",
        headers=auth_headers,
        params={
            "symbol": "SPY",
            "market": "stocks",
            "timeframe": "1m",
            "limit": 20,
            "refresh_if_empty": "true",
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["symbol"] == "SPY"
    assert payload["market"] == "stocks"
    assert payload["timeframe"] == "1m"
    assert isinstance(payload.get("bars"), list)


def test_020_market_data_quote_for_crypto(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    response = api_test_client.get(
        "/api/v1/market-data/quote",
        headers=auth_headers,
        params={"symbol": "BTCUSD", "market": "crypto", "refresh_if_missing": "true"},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["symbol"] == "BTCUSD"
    assert payload["market"] == "crypto"
