from __future__ import annotations

from fastapi.testclient import TestClient


def test_000_accessibility_invalid_market_rejected(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    response = api_test_client.get(
        "/api/v1/market-data/quote",
        headers=auth_headers,
        params={"symbol": "SPY", "market": "invalid_market_name"},
    )
    assert response.status_code == 422, response.text


def test_010_missing_symbol_validation_rejected(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    response = api_test_client.get(
        "/api/v1/market-data/quote",
        headers=auth_headers,
        params={"symbol": "", "market": "stocks"},
    )
    assert response.status_code in {422, 400}, response.text


def test_020_quote_not_found_path_returns_404(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    # Do not trigger upstream refresh here; validate runtime-cache not-found path deterministically.
    response = api_test_client.get(
        "/api/v1/market-data/quote",
        headers=auth_headers,
        params={
            "symbol": "THISSHOULDNOTEXIST123",
            "market": "stocks",
            "refresh_if_missing": "false",
        },
    )
    assert response.status_code == 404, response.text
    payload = response.json()
    assert "detail" in payload
