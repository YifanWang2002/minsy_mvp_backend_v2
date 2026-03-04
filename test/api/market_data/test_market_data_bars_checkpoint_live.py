from __future__ import annotations

from datetime import UTC, datetime

from fastapi.testclient import TestClient

from apps.api.routes import market_data as market_data_route
from packages.domain.market_data.runtime import RuntimeBar
from packages.infra.providers.trading.adapters.base import QuoteSnapshot


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


async def test_030_build_live_bar_aggregates_current_bucket(monkeypatch) -> None:
    async def _fake_resolve_live_quote(*, market: str, symbol: str) -> QuoteSnapshot | None:
        assert market == "stocks"
        assert symbol == "SPY"
        return QuoteSnapshot(
            symbol="SPY",
            bid=None,
            ask=None,
            last=105.0,
            timestamp=datetime(2026, 3, 3, 12, 3, 30, tzinfo=UTC),
            raw={},
        )

    def _fake_get_recent_bars(*, market: str, symbol: str, timeframe: str, limit: int = 200):
        assert market == "stocks"
        assert symbol == "SPY"
        if timeframe == "1m":
            return [
                RuntimeBar(
                    timestamp=datetime(2026, 3, 3, 12, 0, tzinfo=UTC),
                    open=100.0,
                    high=101.0,
                    low=99.5,
                    close=100.5,
                    volume=10.0,
                ),
                RuntimeBar(
                    timestamp=datetime(2026, 3, 3, 12, 1, tzinfo=UTC),
                    open=100.5,
                    high=102.0,
                    low=100.0,
                    close=101.5,
                    volume=12.0,
                ),
            ]
        raise AssertionError(f"Unexpected timeframe: {timeframe}")

    monkeypatch.setattr(market_data_route, "_resolve_live_quote", _fake_resolve_live_quote)
    monkeypatch.setattr(
        market_data_route.market_data_runtime,
        "get_recent_bars",
        _fake_get_recent_bars,
    )

    live_bar = await market_data_route._build_live_bar(
        market="stocks",
        symbol="SPY",
        timeframe="5m",
        historical_bars=[
            RuntimeBar(
                timestamp=datetime(2026, 3, 3, 11, 55, tzinfo=UTC),
                open=98.0,
                high=99.0,
                low=97.5,
                close=99.0,
                volume=50.0,
            )
        ],
    )

    assert live_bar is not None
    assert live_bar.timestamp == datetime(2026, 3, 3, 12, 0, tzinfo=UTC)
    assert live_bar.open == 100.0
    assert live_bar.high == 105.0
    assert live_bar.low == 99.5
    assert live_bar.close == 105.0
    assert live_bar.volume == 22.0
