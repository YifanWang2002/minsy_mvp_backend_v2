from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from src.engine.execution.adapters.base import OhlcvBar, QuoteSnapshot
from src.engine.market_data.runtime import market_data_runtime
from src.main import app


def _register_and_get_token(client: TestClient) -> str:
    email = f"market_data_{uuid4().hex}@test.com"
    response = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "pass1234", "name": "Market Data User"},
    )
    assert response.status_code == 201
    return response.json()["access_token"]


@pytest.mark.parametrize("market,symbol", [("stocks", "AAPL"), ("crypto", "BTCUSD")])
def test_market_data_quote_and_bars_endpoints(
    monkeypatch: pytest.MonkeyPatch,
    market: str,
    symbol: str,
) -> None:
    monkeypatch.setattr("src.api.routers.market_data.enqueue_market_data_refresh", lambda **_: "task-id")
    market_data_runtime.reset()

    quote = QuoteSnapshot(
        symbol=symbol,
        bid=Decimal("100"),
        ask=Decimal("101"),
        last=Decimal("100.5"),
        timestamp=datetime(2026, 1, 5, 10, 0, tzinfo=UTC),
    )
    market_data_runtime.upsert_quote(market=market, symbol=symbol, quote=quote)

    for minute in range(6):
        market_data_runtime.ingest_1m_bar(
            market=market,
            symbol=symbol,
            bar=OhlcvBar(
                timestamp=datetime(2026, 1, 5, 10, minute, tzinfo=UTC),
                open=Decimal(100 + minute),
                high=Decimal(101 + minute),
                low=Decimal(99 + minute),
                close=Decimal(100.5 + minute),
                volume=Decimal("10"),
            ),
        )

    with TestClient(app) as client:
        token = _register_and_get_token(client)
        headers = {"Authorization": f"Bearer {token}"}

        quote_resp = client.get(
            "/api/v1/market-data/quote",
            headers=headers,
            params={"market": market, "symbol": symbol},
        )
        assert quote_resp.status_code == 200
        quote_body = quote_resp.json()
        assert quote_body["symbol"] == symbol
        assert quote_body["bid"] == 100.0
        assert quote_body["ask"] == 101.0

        bars_resp = client.get(
            "/api/v1/market-data/bars",
            headers=headers,
            params={"market": market, "symbol": symbol, "timeframe": "1m", "limit": 3},
        )
        assert bars_resp.status_code == 200
        bars = bars_resp.json()["bars"]
        assert len(bars) == 3
        assert bars[-1]["close"] == 105.5

        agg_resp = client.get(
            "/api/v1/market-data/bars",
            headers=headers,
            params={"market": market, "symbol": symbol, "timeframe": "5m", "limit": 10},
        )
        assert agg_resp.status_code == 200
        agg_bars = agg_resp.json()["bars"]
        assert len(agg_bars) == 1
        assert agg_bars[0]["open"] == 100.0
        assert agg_bars[0]["close"] == 104.5


def test_market_data_subscription_endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    scheduled: list[dict[str, str]] = []

    def _capture_enqueue(**payload: str) -> str:
        scheduled.append(payload)
        return "task-id"

    monkeypatch.setattr("src.api.routers.market_data.enqueue_market_data_refresh", _capture_enqueue)
    market_data_runtime.reset()

    with TestClient(app) as client:
        token = _register_and_get_token(client)
        headers = {"Authorization": f"Bearer {token}"}

        subscribe = client.post(
            "/api/v1/market-data/subscriptions",
            headers=headers,
            params={"market": "crypto"},
            json={"symbols": ["btcusd", "BTCUSD", "ethusd"]},
        )
        assert subscribe.status_code == 200
        payload = subscribe.json()
        assert sorted(payload["added_symbols"]) == ["BTCUSD", "ETHUSD"]
        assert sorted(payload["active_symbols"]) == ["BTCUSD", "ETHUSD"]
        assert len(scheduled) == 2
        assert all(item["market"] == "crypto" for item in scheduled)
        assert sorted(item["symbol"] for item in scheduled) == ["BTCUSD", "ETHUSD"]

        get_state = client.get("/api/v1/market-data/subscriptions", headers=headers)
        assert get_state.status_code == 200
        assert sorted(get_state.json()["active_symbols"]) == ["BTCUSD", "ETHUSD"]

        unsubscribe = client.delete("/api/v1/market-data/subscriptions", headers=headers)
        assert unsubscribe.status_code == 200
        assert sorted(unsubscribe.json()["removed_symbols"]) == ["BTCUSD", "ETHUSD"]

        health = client.get("/api/v1/market-data/health", headers=headers)
        assert health.status_code == 200
        body = health.json()
        assert "error_summary" in body
        assert "recent_errors" in body
