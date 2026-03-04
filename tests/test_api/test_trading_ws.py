"""Tests for the trading WebSocket endpoint."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from apps.api.main import create_app


def _make_fake_user(user_id: str = "user-1") -> MagicMock:
    user = MagicMock()
    user.id = user_id
    return user


@pytest.fixture()
def app():
    """Create a fresh app instance for testing."""
    with patch("apps.api.main.lifespan") as mock_lifespan:
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def _noop_lifespan(_):
            yield

        mock_lifespan.side_effect = _noop_lifespan
        test_app = create_app()
        return test_app


@pytest.fixture()
def client(app):
    return TestClient(app)


class TestTradingWebSocket:
    """Tests for /api/v1/ws/trading WebSocket endpoint."""

    def test_rejects_missing_token(self, client):
        """Connection without token should be rejected."""
        with pytest.raises(Exception):
            with client.websocket_connect("/api/v1/ws/trading"):
                pass

    def test_rejects_invalid_token(self, client):
        """Connection with invalid token should be rejected."""
        with patch(
            "apps.api.routes.trading_ws._authenticate_ws",
            new_callable=AsyncMock,
            return_value=None,
        ):
            with pytest.raises(Exception):
                with client.websocket_connect("/api/v1/ws/trading?token=bad"):
                    pass

    def test_accepts_valid_token_and_responds_to_ping(self, client):
        """Valid token should allow connection; ping should get pong."""
        fake_user = _make_fake_user()
        with patch(
            "apps.api.routes.trading_ws._authenticate_ws",
            new_callable=AsyncMock,
            return_value=fake_user,
        ):
            with client.websocket_connect("/api/v1/ws/trading?token=valid") as ws:
                ws.send_text(json.dumps({"action": "ping"}))
                resp = json.loads(ws.receive_text())
                assert resp["event"] == "pong"
                assert "server_time" in resp

    def test_subscribe_to_nonexistent_deployment(self, client):
        """Subscribing to a deployment that doesn't exist should return error."""
        fake_user = _make_fake_user()
        with patch(
            "apps.api.routes.trading_ws._authenticate_ws",
            new_callable=AsyncMock,
            return_value=fake_user,
        ), patch(
            "apps.api.routes.trading_ws._load_owned_deployment",
            new_callable=AsyncMock,
            side_effect=Exception("not found"),
        ), patch(
            "apps.api.routes.trading_ws.get_db_session",
        ) as mock_get_db:
            mock_session = AsyncMock()
            mock_session.__aiter__ = AsyncMock(return_value=iter([mock_session]))

            async def _fake_db_session():
                yield mock_session

            mock_get_db.return_value = _fake_db_session()

            with client.websocket_connect("/api/v1/ws/trading?token=valid") as ws:
                ws.send_text(json.dumps({
                    "action": "subscribe",
                    "deployment_ids": ["nonexistent"],
                }))
                resp = json.loads(ws.receive_text())
                assert resp["event"] == "error"
                assert "not found" in resp["message"]

    def test_invalid_json_returns_error(self, client):
        """Sending invalid JSON should return an error event."""
        fake_user = _make_fake_user()
        with patch(
            "apps.api.routes.trading_ws._authenticate_ws",
            new_callable=AsyncMock,
            return_value=fake_user,
        ):
            with client.websocket_connect("/api/v1/ws/trading?token=valid") as ws:
                ws.send_text("not json at all")
                resp = json.loads(ws.receive_text())
                assert resp["event"] == "error"
                assert "invalid json" in resp["message"]

    def test_unknown_action_returns_error(self, client):
        """Unknown action should return an error event."""
        fake_user = _make_fake_user()
        with patch(
            "apps.api.routes.trading_ws._authenticate_ws",
            new_callable=AsyncMock,
            return_value=fake_user,
        ):
            with client.websocket_connect("/api/v1/ws/trading?token=valid") as ws:
                ws.send_text(json.dumps({"action": "foobar"}))
                resp = json.loads(ws.receive_text())
                assert resp["event"] == "error"
                assert "unknown action" in resp["message"]

    def test_subscribe_bars_missing_symbol_returns_error(self, client):
        """subscribe_bars without symbol should return an error."""
        fake_user = _make_fake_user()
        with patch(
            "apps.api.routes.trading_ws._authenticate_ws",
            new_callable=AsyncMock,
            return_value=fake_user,
        ):
            with client.websocket_connect("/api/v1/ws/trading?token=valid") as ws:
                ws.send_text(json.dumps({
                    "action": "subscribe_bars",
                    "market": "stocks",
                    "timeframe": "1m",
                }))
                resp = json.loads(ws.receive_text())
                assert resp["event"] == "error"
                assert "symbol" in resp["message"].lower()

    def test_subscribe_bars_returns_snapshot_and_confirmation(self, client):
        """subscribe_bars with valid params should return bar_snapshot + bars_subscribed."""
        from datetime import datetime, UTC

        fake_user = _make_fake_user()
        fake_bar = MagicMock()
        fake_bar.timestamp = datetime(2026, 1, 1, tzinfo=UTC)
        fake_bar.open = 100.0
        fake_bar.high = 105.0
        fake_bar.low = 99.0
        fake_bar.close = 103.0
        fake_bar.volume = 1000.0

        with patch(
            "apps.api.routes.trading_ws._authenticate_ws",
            new_callable=AsyncMock,
            return_value=fake_user,
        ), patch(
            "apps.api.routes.trading_ws.market_data_runtime",
        ) as mock_runtime:
            mock_runtime.get_recent_bars.return_value = [fake_bar]

            with client.websocket_connect("/api/v1/ws/trading?token=valid") as ws:
                ws.send_text(json.dumps({
                    "action": "subscribe_bars",
                    "market": "stocks",
                    "symbol": "AAPL",
                    "timeframe": "1m",
                    "limit": 10,
                }))
                # First message: bar_snapshot
                resp1 = json.loads(ws.receive_text())
                assert resp1["event"] == "bar_snapshot"
                assert resp1["market"] == "stocks"
                assert resp1["symbol"] == "AAPL"
                assert resp1["timeframe"] == "1m"
                assert len(resp1["bars"]) == 1
                bar = resp1["bars"][0]
                assert bar["o"] == 100.0
                assert bar["h"] == 105.0
                assert bar["l"] == 99.0
                assert bar["c"] == 103.0
                assert bar["v"] == 1000.0

                # Second message: bars_subscribed
                resp2 = json.loads(ws.receive_text())
                assert resp2["event"] == "bars_subscribed"
                assert resp2["symbol"] == "AAPL"

    def test_unsubscribe_bars_returns_confirmation(self, client):
        """unsubscribe_bars should return bars_unsubscribed."""
        fake_user = _make_fake_user()
        with patch(
            "apps.api.routes.trading_ws._authenticate_ws",
            new_callable=AsyncMock,
            return_value=fake_user,
        ):
            with client.websocket_connect("/api/v1/ws/trading?token=valid") as ws:
                ws.send_text(json.dumps({
                    "action": "unsubscribe_bars",
                    "market": "stocks",
                    "symbol": "AAPL",
                    "timeframe": "1m",
                }))
                resp = json.loads(ws.receive_text())
                assert resp["event"] == "bars_unsubscribed"
                assert resp["market"] == "stocks"
                assert resp["symbol"] == "AAPL"
                assert resp["timeframe"] == "1m"
