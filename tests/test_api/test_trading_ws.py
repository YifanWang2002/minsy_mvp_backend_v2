"""Tests for the trading WebSocket endpoint."""

from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

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
        from datetime import UTC, datetime

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
            with patch(
                "apps.api.routes.trading_ws._load_bar_snapshot",
                new_callable=AsyncMock,
                return_value=[fake_bar],
            ):

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

    def test_forwards_manual_action_update_event(self, client):
        """Subscribed WS client should receive manual_action_update events."""
        fake_user = _make_fake_user()
        fake_session = AsyncMock()
        poll_calls = {"count": 0}

        async def _fake_poll_outbox_events(_db, *, deployment_id, cursor, limit=50):
            del _db, limit
            if deployment_id != "dep-1":
                return []
            if cursor > 0:
                return []
            if poll_calls["count"] > 0:
                return []
            poll_calls["count"] += 1
            return [
                SimpleNamespace(
                    event_seq=101,
                    event_type="manual_action_update",
                    payload={
                        "deployment_id": "dep-1",
                        "manual_actions": [
                            {
                                "manual_trade_action_id": "ma-1",
                                "status": "executing",
                            }
                        ],
                        "latest_manual_action_id": "ma-1",
                    },
                )
            ]

        async def _fake_load_owned_deployment(*_args, **_kwargs):
            return SimpleNamespace(id="dep-1")

        async def _fake_get_db_session():
            yield fake_session

        with patch(
            "apps.api.routes.trading_ws._authenticate_ws",
            new_callable=AsyncMock,
            return_value=fake_user,
        ), patch(
            "apps.api.routes.trading_ws._load_owned_deployment",
            new_callable=AsyncMock,
            side_effect=_fake_load_owned_deployment,
        ), patch(
            "apps.api.routes.trading_ws.get_db_session",
            side_effect=_fake_get_db_session,
        ), patch(
            "apps.api.routes.trading_ws._poll_outbox_events",
            new_callable=AsyncMock,
            side_effect=_fake_poll_outbox_events,
        ):
            with client.websocket_connect("/api/v1/ws/trading?token=valid") as ws:
                ws.send_text(
                    json.dumps(
                        {
                            "action": "subscribe",
                            "deployment_ids": ["dep-1"],
                            "cursors": {"dep-1": 0},
                        }
                    )
                )

                first = json.loads(ws.receive_text())
                assert first["event"] == "subscribed"
                assert first["deployment_ids"] == ["dep-1"]

                forwarded = json.loads(ws.receive_text())
                assert forwarded["event"] == "manual_action_update"
                assert forwarded["deployment_id"] == "dep-1"
                assert forwarded["event_seq"] == 101
                assert forwarded["payload"]["latest_manual_action_id"] == "ma-1"

    def test_pubsub_path_pushes_events_without_polling_loop(self, client):
        """When Redis pub/sub is available, WS should push realtime events without poll-loop."""

        class _FakePubSub:
            def __init__(self) -> None:
                self._subscribed: set[str] = set()
                self._emitted = False

            async def subscribe(self, *channels: str) -> None:
                for channel in channels:
                    self._subscribed.add(str(channel))

            async def unsubscribe(self, *channels: str) -> None:
                for channel in channels:
                    self._subscribed.discard(str(channel))

            async def get_message(self, *, ignore_subscribe_messages: bool, timeout: float):
                del ignore_subscribe_messages, timeout
                if self._subscribed and not self._emitted:
                    self._emitted = True
                    return {
                        "data": json.dumps(
                            {
                                "deployment_id": "dep-1",
                                "event": "manual_action_update",
                                "event_seq": 201,
                                "payload": {
                                    "deployment_id": "dep-1",
                                    "manual_actions": [
                                        {"manual_trade_action_id": "ma-201", "status": "executing"}
                                    ],
                                    "latest_manual_action_id": "ma-201",
                                },
                            }
                        )
                    }
                await asyncio.sleep(0.01)
                return None

            async def aclose(self) -> None:
                return None

        class _FakeRedis:
            def __init__(self, pubsub: _FakePubSub) -> None:
                self._pubsub = pubsub

            def pubsub(self, *, ignore_subscribe_messages: bool):
                del ignore_subscribe_messages
                return self._pubsub

        fake_user = _make_fake_user()
        fake_session = AsyncMock()
        fake_pubsub = _FakePubSub()
        poll_mock = AsyncMock(return_value=[])

        async def _fake_load_owned_deployment(*_args, **_kwargs):
            return SimpleNamespace(id="dep-1")

        async def _fake_get_db_session():
            yield fake_session

        with patch(
            "apps.api.routes.trading_ws._authenticate_ws",
            new_callable=AsyncMock,
            return_value=fake_user,
        ), patch(
            "apps.api.routes.trading_ws.get_redis_client",
            return_value=_FakeRedis(fake_pubsub),
        ), patch(
            "apps.api.routes.trading_ws._load_owned_deployment",
            new_callable=AsyncMock,
            side_effect=_fake_load_owned_deployment,
        ), patch(
            "apps.api.routes.trading_ws._resolve_cursor",
            new_callable=AsyncMock,
            return_value=200,
        ), patch(
            "apps.api.routes.trading_ws.get_db_session",
            side_effect=_fake_get_db_session,
        ), patch(
            "apps.api.routes.trading_ws._poll_outbox_events",
            poll_mock,
        ):
            with client.websocket_connect("/api/v1/ws/trading?token=valid") as ws:
                ws.send_text(
                    json.dumps(
                        {
                            "action": "subscribe",
                            "deployment_ids": ["dep-1"],
                            "cursors": {"dep-1": 200},
                        }
                    )
                )
                first = json.loads(ws.receive_text())
                assert first["event"] == "subscribed"
                assert first["cursors"]["dep-1"] == 200

                forwarded = json.loads(ws.receive_text())
                assert forwarded["event"] == "manual_action_update"
                assert forwarded["deployment_id"] == "dep-1"
                assert forwarded["event_seq"] == 201
                assert forwarded["payload"]["latest_manual_action_id"] == "ma-201"

                # Replay is queried once on subscribe; no continuous polling loop in pub/sub mode.
                assert poll_mock.await_count == 1

    def test_pubsub_read_failure_degrades_and_recovers_without_ws_disconnect(self, client):
        """A pubsub read failure should switch to fallback, then recover to pubsub."""

        class _FakePubSub:
            def __init__(self) -> None:
                self._subscribed: set[str] = set()
                self._calls = 0

            async def subscribe(self, *channels: str) -> None:
                for channel in channels:
                    self._subscribed.add(str(channel))

            async def unsubscribe(self, *channels: str) -> None:
                for channel in channels:
                    self._subscribed.discard(str(channel))

            async def get_message(self, *, ignore_subscribe_messages: bool, timeout: float):
                del ignore_subscribe_messages, timeout
                self._calls += 1
                if self._calls == 1:
                    raise RuntimeError("simulated pubsub read failure")
                if self._calls == 2 and self._subscribed:
                    return {
                        "data": json.dumps(
                            {
                                "deployment_id": "dep-1",
                                "event": "manual_action_update",
                                "event_seq": 301,
                                "payload": {
                                    "deployment_id": "dep-1",
                                    "manual_actions": [
                                        {"manual_trade_action_id": "ma-301", "status": "executing"}
                                    ],
                                    "latest_manual_action_id": "ma-301",
                                },
                            }
                        )
                    }
                await asyncio.sleep(0.01)
                return None

            async def aclose(self) -> None:
                return None

        class _FakeRedis:
            def __init__(self, pubsub: _FakePubSub) -> None:
                self._pubsub = pubsub

            def pubsub(self, *, ignore_subscribe_messages: bool):
                del ignore_subscribe_messages
                return self._pubsub

        fake_user = _make_fake_user()
        fake_pubsub = _FakePubSub()

        async def _fake_load_owned_deployment(*_args, **_kwargs):
            return SimpleNamespace(id="dep-1")

        async def _fake_get_db_session():
            session = AsyncMock()
            yield session

        with patch(
            "apps.api.routes.trading_ws._authenticate_ws",
            new_callable=AsyncMock,
            return_value=fake_user,
        ), patch(
            "apps.api.routes.trading_ws.get_redis_client",
            return_value=_FakeRedis(fake_pubsub),
        ), patch(
            "apps.api.routes.trading_ws._load_owned_deployment",
            new_callable=AsyncMock,
            side_effect=_fake_load_owned_deployment,
        ), patch(
            "apps.api.routes.trading_ws._resolve_cursor",
            new_callable=AsyncMock,
            return_value=300,
        ), patch(
            "apps.api.routes.trading_ws.get_db_session",
            side_effect=_fake_get_db_session,
        ), patch(
            "apps.api.routes.trading_ws._poll_outbox_events",
            new_callable=AsyncMock,
            return_value=[],
        ), patch(
            "apps.api.routes.trading_ws._PUBSUB_WAIT_SECONDS",
            0.01,
        ), patch(
            "apps.api.routes.trading_ws._FALLBACK_POLL_SECONDS",
            0.05,
        ), patch(
            "apps.api.routes.trading_ws._RECONCILE_SECONDS",
            0.2,
        ), patch(
            "apps.api.routes.trading_ws._PUBSUB_PROBE_BASE_SECONDS",
            0.05,
        ), patch(
            "apps.api.routes.trading_ws._PUBSUB_PROBE_MAX_SECONDS",
            0.2,
        ):
            with client.websocket_connect("/api/v1/ws/trading?token=valid") as ws:
                ws.send_text(
                    json.dumps(
                        {
                            "action": "subscribe",
                            "deployment_ids": ["dep-1"],
                            "cursors": {"dep-1": 300},
                        }
                    )
                )
                transport_modes: list[str] = []
                saw_subscribed = False
                for _ in range(20):
                    ws.send_text(json.dumps({"action": "ping"}))
                    msg = json.loads(ws.receive_text())
                    if msg.get("event") == "subscribed":
                        saw_subscribed = True
                    if msg.get("event") == "transport_mode":
                        transport_modes.append(str(msg.get("mode")))
                    if saw_subscribed and "polling_fallback" in transport_modes and "pubsub" in transport_modes:
                        break
                    time.sleep(0.03)

                assert saw_subscribed is True
                assert "polling_fallback" in transport_modes
                assert "pubsub" in transport_modes

    def test_bar_updates_dedupe_identical_pubsub_payload(self, client):
        """Repeated identical bar_update payload should only be forwarded once."""

        from datetime import UTC, datetime

        class _FakePubSub:
            def __init__(self) -> None:
                self._subscribed: set[str] = set()
                self._emitted = 0

            async def subscribe(self, *channels: str) -> None:
                for channel in channels:
                    self._subscribed.add(str(channel))

            async def unsubscribe(self, *channels: str) -> None:
                for channel in channels:
                    self._subscribed.discard(str(channel))

            async def get_message(self, *, ignore_subscribe_messages: bool, timeout: float):
                del ignore_subscribe_messages, timeout
                if not self._subscribed:
                    await asyncio.sleep(0.01)
                    return None
                if self._emitted < 2:
                    self._emitted += 1
                    payload = {
                        "event": "bar_update",
                        "market": "crypto",
                        "symbol": "BTCUSD",
                        "timeframe": "1m",
                        "bars": [
                            {
                                "t": 1772971200,
                                "o": 67525.7815,
                                "h": 67649.593,
                                "l": 67484.255,
                                "c": 67529.0005,
                                "v": 0.00021,
                            }
                        ],
                    }
                    return {"data": json.dumps(payload)}
                await asyncio.sleep(0.01)
                return None

            async def aclose(self) -> None:
                return None

        class _FakeRedis:
            def __init__(self, pubsub: _FakePubSub) -> None:
                self._pubsub = pubsub

            def pubsub(self, *, ignore_subscribe_messages: bool):
                del ignore_subscribe_messages
                return self._pubsub

        fake_user = _make_fake_user()
        fake_pubsub = _FakePubSub()
        fake_bar = SimpleNamespace(
            timestamp=datetime(2026, 1, 1, tzinfo=UTC),
            open=100.0,
            high=105.0,
            low=99.0,
            close=103.0,
            volume=1000.0,
        )

        with patch(
            "apps.api.routes.trading_ws._authenticate_ws",
            new_callable=AsyncMock,
            return_value=fake_user,
        ), patch(
            "apps.api.routes.trading_ws.get_redis_client",
            return_value=_FakeRedis(fake_pubsub),
        ), patch(
            "apps.api.routes.trading_ws._load_bar_snapshot",
            new_callable=AsyncMock,
            return_value=[fake_bar],
        ), patch(
            "apps.api.routes.trading_ws._PUBSUB_WAIT_SECONDS",
            0.01,
        ):
            with client.websocket_connect("/api/v1/ws/trading?token=valid") as ws:
                ws.send_text(
                    json.dumps(
                        {
                            "action": "subscribe_bars",
                            "market": "crypto",
                            "symbol": "BTCUSD",
                            "timeframe": "1m",
                            "limit": 10,
                        }
                    )
                )
                snapshot = json.loads(ws.receive_text())
                assert snapshot["event"] == "bar_snapshot"
                confirm = json.loads(ws.receive_text())
                assert confirm["event"] == "bars_subscribed"

                bar_update_count = 0
                for _ in range(8):
                    ws.send_text(json.dumps({"action": "ping"}))
                    msg = json.loads(ws.receive_text())
                    if msg.get("event") == "bar_update":
                        bar_update_count += 1
                    if bar_update_count >= 1:
                        # Consume one extra cycle to give the duplicate a chance to appear.
                        ws.send_text(json.dumps({"action": "ping"}))
                        extra = json.loads(ws.receive_text())
                        if extra.get("event") == "bar_update":
                            bar_update_count += 1
                        break

                assert bar_update_count == 1
