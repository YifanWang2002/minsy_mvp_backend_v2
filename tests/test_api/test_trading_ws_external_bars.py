"""External integration test for WS bars snapshot with real Alpaca calls."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from apps.api.main import create_app
from packages.shared_settings.schema.settings import settings


class _DummyDbSession:
    async def rollback(self) -> None:
        return None


async def _fake_db_session() -> AsyncIterator[_DummyDbSession]:
    yield _DummyDbSession()


def _require_alpaca_credentials() -> None:
    if not settings.alpaca_api_key.strip() or not settings.alpaca_api_secret.strip():
        pytest.fail(
            "missing ALPACA_API_KEY/ALPACA_API_SECRET for external test "
            "tests/test_api/test_trading_ws_external_bars.py"
        )


def _make_app():
    with patch("apps.api.main.lifespan") as mock_lifespan:
        @asynccontextmanager
        async def _noop_lifespan(_):
            yield

        mock_lifespan.side_effect = _noop_lifespan
        return create_app()


@pytest.mark.external
def test_ws_subscribe_bars_external_alpaca_snapshot() -> None:
    _require_alpaca_credentials()
    app = _make_app()
    client = TestClient(app)

    attempts = 3
    for attempt in range(attempts):
        try:
            with patch(
                "apps.api.routes.trading_ws._authenticate_ws",
                new_callable=AsyncMock,
                return_value=SimpleNamespace(id="external-user"),
            ), patch(
                "apps.api.routes.trading_ws._load_owned_deployment",
                new_callable=AsyncMock,
                return_value=SimpleNamespace(id="dep-ext"),
            ), patch(
                "apps.api.routes.trading_ws._resolve_cursor",
                new_callable=AsyncMock,
                return_value=0,
            ), patch(
                "apps.api.routes.trading_ws._poll_outbox_events",
                new_callable=AsyncMock,
                return_value=[],
            ), patch(
                "apps.api.routes.trading_ws.get_db_session",
                new=_fake_db_session,
            ), patch(
                "apps.api.routes.trading_ws.enqueue_market_data_refresh",
                return_value=None,
            ):
                with client.websocket_connect("/api/v1/ws/trading?token=external-valid") as ws:
                    ws.send_text(
                        json.dumps(
                            {
                                "action": "subscribe",
                                "deployment_ids": ["dep-ext"],
                                "cursors": {"dep-ext": 0},
                            }
                        )
                    )
                    subscribed = json.loads(ws.receive_text())
                    assert subscribed["event"] == "subscribed"
                    assert subscribed["deployment_ids"] == ["dep-ext"]
                    assert subscribed["transport_mode"] in {"pubsub", "polling_fallback"}

                    ws.send_text(
                        json.dumps(
                            {
                                "action": "subscribe_bars",
                                "market": "crypto",
                                "symbol": "BTCUSD",
                                "timeframe": "1m",
                                "limit": 20,
                            }
                        )
                    )
                    seen_events: dict[str, dict] = {}
                    deadline = time.monotonic() + 20.0
                    while time.monotonic() < deadline and (
                        "bar_snapshot" not in seen_events
                        or "bars_subscribed" not in seen_events
                    ):
                        message = json.loads(ws.receive_text())
                        event_name = message.get("event")
                        if isinstance(event_name, str):
                            seen_events[event_name] = message

                    assert "bar_snapshot" in seen_events
                    assert "bars_subscribed" in seen_events
                    snapshot = seen_events["bar_snapshot"]
                    assert snapshot["market"] == "crypto"
                    assert snapshot["symbol"] == "BTCUSD"
                    assert snapshot["timeframe"] == "1m"
                    assert isinstance(snapshot.get("bars"), list)
            return
        except Exception:
            if attempt >= attempts - 1:
                raise
            time.sleep(2**attempt)
