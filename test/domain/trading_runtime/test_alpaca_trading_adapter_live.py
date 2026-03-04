from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from packages.infra.providers.trading.adapters.alpaca_trading import AlpacaTradingAdapter
from packages.infra.providers.trading.adapters.base import OrderIntent


class _MarketDataClientStub:
    async def aclose(self) -> None:
        return None


async def test_000_accessibility_alpaca_rounds_fractional_stock_orders_down(monkeypatch) -> None:
    captured: dict[str, object] = {}
    adapter = AlpacaTradingAdapter(
        api_key="key",
        api_secret="secret",
        trading_base_url="https://paper-api.alpaca.markets",
        market_data_client=_MarketDataClientStub(),
    )

    async def _fake_request(
        method: str,
        path: str,
        *,
        params: dict[str, object] | None = None,
        json_body: dict[str, object] | None = None,
    ) -> dict[str, object]:
        captured["method"] = method
        captured["path"] = path
        captured["payload"] = dict(json_body or {})
        return {
            "id": "alpaca-order-1",
            "client_order_id": str((json_body or {}).get("client_order_id", "")),
            "symbol": str((json_body or {}).get("symbol", "")),
            "side": str((json_body or {}).get("side", "")),
            "type": str((json_body or {}).get("type", "")),
            "qty": str((json_body or {}).get("qty", "")),
            "filled_qty": "0",
            "status": "accepted",
            "submitted_at": datetime(2026, 3, 3, tzinfo=UTC).isoformat(),
        }

    monkeypatch.setattr(adapter, "_request", _fake_request)

    state = await adapter.submit_order(
        OrderIntent(
            client_order_id="client-1",
            symbol="JNJ",
            side="buy",
            qty=Decimal("150.82"),
            order_type="market",
            metadata={"market": "stocks"},
        )
    )

    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["qty"] == "150"
    assert payload["time_in_force"] == "gtc"
    assert state.qty == Decimal("150")
    assert state.raw["requested_qty"] == "150.82"
    assert state.raw["submitted_qty"] == "150"
    assert state.raw["qty_normalization"] == "rounded_down_to_whole_share"
    await adapter.aclose()


async def test_010_accessibility_alpaca_small_fractional_stock_orders_use_day(monkeypatch) -> None:
    captured: dict[str, object] = {}
    adapter = AlpacaTradingAdapter(
        api_key="key",
        api_secret="secret",
        trading_base_url="https://paper-api.alpaca.markets",
        market_data_client=_MarketDataClientStub(),
    )

    async def _fake_request(
        _method: str,
        _path: str,
        *,
        params: dict[str, object] | None = None,
        json_body: dict[str, object] | None = None,
    ) -> dict[str, object]:
        captured["payload"] = dict(json_body or {})
        return {
            "id": "alpaca-order-2",
            "client_order_id": str((json_body or {}).get("client_order_id", "")),
            "symbol": str((json_body or {}).get("symbol", "")),
            "side": str((json_body or {}).get("side", "")),
            "type": str((json_body or {}).get("type", "")),
            "qty": str((json_body or {}).get("qty", "")),
            "filled_qty": "0",
            "status": "accepted",
            "submitted_at": datetime(2026, 3, 3, tzinfo=UTC).isoformat(),
        }

    monkeypatch.setattr(adapter, "_request", _fake_request)

    state = await adapter.submit_order(
        OrderIntent(
            client_order_id="client-2",
            symbol="BRK.B",
            side="buy",
            qty=Decimal("0.75"),
            order_type="market",
            metadata={"market": "stocks"},
        )
    )

    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["qty"] == "0.75"
    assert payload["time_in_force"] == "day"
    assert state.qty == Decimal("0.75")
    assert state.raw["requested_time_in_force"] == "gtc"
    assert state.raw["submitted_time_in_force"] == "day"
    await adapter.aclose()


async def test_020_accessibility_alpaca_crypto_orders_are_not_rounded(monkeypatch) -> None:
    captured: dict[str, object] = {}
    adapter = AlpacaTradingAdapter(
        api_key="key",
        api_secret="secret",
        trading_base_url="https://paper-api.alpaca.markets",
        market_data_client=_MarketDataClientStub(),
    )

    async def _fake_request(
        _method: str,
        _path: str,
        *,
        params: dict[str, object] | None = None,
        json_body: dict[str, object] | None = None,
    ) -> dict[str, object]:
        captured["payload"] = dict(json_body or {})
        return {
            "id": "alpaca-order-3",
            "client_order_id": str((json_body or {}).get("client_order_id", "")),
            "symbol": str((json_body or {}).get("symbol", "")),
            "side": str((json_body or {}).get("side", "")),
            "type": str((json_body or {}).get("type", "")),
            "qty": str((json_body or {}).get("qty", "")),
            "filled_qty": "0",
            "status": "accepted",
            "submitted_at": datetime(2026, 3, 3, tzinfo=UTC).isoformat(),
        }

    monkeypatch.setattr(adapter, "_request", _fake_request)

    state = await adapter.submit_order(
        OrderIntent(
            client_order_id="client-3",
            symbol="BTC/USD",
            side="buy",
            qty=Decimal("0.123456"),
            order_type="market",
            metadata={"market": "crypto"},
        )
    )

    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["qty"] == "0.123456"
    assert payload["time_in_force"] == "gtc"
    assert state.qty == Decimal("0.123456")
    await adapter.aclose()
