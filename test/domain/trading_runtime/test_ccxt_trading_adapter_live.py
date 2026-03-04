from __future__ import annotations

import asyncio
import sys
import types
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from packages.infra.providers.trading.adapters.base import OrderIntent
from packages.infra.providers.trading.adapters.ccxt_trading import CcxtTradingAdapter


def _install_fake_ccxt(monkeypatch) -> None:
    class _BaseExchange:
        def __init__(self, _params: dict[str, Any]) -> None:
            self.urls: dict[str, Any] = {"api": {"rest": "https://default.rest"}}
            self.headers: dict[str, str] = {}
            self.has: dict[str, Any] = {"fetchCurrencies": True}
            self.markets: dict[str, dict[str, Any]] = {"BTC/USDT": {}}
            self.last_create_order: dict[str, Any] | None = None
            self.last_fetch_order: dict[str, Any] | None = None
            self.sandbox_mode: bool = False

        def set_sandbox_mode(self, enabled: bool) -> None:
            self.sandbox_mode = bool(enabled)

        async def load_markets(self) -> dict[str, Any]:
            return self.markets

        async def create_order(
            self,
            *,
            symbol: str,
            type: str,
            side: str,
            amount: float,
            price: float | None,
            params: dict[str, Any],
        ) -> dict[str, Any]:
            self.last_create_order = {
                "symbol": symbol,
                "type": type,
                "side": side,
                "amount": amount,
                "price": price,
                "params": dict(params),
            }
            return {
                "id": "fake-order-id",
                "clientOrderId": params.get("clientOrderId"),
                "symbol": symbol,
                "side": side,
                "type": type,
                "amount": amount,
                "filled": amount,
                "status": "closed",
                "average": price or 1.0,
                "timestamp": int(datetime.now(UTC).timestamp() * 1000),
            }

        async def fetch_order(
            self,
            order_id: str,
            symbol: str | None = None,
        ) -> dict[str, Any]:
            self.last_fetch_order = {"order_id": order_id, "symbol": symbol}
            return {
                "id": order_id,
                "clientOrderId": "client-fetch-1",
                "symbol": symbol or "BTC/USDT",
                "side": "buy",
                "type": "market",
                "amount": 1,
                "filled": 1,
                "status": "closed",
                "average": 1.0,
                "timestamp": int(datetime.now(UTC).timestamp() * 1000),
            }

        async def close(self) -> None:
            return None

    class _OkxExchange(_BaseExchange):
        def __init__(self, params: dict[str, Any]) -> None:
            super().__init__(params)
            self.urls = {"api": {"rest": "https://www.okx.com"}}

    class _BinanceExchange(_BaseExchange):
        def __init__(self, params: dict[str, Any]) -> None:
            super().__init__(params)
            self.urls = {"api": {"rest": "https://api.binance.com"}}

    async_support = types.ModuleType("ccxt.async_support")
    setattr(async_support, "okx", _OkxExchange)
    setattr(async_support, "binance", _BinanceExchange)

    ccxt_pkg = types.ModuleType("ccxt")
    setattr(ccxt_pkg, "async_support", async_support)

    monkeypatch.setitem(sys.modules, "ccxt", ccxt_pkg)
    monkeypatch.setitem(sys.modules, "ccxt.async_support", async_support)


def test_000_accessibility_okx_sandbox_sets_demo_headers(monkeypatch) -> None:
    _install_fake_ccxt(monkeypatch)

    adapter = CcxtTradingAdapter(
        exchange_id="okx",
        api_key="k",
        api_secret="s",
        password="p",
        sandbox=True,
    )

    exchange = adapter._exchange
    assert exchange.urls["api"]["rest"] == "https://us.okx.com"
    assert exchange.headers.get("x-simulated-trading") == "1"
    assert exchange.has.get("fetchCurrencies") is False
    asyncio.run(adapter.aclose())


async def test_010_accessibility_okx_submit_order_adds_td_mode(monkeypatch) -> None:
    _install_fake_ccxt(monkeypatch)

    adapter = CcxtTradingAdapter(
        exchange_id="okx",
        api_key="k",
        api_secret="s",
        password="p",
        sandbox=True,
    )

    await adapter.submit_order(
        OrderIntent(
            client_order_id="client-1",
            symbol="BTC/USDT",
            side="buy",
            qty=1,
            order_type="market",
        )
    )
    params = adapter._exchange.last_create_order["params"]
    assert params.get("clOrdId") == "client1"
    assert params.get("tdMode") == "cash"
    await adapter.aclose()


def test_020_accessibility_non_okx_not_forced_with_okx_headers(monkeypatch) -> None:
    _install_fake_ccxt(monkeypatch)

    adapter = CcxtTradingAdapter(
        exchange_id="binance",
        api_key="k",
        api_secret="s",
        sandbox=True,
    )
    exchange = adapter._exchange
    assert exchange.urls["api"]["rest"] == "https://api.binance.com"
    assert exchange.headers.get("x-simulated-trading") is None
    asyncio.run(adapter.aclose())


async def test_030_accessibility_okx_account_state_prefers_info_equity_fields(
    monkeypatch,
) -> None:
    _install_fake_ccxt(monkeypatch)

    adapter = CcxtTradingAdapter(
        exchange_id="okx",
        api_key="k",
        api_secret="s",
        password="p",
        sandbox=True,
    )

    async def _fake_fetch_balance() -> dict[str, Any]:
        return {
            "free": {"USDT": "4999178.417952595"},
            "used": {"USDT": "0"},
            "total": {"USDT": "4999178.417952595"},
            "info": {
                "data": [
                    {
                        "totalEq": "5000.0",
                        "availEq": "4999.17",
                        "details": [
                            {
                                "ccy": "USDT",
                                "availEq": "4999.17",
                                "eqUsd": "4999.17",
                            }
                        ],
                    }
                ]
            },
        }

    adapter._exchange.fetch_balance = _fake_fetch_balance
    try:
        state = await adapter.fetch_account_state()

        assert state.equity == Decimal("5000")
        assert state.cash == Decimal("4999.17")
        assert state.buying_power == Decimal("4999.17")
    finally:
        await adapter.aclose()


async def test_040_accessibility_okx_fetch_order_passes_symbol_when_provided(
    monkeypatch,
) -> None:
    _install_fake_ccxt(monkeypatch)

    adapter = CcxtTradingAdapter(
        exchange_id="okx",
        api_key="k",
        api_secret="s",
        password="p",
        sandbox=True,
    )

    try:
        state = await adapter.fetch_order("ord-1", symbol="BTCUSD")

        assert state is not None
        assert adapter._exchange.last_fetch_order == {
            "order_id": "ord-1",
            "symbol": "BTC/USDT",
        }
    finally:
        await adapter.aclose()
