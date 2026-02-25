from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import httpx
import pytest

from src.engine.execution.adapters.base import OhlcvBar
from src.engine.market_data.alpaca_client import AlpacaMarketDataClient
from src.engine.market_data.providers.alpaca_rest import AlpacaRestProvider


def _http_404(path: str) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", f"https://data.alpaca.markets{path}")
    response = httpx.Response(404, request=request, json={"message": "Not Found"})
    return httpx.HTTPStatusError("404 Not Found", request=request, response=response)


@pytest.mark.asyncio
async def test_fetch_latest_bar_crypto_fallback_path() -> None:
    calls: list[str] = []
    client = AlpacaMarketDataClient(api_key="demo", api_secret="demo")

    async def fake_get(path: str, *, params: dict[str, object]) -> dict[str, object]:
        calls.append(path)
        if path.endswith("/bars/latest"):
            raise _http_404(path)
        assert params["symbols"] == "BTC/USD"
        return {
            "bars": {
                "BTC/USD": {
                    "t": "2026-02-21T05:00:00Z",
                    "o": 100,
                    "h": 101,
                    "l": 99,
                    "c": 100.5,
                    "v": 10,
                }
            }
        }

    client._get = fake_get  # type: ignore[method-assign]
    try:
        bar = await client.fetch_latest_bar("BTCUSD", market="crypto")
    finally:
        await client.aclose()

    assert bar is not None
    assert float(bar.close) == 100.5
    assert calls == [
        "/v1beta3/crypto/us/bars/latest",
        "/v1beta3/crypto/us/latest/bars",
    ]


@pytest.mark.asyncio
async def test_rest_provider_quote_falls_back_to_latest_bar_when_quote_missing() -> None:
    client = AlpacaMarketDataClient(api_key="demo", api_secret="demo")
    provider = AlpacaRestProvider(client=client)

    async def fake_quote(symbol: str, *, market: str):
        assert symbol == "BTCUSD"
        assert market == "crypto"
        return None

    async def fake_bar(symbol: str, *, market: str):
        assert symbol == "BTCUSD"
        assert market == "crypto"
        return OhlcvBar(
            timestamp=datetime(2026, 2, 21, 5, 0, tzinfo=UTC),
            open=Decimal("100"),
            high=Decimal("101"),
            low=Decimal("99"),
            close=Decimal("100.5"),
            volume=Decimal("10"),
        )

    client.fetch_latest_quote = fake_quote  # type: ignore[method-assign]
    client.fetch_latest_bar = fake_bar  # type: ignore[method-assign]
    quote = await provider.fetch_quote(symbol="BTCUSD", market="crypto")
    await provider.aclose()

    assert quote is not None
    assert quote.bid is None
    assert quote.ask is None
    assert float(quote.last or Decimal("0")) == 100.5
