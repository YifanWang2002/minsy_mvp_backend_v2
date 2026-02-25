from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest

from src.engine.market_data.alpaca_client import AlpacaMarketDataClient


@pytest.mark.asyncio
async def test_fetch_ohlcv_crypto_symbol_uses_slash_format() -> None:
    captured: dict[str, object] = {}

    client = AlpacaMarketDataClient(api_key="demo", api_secret="demo")

    async def fake_get(path: str, *, params: dict[str, object]) -> dict[str, object]:
        captured["path"] = path
        captured["params"] = dict(params)
        return {
            "bars": {
                "BTC/USD": [
                    {"t": "2026-02-21T05:00:00Z", "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 10}
                ]
            }
        }

    client._get = fake_get  # type: ignore[method-assign]
    try:
        bars = await client.fetch_ohlcv(
            "BTCUSD",
            market="crypto",
            timeframe="1Min",
            limit=1,
            since=datetime(2026, 2, 21, 4, 59, tzinfo=UTC),
        )
    finally:
        await client.aclose()

    assert captured["path"] == "/v1beta3/crypto/us/bars"
    assert isinstance(captured["params"], dict)
    params = captured["params"]
    assert params["symbols"] == "BTC/USD"
    assert params["timeframe"] == "1Min"
    assert len(bars) == 1


def _http_404(path: str) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", f"https://data.alpaca.markets{path}")
    response = httpx.Response(404, request=request, json={"message": "Not Found"})
    return httpx.HTTPStatusError("404 Not Found", request=request, response=response)


@pytest.mark.asyncio
async def test_fetch_latest_quote_crypto_fallback_to_latest_quotes_path() -> None:
    calls: list[str] = []
    client = AlpacaMarketDataClient(api_key="demo", api_secret="demo")

    async def fake_get(path: str, *, params: dict[str, object]) -> dict[str, object]:
        calls.append(path)
        if path.endswith("/latest/quotes"):
            raise _http_404(path)
        assert params["symbols"] == "BTC/USD"
        return {
            "quotes": {
                "BTC/USD": {
                    "bp": 100.0,
                    "ap": 101.0,
                    "p": 100.5,
                    "t": "2026-02-21T05:00:00Z",
                }
            }
        }

    client._get = fake_get  # type: ignore[method-assign]
    try:
        quote = await client.fetch_latest_quote("BTCUSD", market="crypto")
    finally:
        await client.aclose()

    assert quote is not None
    assert quote.symbol == "BTCUSD"
    assert calls == [
        "/v1beta3/crypto/us/latest/quotes",
        "/v1beta3/crypto/us/quotes/latest",
    ]


@pytest.mark.asyncio
async def test_fetch_latest_quote_crypto_returns_none_if_both_paths_404() -> None:
    calls: list[str] = []
    client = AlpacaMarketDataClient(api_key="demo", api_secret="demo")

    async def fake_get(path: str, *, params: dict[str, object]) -> dict[str, object]:
        calls.append(path)
        assert params["symbols"] == "BTC/USD"
        raise _http_404(path)

    client._get = fake_get  # type: ignore[method-assign]
    try:
        quote = await client.fetch_latest_quote("BTCUSD", market="crypto")
    finally:
        await client.aclose()

    assert quote is None
    assert calls == [
        "/v1beta3/crypto/us/latest/quotes",
        "/v1beta3/crypto/us/quotes/latest",
    ]


@pytest.mark.asyncio
async def test_fetch_latest_quote_infers_crypto_for_compact_symbol() -> None:
    captured: dict[str, object] = {}

    client = AlpacaMarketDataClient(api_key="demo", api_secret="demo")

    async def fake_get(path: str, *, params: dict[str, object]) -> dict[str, object]:
        captured["path"] = path
        captured["params"] = dict(params)
        return {
            "quotes": {
                "BTC/USD": {
                    "bp": 100.0,
                    "ap": 101.0,
                    "p": 100.5,
                    "t": "2026-02-21T05:00:00Z",
                }
            }
        }

    client._get = fake_get  # type: ignore[method-assign]
    try:
        quote = await client.fetch_latest_quote("BTCUSD")
    finally:
        await client.aclose()

    assert quote is not None
    assert quote.symbol == "BTCUSD"
    assert captured["path"] == "/v1beta3/crypto/us/latest/quotes"
    assert isinstance(captured["params"], dict)
    params = captured["params"]
    assert params["symbols"] == "BTC/USD"


@pytest.mark.asyncio
async def test_fetch_latest_quote_uses_mid_price_when_last_missing() -> None:
    client = AlpacaMarketDataClient(api_key="demo", api_secret="demo")

    async def fake_get(path: str, *, params: dict[str, object]) -> dict[str, object]:
        assert path == "/v1beta3/crypto/us/latest/quotes"
        assert params["symbols"] == "BTC/USD"
        return {
            "quotes": {
                "BTC/USD": {
                    "bp": 100.0,
                    "ap": 101.0,
                    "t": "2026-02-21T05:00:00Z",
                }
            }
        }

    client._get = fake_get  # type: ignore[method-assign]
    try:
        quote = await client.fetch_latest_quote("BTCUSD", market="crypto")
    finally:
        await client.aclose()

    assert quote is not None
    assert quote.last is not None
    assert float(quote.last) == 100.5


@pytest.mark.asyncio
async def test_fetch_latest_quote_retries_after_timeout() -> None:
    attempts = 0
    client = AlpacaMarketDataClient(
        api_key="demo",
        api_secret="demo",
        max_retries=1,
        retry_backoff_seconds=0.01,
    )

    async def fake_get(url: str, *, headers: dict[str, str], params: dict[str, object]):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            request = httpx.Request("GET", url, headers=headers, params=params)
            raise httpx.ReadTimeout("timeout", request=request)
        request = httpx.Request("GET", url, headers=headers, params=params)
        return httpx.Response(
            200,
            request=request,
            json={
                "quotes": {
                    "AAPL": {"bp": 100.0, "ap": 101.0, "p": 100.5, "t": "2026-02-21T05:00:00Z"}
                }
            },
        )

    client._client.get = fake_get  # type: ignore[method-assign]
    try:
        quote = await client.fetch_latest_quote("AAPL", market="stocks")
    finally:
        await client.aclose()

    assert quote is not None
    assert attempts == 2
