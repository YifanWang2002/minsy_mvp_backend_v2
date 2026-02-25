from __future__ import annotations

import httpx
import pytest

from src.engine.execution.alpaca_account_probe import AlpacaAccountProbe


@pytest.mark.asyncio
async def test_probe_passes_when_paper_200_and_live_401() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "paper-api.alpaca.markets":
            return httpx.Response(200, json={"status": "ACTIVE"})
        if request.url.host == "api.alpaca.markets":
            return httpx.Response(401, json={"message": "unauthorized"})
        raise AssertionError(f"Unexpected host: {request.url.host}")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        probe = AlpacaAccountProbe(client=client)
        result = await probe.probe_credentials(api_key="key", api_secret="secret")

    assert result.ok is True
    assert result.status == "paper_probe_ok"
    assert result.metadata["paper_http_status"] == 200
    assert result.metadata["live_http_status"] == 401
    assert result.metadata["paper_account_status"] == "ACTIVE"


@pytest.mark.asyncio
async def test_probe_fails_when_paper_401() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "paper-api.alpaca.markets":
            return httpx.Response(401, json={"message": "unauthorized"})
        if request.url.host == "api.alpaca.markets":
            return httpx.Response(401, json={"message": "unauthorized"})
        raise AssertionError(f"Unexpected host: {request.url.host}")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        probe = AlpacaAccountProbe(client=client)
        result = await probe.probe_credentials(api_key="key", api_secret="secret")

    assert result.ok is False
    assert result.status == "paper_probe_failed"
    assert result.metadata["paper_http_status"] == 401
    assert result.message == "Invalid Alpaca credentials or wrong endpoint."


@pytest.mark.asyncio
async def test_probe_fails_when_live_returns_200() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "paper-api.alpaca.markets":
            return httpx.Response(200, json={"status": "ACTIVE"})
        if request.url.host == "api.alpaca.markets":
            return httpx.Response(200, json={"status": "ACTIVE"})
        raise AssertionError(f"Unexpected host: {request.url.host}")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        probe = AlpacaAccountProbe(client=client)
        result = await probe.probe_credentials(api_key="key", api_secret="secret")

    assert result.ok is False
    assert result.status == "paper_probe_failed"
    assert result.metadata["paper_http_status"] == 200
    assert result.metadata["live_http_status"] == 200
    assert result.message == "Credentials are not paper-only keys."
