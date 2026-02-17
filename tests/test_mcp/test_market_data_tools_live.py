"""Live market-data MCP tests (real provider calls, no mocks)."""

from __future__ import annotations

import json
from typing import Any

import pytest
from mcp.server.fastmcp import FastMCP

from src.mcp.market_data import tools as market_tools


def _extract_payload(call_result: object) -> dict[str, Any]:
    if isinstance(call_result, tuple) and len(call_result) == 2:
        maybe_result = call_result[1]
        if isinstance(maybe_result, dict):
            raw = maybe_result.get("result")
            if isinstance(raw, str):
                return json.loads(raw)
    raise AssertionError(f"Unexpected call result: {call_result!r}")


def _skip_on_live_rate_limit(payload: dict[str, Any]) -> None:
    if payload.get("ok") is True:
        return
    error = payload.get("error")
    code = str(error.get("code", "")).strip() if isinstance(error, dict) else ""
    if code == "UPSTREAM_RATE_LIMIT":
        pytest.skip("Live yfinance quota/rate-limit reached during test run.")


@pytest.mark.asyncio
async def test_market_data_tools_fetch_live_yfinance_quote_and_candles() -> None:
    mcp = FastMCP("test-market-data-live")
    market_tools.register_market_data_tools(mcp)

    quote_call = await mcp.call_tool(
        "get_symbol_quote",
        {"market": "stock", "symbol": "AAPL"},
    )
    quote_payload = _extract_payload(quote_call)
    _skip_on_live_rate_limit(quote_payload)
    assert quote_payload["ok"] is True
    assert quote_payload["yfinance_symbol"] == "AAPL"
    assert isinstance(quote_payload["quote"], dict)

    candles_call = await mcp.call_tool(
        "get_symbol_candles",
        {
            "market": "stock",
            "symbol": "AAPL",
            "period": "5d",
            "interval": "1d",
        },
    )
    candles_payload = _extract_payload(candles_call)
    _skip_on_live_rate_limit(candles_payload)
    assert candles_payload["ok"] is True
    assert candles_payload["yfinance_symbol"] == "AAPL"
    assert int(candles_payload.get("rows", 0)) > 0

    metadata_call = await mcp.call_tool(
        "get_symbol_metadata",
        {"market": "stock", "symbol": "AAPL"},
    )
    metadata_payload = _extract_payload(metadata_call)
    _skip_on_live_rate_limit(metadata_payload)
    assert metadata_payload["ok"] is True
    assert metadata_payload["yfinance_symbol"] == "AAPL"
    assert isinstance(metadata_payload.get("metadata"), dict)
