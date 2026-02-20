from __future__ import annotations

import json
from typing import Any

import pytest

from src.mcp.server import create_mcp_server, registered_tool_names


def _extract_payload(call_result: object) -> dict[str, Any]:
    if isinstance(call_result, tuple) and len(call_result) == 2:
        maybe_result = call_result[1]
        if isinstance(maybe_result, dict):
            raw = maybe_result.get("result")
            if isinstance(raw, str):
                return json.loads(raw)
    raise AssertionError(f"Unexpected call result: {call_result!r}")


@pytest.mark.asyncio
async def test_strategy_domain_only_registers_strategy_tools() -> None:
    mcp = create_mcp_server(domain="strategy")
    result = await mcp.call_tool("get_indicator_catalog", {})
    payload = _extract_payload(result)

    assert payload["category"] == "strategy"
    assert payload["tool"] == "get_indicator_catalog"
    assert payload["ok"] is True

    with pytest.raises(Exception):  # noqa: BLE001
        await mcp.call_tool("backtest_get_job", {"job_id": "x"})


@pytest.mark.asyncio
async def test_market_alias_domain_registers_market_data_tools() -> None:
    mcp = create_mcp_server(domain="market_data")
    result = await mcp.call_tool(
        "get_symbol_quote",
        {"market": "stock", "symbol": ""},
    )
    payload = _extract_payload(result)

    assert payload["category"] == "market_data"
    assert payload["tool"] == "get_symbol_quote"
    assert payload["ok"] is False
    assert payload["error"]["code"] == "INVALID_INPUT"


@pytest.mark.asyncio
async def test_stress_and_trading_domains_expose_placeholder_tools() -> None:
    stress_mcp = create_mcp_server(domain="stress")
    trading_mcp = create_mcp_server(domain="trading")

    stress_result = _extract_payload(await stress_mcp.call_tool("stress_ping", {}))
    trading_result = _extract_payload(await trading_mcp.call_tool("trading_ping", {}))

    assert stress_result["category"] == "stress"
    assert stress_result["ok"] is True
    assert trading_result["category"] == "trading"
    assert trading_result["ok"] is True


def test_registered_tool_names_and_domain_validation() -> None:
    all_names = registered_tool_names(domain="all")
    stress_names = registered_tool_names(domain="stress")
    market_names = registered_tool_names(domain="market_data")

    assert "strategy_validate_dsl" in all_names
    assert "backtest_create_job" in all_names
    assert "get_symbol_quote" in all_names
    assert "stress_ping" not in all_names  # legacy all-mode compatibility

    assert stress_names == ("stress_ping", "stress_capabilities")
    assert "get_symbol_quote" in market_names

    with pytest.raises(ValueError):
        create_mcp_server(domain="unknown-domain")
