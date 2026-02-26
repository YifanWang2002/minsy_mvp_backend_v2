from __future__ import annotations

import json
import time
from typing import Any

import httpx


def _extract_first_sse_json(raw_text: str) -> dict[str, Any]:
    for line in raw_text.splitlines():
        if line.startswith("data: "):
            return json.loads(line.removeprefix("data: "))
    raise AssertionError(f"No SSE payload found: {raw_text[:300]}")


def _call_tool(domain: str, tool_name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    request_payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments or {},
        },
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }

    deadline = time.monotonic() + 120.0
    last_error = ""
    with httpx.Client(timeout=30.0, trust_env=False) as client:
        while time.monotonic() < deadline:
            response = client.post(
                f"http://127.0.0.1:8110/{domain}/mcp",
                headers=headers,
                json=request_payload,
            )
            if response.status_code == 200:
                payload = _extract_first_sse_json(response.text)
                result = payload.get("result")
                assert isinstance(result, dict), payload
                return result
            last_error = f"status={response.status_code} body={response.text[:300]}"
            time.sleep(2)
    raise AssertionError(f"Tool call failed: {domain}/{tool_name}, {last_error}")


def _decode_tool_text_payload(result: dict[str, Any]) -> dict[str, Any]:
    content = result.get("content")
    assert isinstance(content, list) and content, result
    first = content[0]
    assert isinstance(first, dict), result
    text = first.get("text")
    assert isinstance(text, str) and text, result
    return json.loads(text)


def test_000_accessibility_tools_call_stress_ping() -> None:
    result = _call_tool("stress", "stress_ping")
    assert result.get("isError") is False


def test_010_tools_call_trading_ping() -> None:
    result = _call_tool("trading", "trading_ping")
    assert result.get("isError") is False


def test_020_tools_call_strategy_indicator_catalog() -> None:
    result = _call_tool("strategy", "get_indicator_catalog")
    assert result.get("isError") is False


def test_030_tools_call_market_symbol_available() -> None:
    result = _call_tool(
        "market",
        "check_symbol_available",
        {"symbol": "SPY", "market": "stock"},
    )
    assert result.get("isError") is False


def test_040_tools_call_backtest_get_job_invalid_uuid_returns_structured_error() -> None:
    result = _call_tool(
        "backtest",
        "backtest_get_job",
        {"job_id": "not-a-uuid"},
    )
    assert result.get("isError") is False

    payload = _decode_tool_text_payload(result)
    assert payload["category"] == "backtest"
    assert payload["tool"] == "backtest_get_job"
    assert payload["ok"] is False
    error = payload.get("error")
    assert isinstance(error, dict)
    assert "code" in error
