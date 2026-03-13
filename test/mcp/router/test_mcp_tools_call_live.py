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


def _build_valid_strategy_dsl() -> dict[str, Any]:
    return {
        "dsl_version": "1.0.0",
        "strategy": {
            "name": "Pytest Range Strategy",
            "description": "Validation probe for strategy_validate_dsl input compatibility.",
        },
        "universe": {
            "market": "forex",
            "tickers": ["USDJPY"],
        },
        "timeframe": "1h",
        "factors": {
            "bbands_20_2": {
                "type": "bbands",
                "params": {"length": 20, "std": 2.0, "source": "close"},
                "outputs": ["BBU", "BBM", "BBL"],
            },
            "rsi_14": {
                "type": "rsi",
                "params": {"length": 14, "source": "close"},
            },
        },
        "trade": {
            "long": {
                "entry": {
                    "condition": {
                        "cmp": {
                            "left": {"ref": "rsi_14"},
                            "op": "lt",
                            "right": 35,
                        }
                    }
                },
                "exits": [
                    {
                        "type": "signal_exit",
                        "name": "revert_to_mid",
                        "condition": {
                            "cross": {
                                "a": {"ref": "price.close"},
                                "op": "cross_above",
                                "b": {"ref": "bbands_20_2.BBM"},
                            }
                        },
                    }
                ],
                "position_sizing": {
                    "mode": "pct_equity",
                    "pct": 0.1,
                },
            }
        },
    }


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


def test_050_strategy_validate_dsl_accepts_object_payload() -> None:
    result = _call_tool(
        "strategy",
        "strategy_validate_dsl",
        {"dsl_json": _build_valid_strategy_dsl()},
    )
    assert result.get("isError") is False

    payload = _decode_tool_text_payload(result)
    assert payload["category"] == "strategy"
    assert payload["tool"] == "strategy_validate_dsl"
    assert payload["ok"] is True
    assert payload.get("errors") == []


def test_060_strategy_validate_dsl_auto_recovers_trailing_brace() -> None:
    dsl_text = json.dumps(
        _build_valid_strategy_dsl(),
        ensure_ascii=False,
        separators=(",", ":"),
    ) + "}"
    result = _call_tool(
        "strategy",
        "strategy_validate_dsl",
        {"dsl_json": dsl_text},
    )
    assert result.get("isError") is False

    payload = _decode_tool_text_payload(result)
    assert payload["tool"] == "strategy_validate_dsl"
    assert payload["ok"] is True


def test_070_indicator_catalog_exposes_dsl_alias_outputs() -> None:
    overlap_result = _call_tool(
        "strategy",
        "get_indicator_catalog",
        {"category": "overlap"},
    )
    overlap_payload = _decode_tool_text_payload(overlap_result)
    overlap_categories = overlap_payload.get("categories")
    assert isinstance(overlap_categories, list) and overlap_categories
    bbands = None
    for category in overlap_categories:
        indicators = category.get("indicators")
        if not isinstance(indicators, list):
            continue
        for indicator in indicators:
            if indicator.get("indicator") == "bbands":
                bbands = indicator
                break
        if bbands is not None:
            break
    assert isinstance(bbands, dict)
    bbands_outputs = bbands.get("outputs")
    assert isinstance(bbands_outputs, list) and bbands_outputs
    assert any(
        isinstance(item, dict)
        and item.get("name") == "BBU"
        and item.get("dsl_alias") == "upper"
        for item in bbands_outputs
    )

    momentum_result = _call_tool(
        "strategy",
        "get_indicator_catalog",
        {"category": "momentum"},
    )
    momentum_payload = _decode_tool_text_payload(momentum_result)
    momentum_categories = momentum_payload.get("categories")
    assert isinstance(momentum_categories, list) and momentum_categories
    adx = None
    for category in momentum_categories:
        indicators = category.get("indicators")
        if not isinstance(indicators, list):
            continue
        for indicator in indicators:
            if indicator.get("indicator") == "adx":
                adx = indicator
                break
        if adx is not None:
            break
    assert isinstance(adx, dict)
    adx_outputs = adx.get("outputs")
    assert isinstance(adx_outputs, list) and adx_outputs
    assert any(
        isinstance(item, dict)
        and item.get("name") == "ADX"
        and item.get("dsl_alias") == "adx"
        for item in adx_outputs
    )
