from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import httpx
import pytest

from packages.domain.market_data.data import DataLoader
from test._support.live_helpers import parse_sse_payloads

_BASE_URL = "http://127.0.0.1:8000"
_MCP_URL = "http://127.0.0.1:8110/market/mcp"
_DEFAULT_TIMEOUT = httpx.Timeout(connect=10.0, read=360.0, write=60.0, pool=60.0)
_SYMBOL_CANDIDATES: tuple[str, ...] = (
    "IBM",
    "INTC",
    "CSCO",
    "ADBE",
    "CRM",
    "UBER",
    "SHOP",
    "BABA",
    "PDD",
    "DIS",
    "BA",
    "KO",
    "PEP",
    "JNJ",
    "XOM",
    "CVX",
)


def _build_client() -> httpx.Client:
    return httpx.Client(base_url=_BASE_URL, timeout=_DEFAULT_TIMEOUT, trust_env=False)


def _login_headers(
    client: httpx.Client,
    seeded_user_credentials: tuple[str, str],
) -> dict[str, str]:
    email, password = seeded_user_credentials
    response = client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": password},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    access_token = str(payload["access_token"])
    return {"Authorization": f"Bearer {access_token}"}


def _create_thread(client: httpx.Client, auth_headers: dict[str, str]) -> str:
    response = client.post(
        "/api/v1/chat/new-thread",
        headers=auth_headers,
        json={"metadata": {"source": "pytest-pre-strategy-missing-symbol"}},
    )
    assert response.status_code == 201, response.text
    return str(response.json()["session_id"])


def _send_stream(
    client: httpx.Client,
    auth_headers: dict[str, str],
    *,
    session_id: str,
    message: str,
) -> list[dict[str, Any]]:
    response = client.post(
        "/api/v1/chat/send-openai-stream?language=zh",
        headers=auth_headers,
        json={
            "session_id": session_id,
            "client_turn_id": f"it_{uuid4().hex[:24]}",
            "message": message,
        },
    )
    assert response.status_code == 200, response.text
    payloads = parse_sse_payloads(response.text)
    assert payloads
    assert any(item.get("type") == "done" for item in payloads), payloads[-5:]
    return payloads


def _session_detail(
    client: httpx.Client,
    auth_headers: dict[str, str],
    *,
    session_id: str,
) -> dict[str, Any]:
    response = client.get(f"/api/v1/sessions/{session_id}", headers=auth_headers)
    assert response.status_code == 200, response.text
    payload = response.json()
    assert isinstance(payload, dict)
    return payload


def _latest_assistant_message(detail: dict[str, Any]) -> dict[str, Any]:
    messages = detail.get("messages")
    assert isinstance(messages, list)
    assistants = [
        item for item in messages if isinstance(item, dict) and item.get("role") == "assistant"
    ]
    assert assistants
    return assistants[-1]


def _all_assistant_tool_calls(detail: dict[str, Any]) -> list[dict[str, Any]]:
    messages = detail.get("messages")
    assert isinstance(messages, list)
    calls: list[dict[str, Any]] = []
    for item in messages:
        if not isinstance(item, dict) or item.get("role") != "assistant":
            continue
        tool_calls = item.get("tool_calls")
        if isinstance(tool_calls, list):
            calls.extend([row for row in tool_calls if isinstance(row, dict)])
    return calls


def _find_tool_call(
    calls: list[dict[str, Any]],
    *,
    name: str,
) -> dict[str, Any] | None:
    matched = [
        item
        for item in calls
        if str(item.get("type")) == "mcp_call" and str(item.get("name")) == name
    ]
    if not matched:
        return None
    return matched[-1]


def _json_loads_maybe(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _parse_iso_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _pick_missing_symbol() -> str:
    loader = DataLoader()
    existing = set(loader.get_available_symbols("us_stocks"))
    for symbol in _SYMBOL_CANDIDATES:
        if symbol not in existing:
            return symbol
    pytest.skip("No missing symbol left in predefined candidate list.")


def _call_market_mcp_tool(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    request_payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments,
        },
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    with httpx.Client(timeout=60.0, trust_env=False) as client:
        response = client.post(_MCP_URL, headers=headers, json=request_payload)
    assert response.status_code == 200, response.text

    payload = None
    for line in response.text.splitlines():
        if line.startswith("data: "):
            payload = json.loads(line.removeprefix("data: "))
            break
    assert isinstance(payload, dict), response.text[:500]
    result = payload.get("result")
    assert isinstance(result, dict), payload
    content = result.get("content")
    assert isinstance(content, list) and content, result
    text = content[0].get("text") if isinstance(content[0], dict) else None
    assert isinstance(text, str) and text.strip(), result
    parsed = json.loads(text)
    assert isinstance(parsed, dict)
    return parsed


def test_000_pre_strategy_missing_symbol_download_flow_with_openai_live(
    compose_stack: list[dict[str, Any]],
    seeded_user_credentials: tuple[str, str],
) -> None:
    _ = compose_stack
    symbol = _pick_missing_symbol()

    with _build_client() as client:
        auth_headers = _login_headers(client, seeded_user_credentials)
        session_id = _create_thread(client, auth_headers)

        turn1_payloads = _send_stream(
            client,
            auth_headers,
            session_id=session_id,
            message=(
                f"我想交易美股 {symbol}。"
                "机会频率是 daily，持仓是 swing_days。"
                "如果本地没有这个symbol，请先征求我同意，不要直接下载。"
            ),
        )
        openai_types = [
            str(item.get("openai_type"))
            for item in turn1_payloads
            if item.get("type") == "openai_event"
        ]
        assert "response.mcp_list_tools.completed" in openai_types

        detail_after_turn1 = _session_detail(
            client,
            auth_headers,
            session_id=session_id,
        )
        last_assistant = _latest_assistant_message(detail_after_turn1)
        content_text = str(last_assistant.get("content", ""))
        assert "下载" in content_text
        assert ("1-2分钟" in content_text) or ("1-2 分钟" in content_text) or ("2分钟" in content_text)

        all_calls_turn1 = _all_assistant_tool_calls(detail_after_turn1)
        check_call = _find_tool_call(all_calls_turn1, name="check_symbol_available")
        assert check_call is not None, all_calls_turn1
        check_args = _json_loads_maybe(check_call.get("arguments"))
        assert str(check_args.get("symbol", "")).upper() == symbol
        check_output = _json_loads_maybe(check_call.get("output"))
        assert check_output.get("tool") == "check_symbol_available"
        assert check_output.get("ok") is True
        assert check_output.get("available") is False

        _send_stream(
            client,
            auth_headers,
            session_id=session_id,
            message=(
                "我选择下载这个额外标的，接受等待1-2分钟。"
                "请按近两年 1m 数据下载，然后继续后续流程。"
            ),
        )
        detail_after_turn2 = _session_detail(
            client,
            auth_headers,
            session_id=session_id,
        )
        all_calls_turn2 = _all_assistant_tool_calls(detail_after_turn2)
        fetch_call = _find_tool_call(all_calls_turn2, name="market_data_fetch_missing_ranges")
        assert fetch_call is not None, all_calls_turn2

        fetch_args = _json_loads_maybe(fetch_call.get("arguments"))
        provider_arg = str(fetch_args.get("provider", "")).lower()
        assert provider_arg in {"alpaca", "default", "", "local_parquet"}
        assert str(fetch_args.get("timeframe", "")).lower() == "1m"
        assert str(fetch_args.get("symbol", "")).upper() == symbol

        start_date = str(fetch_args.get("start_date", "")).strip()
        end_date = str(fetch_args.get("end_date", "")).strip()
        max_lookback_days = int(fetch_args.get("max_lookback_days", 0) or 0)
        if start_date and end_date:
            start_dt = _parse_iso_utc(start_date)
            end_dt = _parse_iso_utc(end_date)
            assert end_dt > start_dt
            assert (end_dt - start_dt).days >= 700
        else:
            assert max_lookback_days >= 700

        fetch_output = _json_loads_maybe(fetch_call.get("output"))
        if (
            fetch_output.get("ok") is not True
            and str(fetch_output.get("error", {}).get("code", "")) == "PROVIDER_UNAVAILABLE"
        ):
            _send_stream(
                client,
                auth_headers,
                session_id=session_id,
                message=(
                    "请重试下载，并且 provider 必须使用 alpaca 字面值，"
                    "不要使用 default。仍然按近两年 1m 下载。"
                ),
            )
            detail_after_retry = _session_detail(
                client,
                auth_headers,
                session_id=session_id,
            )
            all_calls_retry = _all_assistant_tool_calls(detail_after_retry)
            fetch_call = _find_tool_call(all_calls_retry, name="market_data_fetch_missing_ranges")
            assert fetch_call is not None, all_calls_retry
            fetch_output = _json_loads_maybe(fetch_call.get("output"))

        assert fetch_output.get("ok") is True, fetch_output
        assert str(fetch_output.get("provider", "")).lower() == "alpaca", fetch_output
        sync_job_id = str(fetch_output.get("sync_job_id", "")).strip()
        assert sync_job_id, fetch_output

    time.sleep(75)
    deadline = time.monotonic() + 60.0
    latest_job_payload: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        latest_job_payload = _call_market_mcp_tool(
            "market_data_get_sync_job",
            {"sync_job_id": sync_job_id},
        )
        status = str(latest_job_payload.get("status", "")).strip().lower()
        if status in {"done", "failed"}:
            break
        time.sleep(10)

    assert isinstance(latest_job_payload, dict)
    final_status = str(latest_job_payload.get("status", "")).strip().lower()
    assert final_status in {"running", "done", "failed"}, latest_job_payload
    assert final_status != "pending", latest_job_payload
