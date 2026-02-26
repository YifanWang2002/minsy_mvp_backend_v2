from __future__ import annotations

import json
import re
import time
from pathlib import Path

import httpx


def _extract_first_sse_json(raw_text: str) -> dict[str, object]:
    for line in raw_text.splitlines():
        if not line.startswith("data: "):
            continue
        return json.loads(line.removeprefix("data: "))
    raise AssertionError(f"No SSE data payload found: {raw_text[:300]}")


def _request_with_retry(
    *,
    method: str,
    url: str,
    timeout_seconds: int = 120,
    **kwargs: object,
) -> httpx.Response:
    deadline = time.monotonic() + float(timeout_seconds)
    last_error: str = ""
    with httpx.Client(timeout=30.0, trust_env=False) as client:
        while time.monotonic() < deadline:
            try:
                response = client.request(method, url, **kwargs)
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                time.sleep(2)
                continue
            if response.status_code < 500:
                return response
            last_error = f"status={response.status_code}, body={response.text[:300]}"
            time.sleep(2)
    raise AssertionError(f"Request did not become healthy: {url} ({last_error})")


def test_000_accessibility_local_mcp_router_domains(
    compose_stack: list[dict[str, object]],
) -> None:
    _ = compose_stack
    domains = ("strategy", "backtest", "market", "stress", "trading")
    for domain in domains:
        response = _request_with_retry(
            method="GET",
            url=f"http://127.0.0.1:8110/{domain}/mcp",
        )
        assert response.status_code < 500, (domain, response.status_code, response.text)


def test_010_local_mcp_list_tools_for_each_domain_returns_tools(
    compose_stack: list[dict[str, object]],
) -> None:
    _ = compose_stack
    request_payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list",
        "params": {},
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }

    domains = ("strategy", "backtest", "market", "stress", "trading")
    for domain in domains:
        response = _request_with_retry(
            method="POST",
            url=f"http://127.0.0.1:8110/{domain}/mcp",
            headers=headers,
            json=request_payload,
        )
        assert response.status_code == 200, (domain, response.status_code, response.text)
        payload = _extract_first_sse_json(response.text)
        result = payload.get("result")
        assert isinstance(result, dict), payload
        tools = result.get("tools")
        assert isinstance(tools, list) and len(tools) > 0, (domain, payload)


def test_020_configured_public_mcp_urls_are_present() -> None:
    content = Path("env/.env.dev").read_text(encoding="utf-8")
    keys = (
        "MCP_SERVER_URL_STRATEGY_DEV",
        "MCP_SERVER_URL_BACKTEST_DEV",
        "MCP_SERVER_URL_MARKET_DATA_DEV",
        "MCP_SERVER_URL_STRESS_DEV",
        "MCP_SERVER_URL_TRADING_DEV",
    )
    for key in keys:
        match = re.search(rf"^{key}=(.+)$", content, flags=re.MULTILINE)
        assert match is not None, key
        value = match.group(1).strip()
        assert value.startswith("https://"), (key, value)
        assert value.endswith("/mcp"), (key, value)
