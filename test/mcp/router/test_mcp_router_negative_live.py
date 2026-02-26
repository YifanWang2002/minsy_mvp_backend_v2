from __future__ import annotations

import httpx


def test_000_accessibility_unknown_mcp_route_returns_404() -> None:
    with httpx.Client(timeout=20.0, trust_env=False) as client:
        response = client.get("http://127.0.0.1:8110/unknown/mcp")
    assert response.status_code == 404, response.text


def test_010_invalid_accept_header_returns_4xx() -> None:
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list",
        "params": {},
    }
    with httpx.Client(timeout=20.0, trust_env=False) as client:
        response = client.post(
            "http://127.0.0.1:8110/strategy/mcp",
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            json=payload,
        )
    assert 400 <= response.status_code < 500, response.text


def test_020_invalid_method_returns_jsonrpc_error() -> None:
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "nonexistent/method",
        "params": {},
    }
    with httpx.Client(timeout=20.0, trust_env=False) as client:
        response = client.post(
            "http://127.0.0.1:8110/strategy/mcp",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            },
            json=payload,
        )
    assert response.status_code in {200, 400, 404}, response.text
