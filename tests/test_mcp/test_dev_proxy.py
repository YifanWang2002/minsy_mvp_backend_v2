from __future__ import annotations

import httpx
from fastapi.testclient import TestClient

from src.mcp.dev_proxy import app, build_upstream_url, resolve_proxy_target


def test_resolve_proxy_target_strategy_path_strips_prefix(monkeypatch) -> None:
    monkeypatch.delenv("MCP_PROXY_UPSTREAM_STRATEGY", raising=False)
    target = resolve_proxy_target("/strategy/mcp")

    assert target is not None
    assert target.upstream_base_url == "http://127.0.0.1:8111"
    assert target.rewritten_path == "/mcp"


def test_resolve_proxy_target_exact_prefix_maps_to_root(monkeypatch) -> None:
    monkeypatch.delenv("MCP_PROXY_UPSTREAM_MARKET", raising=False)
    target = resolve_proxy_target("/market")

    assert target is not None
    assert target.upstream_base_url == "http://127.0.0.1:8113"
    assert target.rewritten_path == "/"


def test_resolve_proxy_target_honors_env_override(monkeypatch) -> None:
    monkeypatch.setenv("MCP_PROXY_UPSTREAM_BACKTEST", "http://127.0.0.1:19112/")
    target = resolve_proxy_target("/backtest/mcp")

    assert target is not None
    assert target.upstream_base_url == "http://127.0.0.1:19112"
    assert target.rewritten_path == "/mcp"


def test_resolve_proxy_target_unknown_path_returns_none() -> None:
    assert resolve_proxy_target("/unknown/mcp") is None
    assert resolve_proxy_target("/mcp") is None


def test_build_upstream_url_joins_query() -> None:
    url = build_upstream_url(
        upstream_base_url="http://127.0.0.1:8111/",
        rewritten_path="/mcp",
        query="a=1&b=2",
    )
    assert url == "http://127.0.0.1:8111/mcp?a=1&b=2"


def test_proxy_request_returns_502_when_upstream_unreachable() -> None:
    with TestClient(app) as client:
        async def _raise_connect_error(request: httpx.Request, stream: bool = False):  # noqa: ARG001
            raise httpx.ConnectError("connection refused", request=request)

        client.app.state.http.send = _raise_connect_error
        response = client.get("/strategy/mcp")

    assert response.status_code == 502
    assert response.text == "MCP upstream unavailable: http://127.0.0.1:8111/mcp"
