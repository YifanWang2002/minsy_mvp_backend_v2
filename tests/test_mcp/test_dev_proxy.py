from __future__ import annotations

from src.mcp.dev_proxy import build_upstream_url, resolve_proxy_target


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


def test_resolve_proxy_target_legacy_mcp_keeps_prefix(monkeypatch) -> None:
    monkeypatch.delenv("MCP_PROXY_UPSTREAM_LEGACY", raising=False)
    target = resolve_proxy_target("/mcp")

    assert target is not None
    assert target.upstream_base_url == "http://127.0.0.1:8111"
    assert target.rewritten_path == "/mcp"


def test_resolve_proxy_target_honors_env_override(monkeypatch) -> None:
    monkeypatch.setenv("MCP_PROXY_UPSTREAM_BACKTEST", "http://127.0.0.1:19112/")
    target = resolve_proxy_target("/backtest/mcp")

    assert target is not None
    assert target.upstream_base_url == "http://127.0.0.1:19112"
    assert target.rewritten_path == "/mcp"


def test_resolve_proxy_target_unknown_path_returns_none() -> None:
    assert resolve_proxy_target("/unknown/mcp") is None


def test_build_upstream_url_joins_query() -> None:
    url = build_upstream_url(
        upstream_base_url="http://127.0.0.1:8111/",
        rewritten_path="/mcp",
        query="a=1&b=2",
    )
    assert url == "http://127.0.0.1:8111/mcp?a=1&b=2"
