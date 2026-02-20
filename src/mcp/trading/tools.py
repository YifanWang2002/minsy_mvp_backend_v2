"""Trading MCP tool placeholders for paper/live execution domains."""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from src.mcp._utils import log_mcp_tool_result, to_json, utc_now_iso

TOOL_NAMES: tuple[str, ...] = (
    "trading_ping",
    "trading_capabilities",
)


def _payload(
    *,
    tool: str,
    ok: bool,
    data: dict[str, Any] | None = None,
) -> str:
    body: dict[str, Any] = {
        "category": "trading",
        "tool": tool,
        "ok": ok,
        "timestamp_utc": utc_now_iso(),
    }
    if isinstance(data, dict) and data:
        body.update(data)
    log_mcp_tool_result(category="trading", tool=tool, ok=ok)
    return to_json(body)


def trading_ping() -> str:
    """Health probe for trading MCP domain."""
    return _payload(
        tool="trading_ping",
        ok=True,
        data={"status": "ready"},
    )


def trading_capabilities() -> str:
    """Advertise currently available trading-domain capabilities."""
    return _payload(
        tool="trading_capabilities",
        ok=True,
        data={
            "available": False,
            "message": (
                "Paper/live trading tools are not enabled yet. "
                "This domain is reserved for future execution workflows."
            ),
        },
    )


def register_trading_tools(mcp: FastMCP) -> None:
    """Register trading-domain tools."""
    mcp.tool()(trading_ping)
    mcp.tool()(trading_capabilities)
