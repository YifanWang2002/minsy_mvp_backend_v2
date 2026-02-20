"""Stress-test MCP tool placeholders for future optimization workflows."""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from src.mcp._utils import log_mcp_tool_result, to_json, utc_now_iso

TOOL_NAMES: tuple[str, ...] = (
    "stress_ping",
    "stress_capabilities",
)


def _payload(
    *,
    tool: str,
    ok: bool,
    data: dict[str, Any] | None = None,
) -> str:
    body: dict[str, Any] = {
        "category": "stress",
        "tool": tool,
        "ok": ok,
        "timestamp_utc": utc_now_iso(),
    }
    if isinstance(data, dict) and data:
        body.update(data)
    log_mcp_tool_result(category="stress", tool=tool, ok=ok)
    return to_json(body)


def stress_ping() -> str:
    """Health probe for stress-test MCP domain."""
    return _payload(
        tool="stress_ping",
        ok=True,
        data={"status": "ready"},
    )


def stress_capabilities() -> str:
    """Advertise currently available stress-test capabilities."""
    return _payload(
        tool="stress_capabilities",
        ok=True,
        data={
            "available": False,
            "message": (
                "Stress optimization tools are not enabled yet. "
                "This domain is reserved for upcoming CPU-intensive workflows."
            ),
        },
    )


def register_stress_tools(mcp: FastMCP) -> None:
    """Register stress-domain tools."""
    mcp.tool()(stress_ping)
    mcp.tool()(stress_capabilities)
