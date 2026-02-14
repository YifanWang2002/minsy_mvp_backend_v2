"""MCP package for Minsy backend."""

from src.mcp.server import (
    ALL_REGISTERED_TOOL_NAMES,
    create_mcp_server,
)

__all__ = [
    "ALL_REGISTERED_TOOL_NAMES",
    "create_mcp_server",
]
