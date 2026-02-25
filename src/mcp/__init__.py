"""MCP package for Minsy backend."""

from __future__ import annotations

__all__ = [
    "create_mcp_server",
    "registered_tool_names",
]


def __getattr__(name: str):  # noqa: ANN201
    if name in __all__:
        from src.mcp import server as server_module

        return getattr(server_module, name)
    raise AttributeError(f"module 'src.mcp' has no attribute {name!r}")
