"""Unified modular MCP server entrypoint."""

from __future__ import annotations

import argparse
import sys

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from src.mcp.backtest import TOOL_NAMES as BACKTEST_TOOL_NAMES
from src.mcp.backtest import register_backtest_tools
from src.mcp.market_data import TOOL_NAMES as MARKET_DATA_TOOL_NAMES
from src.mcp.market_data import register_market_data_tools
from src.mcp.strategy import TOOL_NAMES as STRATEGY_TOOL_NAMES
from src.mcp.strategy import register_strategy_tools

ALL_REGISTERED_TOOL_NAMES: tuple[str, ...] = (
    *MARKET_DATA_TOOL_NAMES,
    *BACKTEST_TOOL_NAMES,
    *STRATEGY_TOOL_NAMES,
)


def create_mcp_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8111,
    mount_path: str = "/",
) -> FastMCP:
    """Create the modular MCP server and register all tool groups."""
    mcp = FastMCP(
        name="Minsy Modular MCP Server",
        instructions=(
            "Modular MCP server with tools grouped by market_data, "
            "backtest, and strategy domains."
        ),
        host=host,
        port=port,
        mount_path=mount_path,
        streamable_http_path="/mcp",
        stateless_http=True,
        # Local development frequently uses reverse tunnels for remote MCP probes.
        # Disable host-header rebinding protection so tunneled hostnames are accepted.
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=False
        ),
    )
    register_market_data_tools(mcp)
    register_backtest_tools(mcp)
    register_strategy_tools(mcp)
    return mcp


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the modular MCP server.")
    parser.add_argument(
        "--transport",
        choices=("streamable-http", "sse", "stdio"),
        default="streamable-http",
        help="MCP transport mode.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host for HTTP/SSE modes.")
    parser.add_argument("--port", type=int, default=8111, help="Port for HTTP/SSE modes.")
    parser.add_argument(
        "--mount-path",
        default="/",
        help="Mount path for HTTP/SSE modes (default '/').",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    mcp = create_mcp_server(host=args.host, port=args.port, mount_path=args.mount_path)
    print(
        f"[mcp] starting modular server transport={args.transport} "
        f"host={args.host} port={args.port} mount_path={args.mount_path}",
        file=sys.stderr,
    )
    print(
        "[mcp] registered tools: " + ", ".join(ALL_REGISTERED_TOOL_NAMES),
        file=sys.stderr,
    )
    mcp.run(transport=args.transport)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
