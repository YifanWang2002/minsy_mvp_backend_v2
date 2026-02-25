"""Domain-aware MCP server entrypoint."""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from dataclasses import dataclass

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from src.mcp.backtest import TOOL_NAMES as BACKTEST_TOOL_NAMES
from src.mcp.backtest import register_backtest_tools
from src.mcp.market_data import TOOL_NAMES as MARKET_DATA_TOOL_NAMES
from src.mcp.market_data import register_market_data_tools
from src.mcp.strategy import TOOL_NAMES as STRATEGY_TOOL_NAMES
from src.mcp.strategy import register_strategy_tools
from src.mcp.stress import TOOL_NAMES as STRESS_TOOL_NAMES
from src.mcp.stress import register_stress_tools
from src.mcp.trading import TOOL_NAMES as TRADING_TOOL_NAMES
from src.mcp.trading import register_trading_tools
from src.observability.sentry_setup import init_backend_sentry
from src.util.logger import configure_logging, logger


@dataclass(frozen=True, slots=True)
class McpDomainSpec:
    domain: str
    display_name: str
    instructions: str
    register_tools: Callable[[FastMCP], None]
    tool_names: tuple[str, ...]


_DOMAIN_SPECS: dict[str, McpDomainSpec] = {
    "strategy": McpDomainSpec(
        domain="strategy",
        display_name="Minsy Strategy MCP Server",
        instructions="MCP server for strategy DSL validation, storage, and versioning tools.",
        register_tools=register_strategy_tools,
        tool_names=STRATEGY_TOOL_NAMES,
    ),
    "backtest": McpDomainSpec(
        domain="backtest",
        display_name="Minsy Backtest MCP Server",
        instructions="MCP server for backtest job execution and analysis tools.",
        register_tools=register_backtest_tools,
        tool_names=BACKTEST_TOOL_NAMES,
    ),
    "market": McpDomainSpec(
        domain="market",
        display_name="Minsy Market Data MCP Server",
        instructions="MCP server for market data coverage, quote, candle, and metadata tools.",
        register_tools=register_market_data_tools,
        tool_names=MARKET_DATA_TOOL_NAMES,
    ),
    "stress": McpDomainSpec(
        domain="stress",
        display_name="Minsy Stress MCP Server",
        instructions="MCP server for stress testing and strategy optimization tools.",
        register_tools=register_stress_tools,
        tool_names=STRESS_TOOL_NAMES,
    ),
    "trading": McpDomainSpec(
        domain="trading",
        display_name="Minsy Trading MCP Server",
        instructions="MCP server reserved for paper/live trading execution tools.",
        register_tools=register_trading_tools,
        tool_names=TRADING_TOOL_NAMES,
    ),
}

_DOMAIN_ALIASES: dict[str, str] = {
    "market_data": "market",
    "market-data": "market",
}

SUPPORTED_DOMAINS: tuple[str, ...] = (
    "strategy",
    "backtest",
    "market",
    "market_data",
    "market-data",
    "stress",
    "trading",
)


def _resolve_domain(value: str) -> str:
    normalized = value.strip().lower()
    aliased = _DOMAIN_ALIASES.get(normalized, normalized)
    if aliased not in _DOMAIN_SPECS:
        raise ValueError(
            f"Unsupported MCP domain '{value}'. "
            f"Use one of: {', '.join(SUPPORTED_DOMAINS)}."
        )
    return aliased


def registered_tool_names(*, domain: str) -> tuple[str, ...]:
    """Return registered tool names for one MCP domain."""
    resolved_domain = _resolve_domain(domain)
    return _DOMAIN_SPECS[resolved_domain].tool_names


def create_mcp_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8111,
    mount_path: str = "/",
    stateless_http: bool = True,
    domain: str = "strategy",
) -> FastMCP:
    """Create one MCP server for a specific domain."""
    resolved_domain = _resolve_domain(domain)
    domain_spec = _DOMAIN_SPECS[resolved_domain]
    name = domain_spec.display_name
    instructions = domain_spec.instructions

    mcp = FastMCP(
        name=name,
        instructions=instructions,
        host=host,
        port=port,
        mount_path=mount_path,
        streamable_http_path="/mcp",
        stateless_http=stateless_http,
        # Local development frequently uses reverse tunnels for remote MCP probes.
        # Disable host-header rebinding protection so tunneled hostnames are accepted.
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=False
        ),
    )
    domain_spec.register_tools(mcp)
    return mcp


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a domain MCP server.")
    parser.add_argument(
        "--domain",
        choices=SUPPORTED_DOMAINS,
        default="strategy",
        help=(
            "Server domain to run: strategy, backtest, "
            "market/market_data, stress, or trading."
        ),
    )
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
    parser.add_argument(
        "--stateful-http",
        action="store_true",
        help="Use stateful streamable HTTP mode (default is stateless).",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    configure_logging(level=os.getenv("LOG_LEVEL", "INFO"), show_sql=False)
    init_backend_sentry(source="mcp")
    resolved_domain = _resolve_domain(args.domain)
    mcp = create_mcp_server(
        host=args.host,
        port=args.port,
        mount_path=args.mount_path,
        stateless_http=not bool(args.stateful_http),
        domain=resolved_domain,
    )
    logger.info(
        "mcp server starting domain=%s transport=%s host=%s port=%s mount_path=%s",
        resolved_domain,
        args.transport,
        args.host,
        args.port,
        args.mount_path,
    )
    logger.info(
        "mcp server registered tools: %s",
        ", ".join(registered_tool_names(domain=resolved_domain)),
    )
    mcp.run(transport=args.transport)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
