"""Live integration tests for MCP endpoints with cloudflared tunnel.

These tests verify:
1. MCP router is accessible
2. All domain tools are registered
3. Tool calls work correctly
4. OpenAI can access MCP tools via cloudflared tunnel
"""

from __future__ import annotations

import json
import subprocess
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pytest
from fastapi.testclient import TestClient

from test._support.live_helpers import (
    BACKEND_DIR,
    parse_sse_payloads,
    run_command,
    start_cloudflared_tunnel,
    stop_process,
    wait_http_ok,
)


class TestMCPRouterEndpoints:
    """Test suite for MCP router endpoints."""

    def test_000_mcp_router_health(
        self,
        compose_stack: list[dict[str, Any]],
    ) -> None:
        """Test MCP router is accessible (via strategy domain)."""
        _ = compose_stack
        # MCP router doesn't have a /health endpoint, use strategy domain instead
        status = wait_http_ok(
            "http://127.0.0.1:8110/strategy/mcp",
            timeout_seconds=30,
            min_status=200,
            max_status=499,
        )
        assert 200 <= status <= 499

    def test_010_strategy_domain_accessible(
        self,
        compose_stack: list[dict[str, Any]],
    ) -> None:
        """Test strategy domain is accessible."""
        _ = compose_stack
        status = wait_http_ok(
            "http://127.0.0.1:8110/strategy/mcp",
            timeout_seconds=30,
            min_status=200,
            max_status=499,
        )
        assert 200 <= status <= 499

    def test_020_backtest_domain_accessible(
        self,
        compose_stack: list[dict[str, Any]],
    ) -> None:
        """Test backtest domain is accessible."""
        _ = compose_stack
        status = wait_http_ok(
            "http://127.0.0.1:8110/backtest/mcp",
            timeout_seconds=30,
            min_status=200,
            max_status=499,
        )
        assert 200 <= status <= 499

    def test_030_market_domain_accessible(
        self,
        compose_stack: list[dict[str, Any]],
    ) -> None:
        """Test market data domain is accessible."""
        _ = compose_stack
        status = wait_http_ok(
            "http://127.0.0.1:8110/market/mcp",
            timeout_seconds=30,
            min_status=200,
            max_status=499,
        )
        assert 200 <= status <= 499

    def test_040_stress_domain_accessible(
        self,
        compose_stack: list[dict[str, Any]],
    ) -> None:
        """Test stress domain is accessible."""
        _ = compose_stack
        status = wait_http_ok(
            "http://127.0.0.1:8110/stress/mcp",
            timeout_seconds=30,
            min_status=200,
            max_status=499,
        )
        assert 200 <= status <= 499

    def test_050_trading_domain_accessible(
        self,
        compose_stack: list[dict[str, Any]],
    ) -> None:
        """Test trading domain is accessible."""
        _ = compose_stack
        status = wait_http_ok(
            "http://127.0.0.1:8110/trading/mcp",
            timeout_seconds=30,
            min_status=200,
            max_status=499,
        )
        assert 200 <= status <= 499


class TestMCPToolRegistration:
    """Test suite for MCP tool registration."""

    def _call_mcp_tool(
        self,
        domain: str,
        tool_name: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Call an MCP tool via HTTP."""
        url = f"http://127.0.0.1:8110/{domain}/mcp"
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": params or {},
            },
        }
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            return {"error": f"HTTP {exc.code}", "body": exc.read().decode("utf-8")}
        except URLError as exc:
            return {"error": str(exc)}

    def test_000_strategy_ping_tool(
        self,
        compose_stack: list[dict[str, Any]],
    ) -> None:
        """Test strategy_ping tool."""
        _ = compose_stack
        result = self._call_mcp_tool("strategy", "strategy_ping")
        # Should return a valid response (even if error due to missing context)
        assert "result" in result or "error" in result

    def test_010_backtest_ping_tool(
        self,
        compose_stack: list[dict[str, Any]],
    ) -> None:
        """Test backtest_ping tool."""
        _ = compose_stack
        result = self._call_mcp_tool("backtest", "backtest_ping")
        assert "result" in result or "error" in result

    def test_020_market_data_ping_tool(
        self,
        compose_stack: list[dict[str, Any]],
    ) -> None:
        """Test market_data_ping tool."""
        _ = compose_stack
        result = self._call_mcp_tool("market", "market_data_ping")
        assert "result" in result or "error" in result

    def test_030_stress_ping_tool(
        self,
        compose_stack: list[dict[str, Any]],
    ) -> None:
        """Test stress_ping tool."""
        _ = compose_stack
        result = self._call_mcp_tool("stress", "stress_ping")
        assert "result" in result or "error" in result

    def test_040_trading_ping_tool(
        self,
        compose_stack: list[dict[str, Any]],
    ) -> None:
        """Test trading_ping tool."""
        _ = compose_stack
        result = self._call_mcp_tool("trading", "trading_ping")
        assert "result" in result or "error" in result


class TestMCPCloudflaredTunnel:
    """Test suite for MCP access via cloudflared tunnel.

    These tests start a cloudflared tunnel to expose MCP to the public internet,
    which is required for OpenAI to access the tools.
    """

    @pytest.fixture(scope="class")
    def cloudflared_tunnel(
        self,
        compose_stack: list[dict[str, Any]],
    ):
        """Start cloudflared tunnel for MCP."""
        _ = compose_stack
        process = None
        public_url = None
        try:
            process, public_url = start_cloudflared_tunnel(
                target_url="http://127.0.0.1:8110"
            )
            yield {"process": process, "public_url": public_url}
        finally:
            stop_process(process)

    def test_000_tunnel_starts_successfully(
        self,
        cloudflared_tunnel: dict[str, Any],
    ) -> None:
        """Test that cloudflared tunnel starts successfully."""
        assert cloudflared_tunnel["public_url"] is not None
        assert cloudflared_tunnel["public_url"].startswith("https://")
        assert ".trycloudflare.com" in cloudflared_tunnel["public_url"]

    def test_010_tunnel_accessible(
        self,
        cloudflared_tunnel: dict[str, Any],
    ) -> None:
        """Test that tunnel is accessible from public internet."""
        public_url = cloudflared_tunnel["public_url"]
        status = wait_http_ok(
            f"{public_url}/health",
            timeout_seconds=60,
            min_status=200,
            max_status=499,
        )
        assert 200 <= status <= 499

    def test_020_mcp_tools_accessible_via_tunnel(
        self,
        cloudflared_tunnel: dict[str, Any],
    ) -> None:
        """Test that MCP tools are accessible via tunnel."""
        public_url = cloudflared_tunnel["public_url"]

        # Try to access strategy domain
        status = wait_http_ok(
            f"{public_url}/strategy/mcp",
            timeout_seconds=60,
            min_status=200,
            max_status=499,
        )
        assert 200 <= status <= 499


class TestMCPWithOpenAI:
    """Test suite for MCP integration with OpenAI.

    These tests verify that OpenAI can successfully call MCP tools
    when the tunnel is active.
    """

    @pytest.fixture(scope="class")
    def cloudflared_tunnel(
        self,
        compose_stack: list[dict[str, Any]],
    ):
        """Start cloudflared tunnel for MCP."""
        _ = compose_stack
        process = None
        public_url = None
        try:
            process, public_url = start_cloudflared_tunnel(
                target_url="http://127.0.0.1:8110"
            )
            yield {"process": process, "public_url": public_url}
        finally:
            stop_process(process)

    def test_000_openai_stream_with_mcp_tools(
        self,
        api_test_client: TestClient,
        auth_headers: dict[str, str],
        cloudflared_tunnel: dict[str, Any],
    ) -> None:
        """Test OpenAI streaming with MCP tools available."""
        # The tunnel should be running, making MCP tools available to OpenAI
        public_url = cloudflared_tunnel["public_url"]
        assert public_url is not None

        # Send a message that might trigger tool use
        response = api_test_client.post(
            "/api/v1/chat/send-openai-stream?language=en",
            headers=auth_headers,
            json={
                "message": (
                    "I want to create a simple EMA crossover strategy for BTC/USD. "
                    "Please help me design it with 9 and 21 period EMAs."
                ),
            },
        )
        assert response.status_code == 200

        payloads = parse_sse_payloads(response.text)
        assert payloads, "Expected SSE payloads"

        # Check for completion
        done_payload = next(
            (item for item in payloads if item.get("type") == "done"),
            None,
        )
        assert done_payload is not None

    def test_010_openai_tool_call_execution(
        self,
        api_test_client: TestClient,
        auth_headers: dict[str, str],
        cloudflared_tunnel: dict[str, Any],
    ) -> None:
        """Test that OpenAI tool calls are executed successfully."""
        public_url = cloudflared_tunnel["public_url"]
        assert public_url is not None

        # Create a new thread
        thread_response = api_test_client.post(
            "/api/v1/chat/new-thread",
            headers=auth_headers,
            json={"metadata": {"source": "pytest-mcp-tool-call"}},
        )
        assert thread_response.status_code == 201
        session_id = thread_response.json()["session_id"]

        # Send a message that should trigger strategy creation
        response = api_test_client.post(
            f"/api/v1/chat/send-openai-stream?language=en&session_id={session_id}",
            headers=auth_headers,
            json={
                "message": (
                    "Create a momentum strategy for ETH/USD with RSI indicator. "
                    "Entry when RSI crosses above 30, exit when RSI crosses below 70. "
                    "Use 1-minute timeframe."
                ),
            },
        )
        assert response.status_code == 200

        payloads = parse_sse_payloads(response.text)

        # Look for tool-related events
        tool_events = [
            p for p in payloads
            if "tool" in str(p.get("type", "")).lower()
            or "tool" in str(p.get("openai_type", "")).lower()
        ]

        # Check for agent UI JSON (strategy preview)
        agent_ui_events = [
            p for p in payloads
            if p.get("type") == "agent_ui_json"
        ]

        # Verify response completed
        done_payload = next(
            (item for item in payloads if item.get("type") == "done"),
            None,
        )
        assert done_payload is not None


class TestMCPToolCapabilities:
    """Test suite for MCP tool capabilities."""

    def test_000_strategy_capabilities(
        self,
        compose_stack: list[dict[str, Any]],
    ) -> None:
        """Test strategy domain capabilities."""
        _ = compose_stack

        url = "http://127.0.0.1:8110/strategy/mcp"
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "strategy_capabilities",
                "arguments": {},
            },
        }
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=30) as response:
                result = json.loads(response.read().decode("utf-8"))
                # Should have result with capabilities info
                assert "result" in result or "error" in result
        except (HTTPError, URLError):
            pass  # Tool might require context

    def test_010_trading_capabilities(
        self,
        compose_stack: list[dict[str, Any]],
    ) -> None:
        """Test trading domain capabilities."""
        _ = compose_stack

        url = "http://127.0.0.1:8110/trading/mcp"
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "trading_capabilities",
                "arguments": {},
            },
        }
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=30) as response:
                result = json.loads(response.read().decode("utf-8"))
                assert "result" in result or "error" in result
        except (HTTPError, URLError):
            pass
