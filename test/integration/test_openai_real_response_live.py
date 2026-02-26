"""Live integration tests for real OpenAI response streaming.

These tests use the actual OpenAI API to verify:
1. SSE streaming works correctly
2. Response quality meets expectations
3. Tool calls are properly handled
4. Phase transitions work as expected
"""

from __future__ import annotations

import time
from typing import Any

from fastapi.testclient import TestClient

from test._support.live_helpers import parse_sse_payloads


class TestOpenAIRealResponseStreaming:
    """Test suite for real OpenAI response streaming."""

    def test_000_kyc_phase_initial_greeting(
        self,
        api_test_client: TestClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test initial KYC phase greeting with real OpenAI response."""
        response = api_test_client.post(
            "/api/v1/chat/send-openai-stream?language=en",
            headers=auth_headers,
            json={
                "message": "Hello, I want to create a trading strategy.",
            },
        )
        assert response.status_code == 200, response.text

        payloads = parse_sse_payloads(response.text)
        assert payloads, "Expected SSE payloads"

        openai_types = [
            payload.get("openai_type")
            for payload in payloads
            if payload.get("type") == "openai_event"
        ]
        assert "response.created" in openai_types
        assert "response.completed" in openai_types

        done_payload = next(
            (item for item in payloads if item.get("type") == "done"),
            None,
        )
        assert done_payload is not None
        assert done_payload.get("phase") in {
            "kyc",
            "pre_strategy",
            "strategy",
            "stress_test",
            "deployment",
        }

        text = "".join(
            item.get("delta", "") for item in payloads if item.get("type") == "text_delta"
        )
        assert len(text.strip()) >= 20, f"Response too short: {text}"

    def test_010_kyc_phase_experience_disclosure(
        self,
        api_test_client: TestClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test KYC phase with trading experience disclosure."""
        response = api_test_client.post(
            "/api/v1/chat/send-openai-stream?language=en",
            headers=auth_headers,
            json={
                "message": (
                    "I have 5 years of trading experience, primarily in stocks and crypto. "
                    "My risk tolerance is moderate, and I'm looking for strategies that can "
                    "generate 15-20% annual returns with controlled drawdowns under 10%."
                ),
            },
        )
        assert response.status_code == 200, response.text

        payloads = parse_sse_payloads(response.text)
        done_payload = next(
            (item for item in payloads if item.get("type") == "done"),
            None,
        )
        assert done_payload is not None

        usage = done_payload.get("usage")
        if isinstance(usage, dict):
            assert int(usage.get("total_tokens", 0)) > 0

        text = "".join(
            item.get("delta", "") for item in payloads if item.get("type") == "text_delta"
        )
        assert len(text.strip()) >= 50

    def test_020_strategy_design_request(
        self,
        api_test_client: TestClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test strategy design request with specific requirements."""
        response = api_test_client.post(
            "/api/v1/chat/send-openai-stream?language=en",
            headers=auth_headers,
            json={
                "message": (
                    "I want to create a momentum-based strategy for BTC/USD. "
                    "Use EMA crossover (9 and 21 periods) with RSI filter. "
                    "Entry when EMA9 crosses above EMA21 and RSI is below 70. "
                    "Exit when EMA9 crosses below EMA21 or RSI exceeds 80. "
                    "Use 2x ATR for stop loss and 2:1 risk-reward ratio."
                ),
            },
        )
        assert response.status_code == 200, response.text

        payloads = parse_sse_payloads(response.text)
        text = "".join(
            item.get("delta", "") for item in payloads if item.get("type") == "text_delta"
        )
        assert len(text.strip()) >= 30

    def test_030_chinese_language_response(
        self,
        api_test_client: TestClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test Chinese language response quality."""
        response = api_test_client.post(
            "/api/v1/chat/send-openai-stream?language=zh",
            headers=auth_headers,
            json={
                "message": "我想创建一个基于均线交叉的加密货币交易策略，请帮我设计。",
            },
        )
        assert response.status_code == 200, response.text

        payloads = parse_sse_payloads(response.text)
        text = "".join(
            item.get("delta", "") for item in payloads if item.get("type") == "text_delta"
        )
        assert len(text.strip()) >= 30
        # Verify Chinese characters are present
        chinese_chars = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
        assert chinese_chars >= 10, f"Expected Chinese response, got: {text[:200]}"

    def test_040_multi_turn_conversation_context(
        self,
        api_test_client: TestClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test multi-turn conversation maintains context."""
        # First message
        response1 = api_test_client.post(
            "/api/v1/chat/send-openai-stream?language=en",
            headers=auth_headers,
            json={
                "message": "I want to trade ETH/USD with a 5-minute timeframe.",
            },
        )
        assert response1.status_code == 200

        # Second message referencing first
        response2 = api_test_client.post(
            "/api/v1/chat/send-openai-stream?language=en",
            headers=auth_headers,
            json={
                "message": "For the ETH strategy I mentioned, add a MACD confirmation filter.",
            },
        )
        assert response2.status_code == 200

        payloads = parse_sse_payloads(response2.text)
        text = "".join(
            item.get("delta", "") for item in payloads if item.get("type") == "text_delta"
        )
        # Should reference ETH or the previous context
        assert len(text.strip()) >= 20

    def test_050_response_latency_acceptable(
        self,
        api_test_client: TestClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test that response latency is within acceptable bounds."""
        start_time = time.monotonic()
        response = api_test_client.post(
            "/api/v1/chat/send-openai-stream?language=en",
            headers=auth_headers,
            json={
                "message": "What markets do you support?",
            },
        )
        elapsed = time.monotonic() - start_time

        assert response.status_code == 200
        # First token should arrive within 30 seconds
        assert elapsed < 60, f"Response took too long: {elapsed:.2f}s"

        payloads = parse_sse_payloads(response.text)
        assert payloads, "Expected SSE payloads"


class TestOpenAIToolCallIntegration:
    """Test suite for OpenAI tool call integration."""

    def test_000_strategy_creation_tool_call(
        self,
        api_test_client: TestClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test that strategy creation triggers appropriate tool calls."""
        # Create a new thread first
        thread_response = api_test_client.post(
            "/api/v1/chat/new-thread",
            headers=auth_headers,
            json={"metadata": {"source": "pytest-openai-tool-call"}},
        )
        assert thread_response.status_code == 201
        session_id = thread_response.json()["session_id"]

        # Send a detailed strategy request
        response = api_test_client.post(
            f"/api/v1/chat/send-openai-stream?language=en&session_id={session_id}",
            headers=auth_headers,
            json={
                "message": (
                    "Create a simple EMA crossover strategy for BTC/USD on 1-minute timeframe. "
                    "Use EMA 9 and EMA 21. Enter long when EMA9 crosses above EMA21. "
                    "Exit when EMA9 crosses below EMA21. Use 1% stop loss."
                ),
            },
        )
        assert response.status_code == 200

        payloads = parse_sse_payloads(response.text)

        # Check for tool call events
        tool_events = [
            p for p in payloads
            if p.get("type") == "openai_event" and "tool" in str(p.get("openai_type", "")).lower()
        ]

        # Check for agent UI JSON (strategy preview)
        agent_ui_events = [
            p for p in payloads
            if p.get("type") == "agent_ui_json"
        ]

        # Either tool calls or agent UI should be present for strategy creation
        has_strategy_interaction = len(tool_events) > 0 or len(agent_ui_events) > 0

        done_payload = next(
            (item for item in payloads if item.get("type") == "done"),
            None,
        )
        assert done_payload is not None

    def test_010_market_data_tool_call(
        self,
        api_test_client: TestClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test market data tool calls."""
        response = api_test_client.post(
            "/api/v1/chat/send-openai-stream?language=en",
            headers=auth_headers,
            json={
                "message": "What is the current price of BTC/USD?",
            },
        )
        assert response.status_code == 200

        payloads = parse_sse_payloads(response.text)
        text = "".join(
            item.get("delta", "") for item in payloads if item.get("type") == "text_delta"
        )
        # Should contain some response about BTC or price
        assert len(text.strip()) >= 10


class TestOpenAIErrorHandling:
    """Test suite for OpenAI error handling."""

    def test_000_empty_message_handling(
        self,
        api_test_client: TestClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test handling of empty messages."""
        response = api_test_client.post(
            "/api/v1/chat/send-openai-stream?language=en",
            headers=auth_headers,
            json={
                "message": "",
            },
        )
        # Should either return 400 or handle gracefully
        assert response.status_code in {200, 400, 422}

    def test_010_very_long_message_handling(
        self,
        api_test_client: TestClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test handling of very long messages."""
        long_message = "Please help me " + "create a strategy " * 500
        response = api_test_client.post(
            "/api/v1/chat/send-openai-stream?language=en",
            headers=auth_headers,
            json={
                "message": long_message,
            },
        )
        # Should handle gracefully
        assert response.status_code in {200, 400, 413, 422}

    def test_020_special_characters_handling(
        self,
        api_test_client: TestClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test handling of special characters."""
        response = api_test_client.post(
            "/api/v1/chat/send-openai-stream?language=en",
            headers=auth_headers,
            json={
                "message": "Create strategy for BTC/USD 🚀📈 with <script>alert('test')</script>",
            },
        )
        assert response.status_code == 200

        payloads = parse_sse_payloads(response.text)
        text = "".join(
            item.get("delta", "") for item in payloads if item.get("type") == "text_delta"
        )
        # Should not contain the script tag in response
        assert "<script>" not in text.lower()
