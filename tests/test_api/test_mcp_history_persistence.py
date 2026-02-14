"""Regression tests for MCP tool-call persistence in session history."""

from __future__ import annotations

import json
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient

from src.main import app


def _register_and_get_token(client: TestClient) -> str:
    email = f"mcp_hist_{uuid4().hex}@test.com"
    response = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "pass1234", "name": "MCP History User"},
    )
    assert response.status_code == 201
    return response.json()["access_token"]


def _parse_sse_payloads(raw_text: str) -> list[dict]:
    payloads: list[dict] = []
    blocks = [block.strip() for block in raw_text.split("\n\n") if block.strip()]
    for block in blocks:
        for line in block.splitlines():
            if line.startswith("data: "):
                payloads.append(json.loads(line.removeprefix("data: ")))
    return payloads


def test_session_history_persists_mcp_final_results_only() -> None:
    async def _mock_stream_events(
        *,
        model: str,
        input_text: str,
        instructions: str | None = None,
        previous_response_id: str | None = None,
        tools: list | None = None,
        tool_choice: dict | None = None,
        reasoning: dict | None = None,
    ):
        yield {
            "type": "response.output_text.delta",
            "delta": "I checked the tool results.",
            "sequence_number": 1,
        }
        yield {
            "type": "response.output_item.added",
            "sequence_number": 2,
            "item": {
                "type": "mcp_call",
                "id": "call_success",
                "name": "check_symbol_available",
                "status": "in_progress",
                "arguments": {"symbol": "SPY"},
            },
        }
        yield {
            "type": "response.mcp_call.completed",
            "sequence_number": 3,
            "call_id": "call_success",
            "name": "check_symbol_available",
            "result": {"ok": True},
        }
        yield {
            "type": "response.output_item.done",
            "sequence_number": 4,
            "item": {
                "type": "mcp_call",
                "id": "call_success",
                "name": "check_symbol_available",
                "status": "completed",
                "arguments": {"symbol": "SPY"},
                "output": {"ok": True},
            },
        }
        yield {
            "type": "response.output_item.added",
            "sequence_number": 5,
            "item": {
                "type": "mcp_call",
                "id": "call_failure",
                "name": "get_quote",
                "status": "in_progress",
                "arguments": {"symbol": "SPY"},
            },
        }
        yield {
            "type": "response.mcp_call.failed",
            "sequence_number": 6,
            "call_id": "call_failure",
            "name": "get_quote",
            "error": "upstream timeout",
        }
        yield {
            "type": "response.output_item.done",
            "sequence_number": 7,
            "item": {
                "type": "mcp_call",
                "id": "call_failure",
                "name": "get_quote",
                "status": "failed",
                "arguments": {"symbol": "SPY"},
                "error": "upstream timeout",
            },
        }
        yield {
            "type": "response.completed",
            "response": {
                "id": "resp_mock_mcp_history",
                "usage": {"input_tokens": 32, "output_tokens": 18},
            },
        }

    with patch(
        "src.services.openai_stream_service.OpenAIResponsesEventStreamer.stream_events",
        side_effect=_mock_stream_events,
    ):
        with TestClient(app) as client:
            token = _register_and_get_token(client)
            headers = {"Authorization": f"Bearer {token}"}

            response = client.post(
                "/api/v1/chat/send-openai-stream",
                headers=headers,
                json={"message": "Please run tools."},
            )
            assert response.status_code == 200
            payloads = _parse_sse_payloads(response.text)
            done = next(item for item in payloads if item.get("type") == "done")
            session_id = done["session_id"]

            detail = client.get(f"/api/v1/sessions/{session_id}", headers=headers)
            assert detail.status_code == 200
            body = detail.json()

            assistant_messages = [
                message
                for message in body["messages"]
                if message.get("role") == "assistant"
            ]
            assert assistant_messages
            tool_calls = assistant_messages[-1].get("tool_calls") or []

            mcp_calls = [
                item for item in tool_calls if isinstance(item, dict) and item.get("type") == "mcp_call"
            ]
            assert len(mcp_calls) == 2
            assert all(item.get("status") in {"success", "failure"} for item in mcp_calls)

            by_name = {
                str(item.get("name")): item
                for item in mcp_calls
                if isinstance(item.get("name"), str)
            }
            assert by_name["check_symbol_available"]["status"] == "success"
            assert by_name["check_symbol_available"]["arguments"]["symbol"] == "SPY"
            assert by_name["get_quote"]["status"] == "failure"
            assert by_name["get_quote"]["error"] == "upstream timeout"
