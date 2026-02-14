"""Regression tests for detailed stream-error payload propagation."""

from __future__ import annotations

import json
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient

from src.main import app


def _register_and_get_token(client: TestClient) -> str:
    email = f"stream_error_detail_{uuid4().hex}@test.com"
    response = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "pass1234", "name": "Stream Error Detail User"},
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


def test_done_event_contains_structured_stream_error_detail() -> None:
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
            "type": "response.stream_error",
            "error": {
                "class": "APIStatusError",
                "message": (
                    "Error retrieving tool list from mcp server: "
                    "dial tcp 1.2.3.4:443: i/o timeout (status=424)"
                ),
                "retryable": True,
                "attempt": 3,
                "diagnostics": {
                    "category": "mcp_list_tools_fetch_failed",
                    "status_code": 424,
                    "request_id": "req_stream_001",
                    "upstream_error_code": "MCP_LIST_TOOLS_FAILED",
                    "upstream_message": (
                        "Error retrieving tool list from mcp server: "
                        "dial tcp 1.2.3.4:443: i/o timeout"
                    ),
                },
            },
        }

    with patch(
        "src.services.openai_stream_service.OpenAIResponsesEventStreamer.stream_events",
        side_effect=_mock_stream_events,
    ):
        with TestClient(app) as client:
            token = _register_and_get_token(client)
            response = client.post(
                "/api/v1/chat/send-openai-stream?language=en",
                headers={"Authorization": f"Bearer {token}"},
                json={"message": "hello"},
            )

    assert response.status_code == 200
    payloads = _parse_sse_payloads(response.text)
    done = next(item for item in payloads if item.get("type") == "done")

    assert "tool list" in str(done.get("stream_error", "")).lower()
    detail = done.get("stream_error_detail")
    assert isinstance(detail, dict)
    diagnostics = detail.get("diagnostics")
    assert isinstance(diagnostics, dict)
    assert diagnostics.get("category") == "mcp_list_tools_fetch_failed"
    assert diagnostics.get("status_code") == 424
    assert diagnostics.get("request_id") == "req_stream_001"
