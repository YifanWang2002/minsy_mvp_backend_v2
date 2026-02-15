"""Regression tests for chat debug trace headers and stream compatibility."""

from __future__ import annotations

import json
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient

from src.main import app


def _register_and_get_token(client: TestClient) -> str:
    email = f"trace_header_{uuid4().hex}@test.com"
    response = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "pass1234", "name": "Trace Header User"},
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


def test_chat_stream_sets_trace_header_when_enabled() -> None:
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
        del (
            model,
            input_text,
            instructions,
            previous_response_id,
            tools,
            tool_choice,
            reasoning,
        )
        yield {"type": "response.output_text.delta", "delta": "ok", "sequence_number": 1}
        yield {
            "type": "response.completed",
            "response": {"id": f"resp_{uuid4().hex}", "usage": {"output_tokens": 1}},
        }

    with patch(
        "src.services.openai_stream_service.OpenAIResponsesEventStreamer.stream_events",
        side_effect=_mock_stream_events,
    ):
        with TestClient(app) as client:
            token = _register_and_get_token(client)
            trace_id = "frontend_trace_001"
            response = client.post(
                "/api/v1/chat/send-openai-stream?language=en",
                headers={
                    "Authorization": f"Bearer {token}",
                    "x-minsy-debug-trace": "1",
                    "x-minsy-debug-trace-id": trace_id,
                },
                json={"message": "hello"},
            )

    assert response.status_code == 200
    assert response.headers.get("x-minsy-debug-trace-id") == "frontend_trace_001"

    payloads = _parse_sse_payloads(response.text)
    types = [item.get("type") for item in payloads]
    assert "stream_start" in types
    assert "done" in types
