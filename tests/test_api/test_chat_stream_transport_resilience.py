"""Transport-level resilience tests for chat SSE streaming."""

from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient

from src.main import app


def _register_and_get_token(client: TestClient) -> str:
    email = f"stream_transport_{uuid4().hex}@test.com"
    response = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "pass1234", "name": "Stream Transport User"},
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


def _coerce_line_text(line: str | bytes) -> str:
    if isinstance(line, bytes):
        return line.decode("utf-8", errors="replace")
    return line


def test_chat_stream_emits_keepalive_comment_when_upstream_is_temporarily_idle() -> None:
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
        await asyncio.sleep(0.08)
        yield {
            "type": "response.output_text.delta",
            "delta": "hello",
            "sequence_number": 1,
        }
        yield {
            "type": "response.completed",
            "response": {
                "id": f"resp_{uuid4().hex}",
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            },
        }

    with patch(
        "src.services.openai_stream_service.OpenAIResponsesEventStreamer.stream_events",
        side_effect=_mock_stream_events,
    ):
        with patch("src.api.routers.chat._CHAT_SSE_HEARTBEAT_SECONDS", 0.02):
            with TestClient(app) as client:
                token = _register_and_get_token(client)
                response = client.post(
                    "/api/v1/chat/send-openai-stream?language=en",
                    headers={"Authorization": f"Bearer {token}"},
                    json={"message": "heartbeat please"},
                )

    assert response.status_code == 200
    assert ": keepalive" in response.text
    payloads = _parse_sse_payloads(response.text)
    assert any(item.get("type") == "done" for item in payloads)


def test_chat_stream_continues_to_persist_assistant_message_after_client_disconnect() -> None:
    expected_text = "drain-one drain-two"

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
        await asyncio.sleep(0.05)
        yield {
            "type": "response.output_text.delta",
            "delta": "drain-one ",
            "sequence_number": 1,
        }
        await asyncio.sleep(0.05)
        yield {
            "type": "response.output_text.delta",
            "delta": "drain-two",
            "sequence_number": 2,
        }
        await asyncio.sleep(0.05)
        yield {
            "type": "response.completed",
            "response": {
                "id": f"resp_{uuid4().hex}",
                "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
            },
        }

    with patch(
        "src.services.openai_stream_service.OpenAIResponsesEventStreamer.stream_events",
        side_effect=_mock_stream_events,
    ):
        with patch("src.api.routers.chat._CHAT_SSE_HEARTBEAT_SECONDS", 0.02):
            with TestClient(app) as client:
                token = _register_and_get_token(client)
                session_id: str | None = None

                with client.stream(
                    "POST",
                    "/api/v1/chat/send-openai-stream?language=en",
                    headers={"Authorization": f"Bearer {token}"},
                    json={"message": "disconnect and continue"},
                ) as response:
                    assert response.status_code == 200
                    for raw_line in response.iter_lines():
                        line = _coerce_line_text(raw_line)
                        if not line.startswith("data: "):
                            continue
                        payload = json.loads(line.removeprefix("data: "))
                        if payload.get("type") == "stream_start":
                            value = payload.get("session_id")
                            if isinstance(value, str) and value.strip():
                                session_id = value.strip()
                            break

                assert session_id is not None

                recovered_text: str | None = None
                for _ in range(25):
                    detail = client.get(
                        f"/api/v1/sessions/{session_id}",
                        headers={"Authorization": f"Bearer {token}"},
                    )
                    assert detail.status_code == 200
                    messages = detail.json().get("messages", [])
                    assistant_messages = [
                        item
                        for item in messages
                        if isinstance(item, dict) and item.get("role") == "assistant"
                    ]
                    if assistant_messages:
                        latest = assistant_messages[-1].get("content")
                        if isinstance(latest, str) and expected_text in latest:
                            recovered_text = latest
                            break
                    time.sleep(0.12)

    assert isinstance(recovered_text, str)
    assert expected_text in recovered_text
