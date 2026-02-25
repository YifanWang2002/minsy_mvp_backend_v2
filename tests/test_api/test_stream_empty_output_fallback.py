"""Regression tests for stream turns with missing text deltas."""

from __future__ import annotations

import json
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient

from src.main import app


def _register_and_get_token(client: TestClient) -> str:
    email = f"empty_stream_{uuid4().hex}@test.com"
    response = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "pass1234", "name": "Stream Empty User"},
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


def _joined_text_delta(payloads: list[dict]) -> str:
    return "".join(
        item.get("delta", "")
        for item in payloads
        if item.get("type") == "text_delta"
    )


def test_stream_falls_back_to_output_text_done_when_no_delta() -> None:
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
            "type": "response.output_text.done",
            "sequence_number": 1,
            "text": "Please tell me your trading experience bucket.",
        }
        yield {
            "type": "response.completed",
            "response": {
                "id": "resp_done_only_text",
                "usage": {"input_tokens": 11, "output_tokens": 6},
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
                json={"message": "hi"},
            )

            assert response.status_code == 200
            payloads = _parse_sse_payloads(response.text)
            text = _joined_text_delta(payloads)
            assert "trading experience bucket" in text

            done = next(item for item in payloads if item.get("type") == "done")
            usage = done.get("usage") or {}
            assert usage["input_tokens"] == 11
            assert usage["output_tokens"] == 6
            assert usage["total_tokens"] == 17
            assert usage["model"]
            assert "cost_usd" in usage
            session_id = done["session_id"]

            detail = client.get(
                f"/api/v1/sessions/{session_id}",
                headers={"Authorization": f"Bearer {token}"},
            )
            assert detail.status_code == 200
            body = detail.json()
            metadata = body["metadata"]
            openai_cost = metadata.get("openai_cost") or {}
            assert openai_cost["totals"]["input_tokens"] == 11
            assert openai_cost["totals"]["output_tokens"] == 6
            assert openai_cost["totals"]["total_tokens"] == 17
            assert openai_cost["totals"]["turn_count"] == 1

            messages = body["messages"]
            assistant_messages = [
                item for item in messages if item.get("role") == "assistant"
            ]
            assert assistant_messages
            assert "trading experience bucket" in assistant_messages[-1]["content"]
            assert assistant_messages[-1]["token_usage"]["total_tokens"] == 17


def test_stream_empty_output_still_emits_non_empty_fallback_text() -> None:
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
            "type": "response.output_text.done",
            "sequence_number": 1,
            "text": "",
        }
        yield {
            "type": "response.completed",
            "response": {
                "id": "resp_empty_text",
                "usage": {"input_tokens": 9, "output_tokens": 5},
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
                json={"message": "hi"},
            )

            assert response.status_code == 200
            payloads = _parse_sse_payloads(response.text)
            text = _joined_text_delta(payloads)
            assert text.strip()
            assert "displayable reply" in text

            done = next(item for item in payloads if item.get("type") == "done")
            usage = done.get("usage") or {}
            assert usage["input_tokens"] == 9
            assert usage["output_tokens"] == 5
            assert usage["total_tokens"] == 14
            assert usage["model"]
            assert "cost_usd" in usage
            session_id = done["session_id"]
            assert "trading_years_bucket" in (done.get("missing_fields") or [])

            detail = client.get(
                f"/api/v1/sessions/{session_id}",
                headers={"Authorization": f"Bearer {token}"},
            )
            assert detail.status_code == 200
            body = detail.json()
            metadata = body["metadata"]
            openai_cost = metadata.get("openai_cost") or {}
            assert openai_cost["totals"]["input_tokens"] == 9
            assert openai_cost["totals"]["output_tokens"] == 5
            assert openai_cost["totals"]["total_tokens"] == 14
            assert openai_cost["totals"]["turn_count"] == 1

            messages = body["messages"]
            assistant_messages = [
                item for item in messages if item.get("role") == "assistant"
            ]
            assert assistant_messages
            assert assistant_messages[-1]["content"].strip()
            assert "displayable reply" in assistant_messages[-1]["content"]
            assert assistant_messages[-1]["token_usage"]["total_tokens"] == 14
