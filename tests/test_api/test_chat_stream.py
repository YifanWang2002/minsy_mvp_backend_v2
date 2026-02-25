"""Integration tests for OpenAI streaming endpoints (real API calls)."""

from __future__ import annotations

import json
from uuid import uuid4

from fastapi.testclient import TestClient

from src.main import app


def _register_and_get_token(client: TestClient) -> str:
    email = f"stream_{uuid4().hex}@test.com"
    response = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "pass1234", "name": "Stream User"},
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


def test_openai_stream_emits_text_and_done() -> None:
    with TestClient(app) as client:
        token = _register_and_get_token(client)
        response = client.post(
            "/api/v1/chat/send-openai-stream?language=zh",
            headers={"Authorization": f"Bearer {token}"},
            json={"message": "我只有一个信息：我有5年交易经验。请继续问我剩余问题，不要猜测。"},
        )

    assert response.status_code == 200
    payloads = _parse_sse_payloads(response.text)

    types = [item["type"] for item in payloads]
    assert "stream_start" in types
    assert "text_delta" in types
    assert "done" in types

    done = next(item for item in payloads if item["type"] == "done")
    assert done["phase"] in {"kyc", "pre_strategy", "strategy"}


def test_openai_stream_forwards_openai_events() -> None:
    with TestClient(app) as client:
        token = _register_and_get_token(client)
        response = client.post(
            "/api/v1/chat/send-openai-stream?language=en",
            headers={"Authorization": f"Bearer {token}"},
            json={"message": "Summarize in one sentence what this platform does."},
        )

    assert response.status_code == 200
    payloads = _parse_sse_payloads(response.text)

    openai_events = [item for item in payloads if item.get("type") == "openai_event"]
    assert openai_events, "No proxied OpenAI events received"

    openai_types = [item.get("openai_type") for item in openai_events]
    assert "response.created" in openai_types
    assert "response.in_progress" in openai_types
    assert "response.completed" in openai_types

    done = next(item for item in payloads if item.get("type") == "done")
    assert "usage" in done
    usage = done.get("usage")
    if isinstance(usage, dict) and usage:
        assert "model" in usage
        assert "input_tokens" in usage
        assert "output_tokens" in usage
        assert "total_tokens" in usage
        assert "cost_usd" in usage
    if done.get("session_openai_cost") is not None:
        assert "total_tokens" in done["session_openai_cost"]

    text = "".join(item.get("delta", "") for item in payloads if item.get("type") == "text_delta")
    assert text.strip(), "No text response content"
