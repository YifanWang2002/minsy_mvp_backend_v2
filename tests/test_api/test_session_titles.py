"""API coverage for session title projection and streaming payloads."""

from __future__ import annotations

import json
from copy import deepcopy
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient

from src.engine.strategy import EXAMPLE_PATH, load_strategy_payload
from src.main import app


def _register_and_get_token(client: TestClient) -> str:
    email = f"session_titles_{uuid4().hex}@test.com"
    response = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "pass1234", "name": "Session Title User"},
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


def test_new_thread_and_session_endpoints_expose_session_title_fields() -> None:
    with TestClient(app) as client:
        token = _register_and_get_token(client)
        headers = {"Authorization": f"Bearer {token}"}

        create = client.post("/api/v1/chat/new-thread", headers=headers, json={})
        assert create.status_code == 201
        body = create.json()

        assert body["phase"] == "kyc"
        assert body["session_title"] == "KYC-In Progress"
        assert isinstance(body["session_title_record"], dict)
        assert body["session_title_record"]["kind"] == "kyc_in_progress"
        assert body["session_title_record"]["phase"] == "kyc"

        session_id = body["session_id"]
        listing = client.get("/api/v1/sessions", headers=headers)
        assert listing.status_code == 200
        rows = listing.json()
        assert len(rows) == 1
        assert rows[0]["session_id"] == session_id
        assert rows[0]["session_title"] == "KYC-In Progress"
        assert isinstance(rows[0]["session_title_record"], dict)
        assert rows[0]["session_title_record"]["kind"] == "kyc_in_progress"

        detail = client.get(f"/api/v1/sessions/{session_id}", headers=headers)
        assert detail.status_code == 200
        detail_body = detail.json()
        assert detail_body["session_title"] == "KYC-In Progress"
        assert isinstance(detail_body["session_title_record"], dict)
        assert detail_body["session_title_record"]["kind"] == "kyc_in_progress"


def test_done_event_contains_session_title_for_strategy_phase_session() -> None:
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
        del model, input_text, instructions, previous_response_id, tools, tool_choice, reasoning
        yield {
            "type": "response.output_text.delta",
            "delta": "继续优化策略。",
            "sequence_number": 1,
        }
        yield {
            "type": "response.completed",
            "response": {
                "id": "resp_session_titles_done",
                "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
            },
        }

    with patch(
        "src.services.openai_stream_service.OpenAIResponsesEventStreamer.stream_events",
        side_effect=_mock_stream_events,
    ):
        with TestClient(app) as client:
            token = _register_and_get_token(client)
            headers = {"Authorization": f"Bearer {token}"}

            create = client.post("/api/v1/chat/new-thread", headers=headers, json={})
            assert create.status_code == 201
            session_id = create.json()["session_id"]

            payload = deepcopy(load_strategy_payload(EXAMPLE_PATH))
            strategy_block = payload.get("strategy")
            if isinstance(strategy_block, dict):
                strategy_block["name"] = "SessionTitleStrategy"

            confirm = client.post(
                "/api/v1/strategies/confirm",
                headers=headers,
                json={
                    "session_id": session_id,
                    "dsl_json": payload,
                    "auto_start_backtest": False,
                    "language": "zh",
                },
            )
            assert confirm.status_code == 200

            stream = client.post(
                "/api/v1/chat/send-openai-stream?language=zh",
                headers=headers,
                json={
                    "session_id": session_id,
                    "message": "继续",
                },
            )
            assert stream.status_code == 200
            payloads = _parse_sse_payloads(stream.text)
            done = next(item for item in payloads if item.get("type") == "done")

            assert isinstance(done.get("session_title"), str)
            assert "SessionTitleStrategy" in done["session_title"]
            assert isinstance(done.get("session_title_record"), dict)
            assert done["session_title_record"]["kind"] == "strategy_named"
            assert done["session_title_record"]["phase"] == "strategy"
            assert done["session_title_record"]["strategy_name"] == "SessionTitleStrategy"
