"""API tests for frontend strategy confirmation flow."""

from __future__ import annotations

import json
from copy import deepcopy
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient

from src.engine.strategy import EXAMPLE_PATH, load_strategy_payload
from src.main import app

_TURN1_RESPONSE = (
    "经验已记录。"
    '<AGENT_STATE_PATCH>{"trading_years_bucket":"years_5_plus"}</AGENT_STATE_PATCH>'
)

_TURN2_RESPONSE = (
    "风险偏好已记录。"
    '<AGENT_STATE_PATCH>{"risk_tolerance":"aggressive"}</AGENT_STATE_PATCH>'
)

_TURN3_RESPONSE = (
    "KYC 完成。"
    '<AGENT_STATE_PATCH>{"return_expectation":"high_growth"}</AGENT_STATE_PATCH>'
)

_TURN4_RESPONSE = (
    "策略范围完成。"
    '<AGENT_STATE_PATCH>{"target_market":"crypto","target_instrument":"BTCUSD",'
    '"opportunity_frequency_bucket":"daily","holding_period_bucket":"swing_days"}</AGENT_STATE_PATCH>'
)

_AUTO_STRESS_RESPONSE = (
    "Backtest started."
    '<AGENT_STATE_PATCH>{"backtest_job_id":"11111111-1111-1111-1111-111111111111",'
    '"backtest_status":"pending"}</AGENT_STATE_PATCH>'
)


def _register_and_get_token(client: TestClient) -> str:
    email = f"strategy_confirm_{uuid4().hex}@test.com"
    response = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "pass1234", "name": "Strategy Confirm User"},
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


def _make_mock_response_event(text: str, response_id: str) -> dict:
    return {
        "type": "response.completed",
        "response": {
            "id": response_id,
            "usage": {"input_tokens": 10, "output_tokens": 10},
        },
    }


def test_strategy_confirm_persists_and_auto_starts_backtest_turn_in_strategy_phase() -> None:
    mock_responses = [
        _TURN1_RESPONSE,
        _TURN2_RESPONSE,
        _TURN3_RESPONSE,
        _TURN4_RESPONSE,
        _AUTO_STRESS_RESPONSE,
    ]
    call_idx = {"i": 0}
    captured_tools: list[list[dict] | None] = []

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
        idx = call_idx["i"]
        call_idx["i"] += 1
        captured_tools.append(deepcopy(tools) if isinstance(tools, list) else None)
        text = mock_responses[idx]
        yield {"type": "response.output_text.delta", "delta": text, "sequence_number": 1}
        yield _make_mock_response_event(text, f"resp_strategy_confirm_{idx}")

    with patch(
        "src.services.openai_stream_service.OpenAIResponsesEventStreamer.stream_events",
        side_effect=_mock_stream_events,
    ):
        with TestClient(app) as client:
            token = _register_and_get_token(client)
            headers = {"Authorization": f"Bearer {token}"}

            turn1 = client.post(
                "/api/v1/chat/send-openai-stream",
                headers=headers,
                json={"message": "I have over 5 years of trading experience."},
            )
            assert turn1.status_code == 200
            done1 = next(
                p for p in _parse_sse_payloads(turn1.text) if p.get("type") == "done"
            )
            session_id = done1["session_id"]

            turn2 = client.post(
                "/api/v1/chat/send-openai-stream",
                headers=headers,
                json={"session_id": session_id, "message": "Risk aggressive."},
            )
            assert turn2.status_code == 200

            turn3 = client.post(
                "/api/v1/chat/send-openai-stream",
                headers=headers,
                json={"session_id": session_id, "message": "High growth."},
            )
            assert turn3.status_code == 200

            turn4 = client.post(
                "/api/v1/chat/send-openai-stream",
                headers=headers,
                json={
                    "session_id": session_id,
                    "message": "crypto + BTCUSD + daily + swing_days",
                },
            )
            assert turn4.status_code == 200
            done4 = next(
                p for p in _parse_sse_payloads(turn4.text) if p.get("type") == "done"
            )
            assert done4["phase"] == "strategy"

            strategy_payload = load_strategy_payload(EXAMPLE_PATH)
            confirm = client.post(
                "/api/v1/strategies/confirm",
                headers=headers,
                json={
                    "session_id": session_id,
                    "dsl_json": strategy_payload,
                    "auto_start_backtest": True,
                    "language": "en",
                },
            )
            assert confirm.status_code == 200
            body = confirm.json()

            assert body["strategy_id"]
            assert body["phase"] == "strategy"
            assert body["auto_started"] is True
            assert body["auto_done_payload"] is not None
            assert body["auto_done_payload"]["phase"] == "strategy"

            detail = client.get(f"/api/v1/sessions/{session_id}", headers=headers)
            assert detail.status_code == 200
            artifacts = detail.json()["artifacts"]
            assert artifacts["strategy"]["profile"]["strategy_id"] == body["strategy_id"]
            assert artifacts["stress_test"]["profile"]["strategy_id"] == body["strategy_id"]

    # Last call is auto backtest bootstrap turn in strategy phase:
    # strategy + backtest tools should be available together.
    assert captured_tools
    last_tools = captured_tools[-1]
    assert isinstance(last_tools, list) and last_tools
    labels = {
        item.get("server_label")
        for item in last_tools
        if isinstance(item, dict)
    }
    assert labels == {"strategy", "backtest"}


def test_get_strategy_detail_by_id() -> None:
    with patch(
        "src.services.openai_stream_service.OpenAIResponsesEventStreamer.stream_events",
        side_effect=Exception("stream not expected"),
    ):
        with TestClient(app) as client:
            token = _register_and_get_token(client)
            headers = {"Authorization": f"Bearer {token}"}

            create = client.post(
                "/api/v1/chat/new-thread",
                headers=headers,
                json={"metadata": {}},
            )
            assert create.status_code == 201
            session_id = create.json()["session_id"]

            strategy_payload = load_strategy_payload(EXAMPLE_PATH)
            confirm = client.post(
                "/api/v1/strategies/confirm",
                headers=headers,
                json={
                    "session_id": session_id,
                    "dsl_json": strategy_payload,
                    "auto_start_backtest": False,
                    "language": "en",
                },
            )
            assert confirm.status_code == 200
            strategy_id = confirm.json()["strategy_id"]

            detail = client.get(f"/api/v1/strategies/{strategy_id}", headers=headers)
            assert detail.status_code == 200
            body = detail.json()
            assert body["strategy_id"] == strategy_id
            assert body["session_id"] == session_id
            assert isinstance(body.get("dsl_json"), dict)
            assert body.get("dsl_json", {}).get("strategy")
