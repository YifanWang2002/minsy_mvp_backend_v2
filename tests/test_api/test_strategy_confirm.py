"""API tests for frontend strategy confirmation flow."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from unittest.mock import patch
from uuid import UUID, uuid4

from fastapi.testclient import TestClient

from src.engine.strategy import EXAMPLE_PATH, load_strategy_payload
from src.engine.strategy.draft_store import StrategyDraftRecord
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


def test_strategy_confirm_persists_and_auto_starts_backtest_turn_in_strategy_phase() -> (
    None
):
    mock_responses = [
        _TURN1_RESPONSE,
        _TURN2_RESPONSE,
        _TURN3_RESPONSE,
        _TURN4_RESPONSE,
        _AUTO_STRESS_RESPONSE,
    ]
    call_idx = {"i": 0}
    captured_tools: list[list[dict] | None] = []
    captured_previous_response_ids: list[str | None] = []

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
        captured_previous_response_ids.append(previous_response_id)
        text = mock_responses[idx]
        yield {
            "type": "response.output_text.delta",
            "delta": text,
            "sequence_number": 1,
        }
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
                    "advance_to_stress_test": True,
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
            detail_body = detail.json()
            artifacts = detail_body["artifacts"]
            assert (
                artifacts["strategy"]["profile"]["strategy_id"] == body["strategy_id"]
            )
            assert (
                artifacts["stress_test"]["profile"]["strategy_id"]
                == body["strategy_id"]
            )
            assert (
                artifacts["deployment"]["profile"]["strategy_id"] == body["strategy_id"]
            )
            assert artifacts["deployment"]["profile"]["deployment_status"] == "ready"
            expected_market = (
                strategy_payload.get("universe", {}).get("market")
                if isinstance(strategy_payload.get("universe"), dict)
                else None
            )
            expected_tickers = (
                strategy_payload.get("universe", {}).get("tickers")
                if isinstance(strategy_payload.get("universe"), dict)
                else None
            )
            expected_timeframe = strategy_payload.get("timeframe")
            if isinstance(expected_market, str) and expected_market:
                assert (
                    artifacts["strategy"]["profile"].get("strategy_market")
                    == expected_market
                )
                assert (
                    artifacts["stress_test"]["profile"].get("strategy_market")
                    == expected_market
                )
            if isinstance(expected_tickers, list) and expected_tickers:
                assert (
                    artifacts["strategy"]["profile"].get("strategy_tickers")
                    == expected_tickers
                )
                assert (
                    artifacts["strategy"]["profile"].get("strategy_primary_symbol")
                    == expected_tickers[0]
                )
            if isinstance(expected_timeframe, str) and expected_timeframe:
                assert (
                    artifacts["strategy"]["profile"].get("strategy_timeframe")
                    == expected_timeframe
                )
            metadata = detail_body.get("metadata", {})
            assert metadata.get("advance_to_stress_test_ignored") is True
            assert isinstance(metadata.get("advance_to_stress_test_ignored_at"), str)
            assert metadata.get("strategy_id") == body["strategy_id"]
            if isinstance(expected_market, str) and expected_market:
                assert metadata.get("strategy_market") == expected_market
            if isinstance(expected_tickers, list) and expected_tickers:
                assert metadata.get("strategy_tickers") == expected_tickers
            if isinstance(expected_timeframe, str) and expected_timeframe:
                assert metadata.get("strategy_timeframe") == expected_timeframe
            assert metadata.get("deployment_status") == "ready"

    # Last call happens in strategy phase and should use strategy artifact-ops tools.
    assert captured_tools
    assert captured_previous_response_ids
    assert captured_previous_response_ids[-1] is None
    last_tools = captured_tools[-1]
    assert isinstance(last_tools, list) and last_tools
    labels = {item.get("server_label") for item in last_tools if isinstance(item, dict)}
    assert labels == {"strategy", "market_data", "backtest"}


def test_chat_can_advance_strategy_to_deployment_without_confirm_endpoint() -> None:
    strategy_id = str(uuid4())
    mock_responses = [
        _TURN1_RESPONSE,
        _TURN2_RESPONSE,
        _TURN3_RESPONSE,
        _TURN4_RESPONSE,
        (
            "策略已确认并可进入部署。"
            f'<AGENT_STATE_PATCH>{{"strategy_id":"{strategy_id}","strategy_confirmed":true}}</AGENT_STATE_PATCH>'
        ),
    ]
    call_idx = {"i": 0}

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
        text = mock_responses[idx]
        yield {
            "type": "response.output_text.delta",
            "delta": text,
            "sequence_number": 1,
        }
        yield _make_mock_response_event(text, f"resp_chat_confirm_{idx}")

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

            turn5 = client.post(
                "/api/v1/chat/send-openai-stream",
                headers=headers,
                json={
                    "session_id": session_id,
                    "message": "I confirm this strategy is finalized and ready to deploy.",
                },
            )
            assert turn5.status_code == 200
            payloads5 = _parse_sse_payloads(turn5.text)
            done5 = next(p for p in payloads5 if p.get("type") == "done")
            assert done5["phase"] == "deployment"
            assert any(
                item.get("type") == "phase_change"
                and item.get("from_phase") == "strategy"
                and item.get("to_phase") == "deployment"
                for item in payloads5
            )

            detail = client.get(f"/api/v1/sessions/{session_id}", headers=headers)
            assert detail.status_code == 200
            artifacts = detail.json()["artifacts"]
            assert artifacts["strategy"]["profile"]["strategy_id"] == strategy_id
            assert artifacts["strategy"]["profile"]["strategy_confirmed"] is True
            assert artifacts["deployment"]["profile"]["strategy_id"] == strategy_id
            assert artifacts["deployment"]["profile"]["deployment_status"] == "ready"


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


def test_get_strategy_draft_detail_by_id() -> None:
    with patch(
        "src.services.openai_stream_service.OpenAIResponsesEventStreamer.stream_events",
        side_effect=Exception("stream not expected"),
    ):
        with TestClient(app) as client:
            token = _register_and_get_token(client)
            headers = {"Authorization": f"Bearer {token}"}

            me_resp = client.get("/api/v1/auth/me", headers=headers)
            assert me_resp.status_code == 200
            user_id = me_resp.json().get("user_id")
            assert isinstance(user_id, str) and user_id

            create = client.post(
                "/api/v1/chat/new-thread",
                headers=headers,
                json={"metadata": {}},
            )
            assert create.status_code == 201
            session_id = create.json()["session_id"]

            strategy_payload = load_strategy_payload(EXAMPLE_PATH)
            strategy_draft_id = uuid4()
            now = datetime.now(UTC)
            draft = StrategyDraftRecord(
                strategy_draft_id=strategy_draft_id,
                user_id=UUID(user_id),
                session_id=UUID(session_id),
                dsl_json=strategy_payload,
                payload_hash="abc123",
                created_at=now,
                expires_at=now + timedelta(hours=1),
                ttl_seconds=3600,
            )

            async def _mock_get_strategy_draft(_strategy_draft_id: UUID):
                if _strategy_draft_id != strategy_draft_id:
                    return None
                return draft

            with patch(
                "src.api.routers.strategies.get_strategy_draft",
                side_effect=_mock_get_strategy_draft,
            ):
                detail = client.get(
                    f"/api/v1/strategies/drafts/{strategy_draft_id}",
                    headers=headers,
                )

            assert detail.status_code == 200
            body = detail.json()
            assert body["strategy_draft_id"] == str(strategy_draft_id)
            assert body["session_id"] == session_id
            assert isinstance(body.get("dsl_json"), dict)
