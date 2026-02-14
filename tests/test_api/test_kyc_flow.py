"""Unit tests for KYC flow – mocks OpenAI Responses API so no real API calls needed."""

from __future__ import annotations

import json
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient

from src.main import app

# Simulated AI responses with AGENT_STATE_PATCH blocks
_TURN1_RESPONSE = (
    "Got it, you have 5+ years of experience! "
    "What is your risk tolerance?"
    '<AGENT_STATE_PATCH>{"trading_years_bucket":"years_5_plus"}</AGENT_STATE_PATCH>'
    '<AGENT_UI_JSON>{"type":"choice_prompt","choice_id":"kyc_risk_tolerance",'
    '"question":"What is your risk tolerance?",'
    '"options":[{"id":"conservative","label":"Conservative"},'
    '{"id":"moderate","label":"Moderate"},'
    '{"id":"aggressive","label":"Aggressive"},'
    '{"id":"very_aggressive","label":"Very aggressive"}]}</AGENT_UI_JSON>'
)

_TURN2_RESPONSE = (
    "Aggressive, noted! What return expectation do you target?"
    '<AGENT_STATE_PATCH>{"trading_years_bucket":"years_5_plus","risk_tolerance":"aggressive"}</AGENT_STATE_PATCH>'
    '<AGENT_UI_JSON>{"type":"choice_prompt","choice_id":"kyc_return_expectation",'
    '"question":"What return expectation do you target?",'
    '"options":[{"id":"capital_preservation","label":"Capital preservation"},'
    '{"id":"balanced_growth","label":"Balanced growth"},'
    '{"id":"growth","label":"Growth"},'
    '{"id":"high_growth","label":"High growth"}]}</AGENT_UI_JSON>'
)

_TURN3_RESPONSE = (
    "Your KYC is complete! Summary: 5+ years, aggressive, high growth."
    '<AGENT_STATE_PATCH>{"trading_years_bucket":"years_5_plus","risk_tolerance":"aggressive","return_expectation":"high_growth"}</AGENT_STATE_PATCH>'
)

_TURN4_PRE_STRATEGY_RESPONSE = (
    "Great, your KYC is complete. Which market do you want to trade?"
    '<AGENT_UI_JSON>{"type":"choice_prompt","choice_id":"target_market",'
    '"question":"Which market do you want to trade?",'
    '"options":[{"id":"us_stocks","label":"US Stocks"},'
    '{"id":"crypto","label":"Crypto"},'
    '{"id":"forex","label":"Forex"},'
    '{"id":"futures","label":"Futures"}]}</AGENT_UI_JSON>'
)

_TURN5_PRE_STRATEGY_RESPONSE = (
    "US equities noted. Which instrument do you want to focus on?"
    '<AGENT_STATE_PATCH>{"target_market":"us_stocks"}</AGENT_STATE_PATCH>'
    '<AGENT_UI_JSON>{"type":"choice_prompt","choice_id":"target_instrument",'
    '"question":"Which instrument do you want to focus on?",'
    '"options":[{"id":"SPY","label":"SPY"},'
    '{"id":"QQQ","label":"QQQ"},'
    '{"id":"AAPL","label":"AAPL"},'
    '{"id":"NVDA","label":"NVDA"}]}</AGENT_UI_JSON>'
)

_TURN6_PRE_STRATEGY_RESPONSE = (
    "SPY noted. What opportunity frequency do you expect?"
    '<AGENT_STATE_PATCH>{"target_market":"us_stocks","target_instrument":"SPY"}</AGENT_STATE_PATCH>'
    '<AGENT_UI_JSON>{"type":"choice_prompt","choice_id":"opportunity_frequency_bucket",'
    '"question":"What opportunity frequency do you expect?",'
    '"options":[{"id":"few_per_month","label":"Few per month"},'
    '{"id":"few_per_week","label":"Few per week"},'
    '{"id":"daily","label":"Daily"},'
    '{"id":"multiple_per_day","label":"Multiple per day"}]}</AGENT_UI_JSON>'
)

_TURN7_PRE_STRATEGY_RESPONSE = (
    "Daily opportunities noted. What holding-period style do you prefer?"
    '<AGENT_STATE_PATCH>{"target_market":"us_stocks","target_instrument":"SPY","opportunity_frequency_bucket":"daily"}</AGENT_STATE_PATCH>'
    '<AGENT_UI_JSON>{"type":"choice_prompt","choice_id":"holding_period_bucket",'
    '"question":"What holding-period style do you prefer?",'
    '"options":[{"id":"intraday_scalp","label":"Scalp intraday"},'
    '{"id":"intraday","label":"Intraday"},'
    '{"id":"swing_days","label":"Swing (days)"},'
    '{"id":"position_weeks_plus","label":"Position (weeks+)"}]}</AGENT_UI_JSON>'
)

_TURN8_PRE_STRATEGY_RESPONSE = (
    "Thanks! Pre-strategy setup is complete."
    '<AGENT_STATE_PATCH>{"target_market":"us_stocks","target_instrument":"SPY","opportunity_frequency_bucket":"daily","holding_period_bucket":"swing_days"}</AGENT_STATE_PATCH>'
)

_MOCK_RESPONSES = [
    _TURN1_RESPONSE,
    _TURN2_RESPONSE,
    _TURN3_RESPONSE,
    _TURN4_PRE_STRATEGY_RESPONSE,
    _TURN5_PRE_STRATEGY_RESPONSE,
    _TURN6_PRE_STRATEGY_RESPONSE,
    _TURN7_PRE_STRATEGY_RESPONSE,
    _TURN8_PRE_STRATEGY_RESPONSE,
]


def _register_and_get_token(client: TestClient) -> str:
    email = f"kyc_api_{uuid4().hex}@test.com"
    resp = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "pass1234", "name": "KYC User"},
    )
    assert resp.status_code == 201
    return resp.json()["access_token"]


def _parse_sse_payloads(raw_text: str) -> list[dict]:
    payloads: list[dict] = []
    blocks = [block.strip() for block in raw_text.split("\n\n") if block.strip()]
    for block in blocks:
        for line in block.splitlines():
            if line.startswith("data: "):
                payloads.append(json.loads(line.removeprefix("data: ")))
    return payloads


def _make_mock_response_event(text: str, response_id: str) -> dict:
    """Build a fake response.completed event for the mock."""
    return {
        "type": "response.completed",
        "response": {
            "id": response_id,
            "usage": {"input_tokens": 100, "output_tokens": 50},
        },
    }


def test_new_thread_creates_kyc_session() -> None:
    with TestClient(app) as client:
        token = _register_and_get_token(client)
        response = client.post(
            "/api/v1/chat/new-thread",
            json={},
            headers={"Authorization": f"Bearer {token}"},
        )

    assert response.status_code == 201
    body = response.json()
    assert body["session_id"]
    assert body["phase"] == "kyc"
    assert body["status"] == "active"
    assert body["kyc_status"] == "incomplete"


def test_kyc_flow_completes_and_updates_user_profile() -> None:
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
        """Yield fake SSE events mimicking the Responses API."""
        idx = call_idx["i"]
        call_idx["i"] += 1
        text = _MOCK_RESPONSES[idx] if idx < len(_MOCK_RESPONSES) else "done"

        # Simulate text delta events
        yield {"type": "response.output_text.delta", "delta": text, "sequence_number": 1}
        # Simulate response.completed event
        yield _make_mock_response_event(text, f"resp_mock_{idx}")

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
            payloads1 = _parse_sse_payloads(turn1.text)
            done1 = next(p for p in payloads1 if p.get("type") == "done")
            session_id = done1["session_id"]
            assert done1["phase"] == "kyc"
            assert done1["missing_fields"]

            turn2 = client.post(
                "/api/v1/chat/send-openai-stream",
                headers=headers,
                json={
                    "session_id": session_id,
                    "message": "My risk tolerance is aggressive.",
                },
            )
            assert turn2.status_code == 200

            turn3 = client.post(
                "/api/v1/chat/send-openai-stream",
                headers=headers,
                json={
                    "session_id": session_id,
                    "message": "I target high growth returns.",
                },
            )
            assert turn3.status_code == 200
            payloads3 = _parse_sse_payloads(turn3.text)
            done3 = next(p for p in payloads3 if p.get("type") == "done")
            assert done3["phase"] == "pre_strategy"
            assert done3["kyc_status"] == "complete"
            assert done3["missing_fields"] == []

            me = client.get("/api/v1/auth/me", headers=headers)
            assert me.status_code == 200
            assert me.json()["kyc_status"] == "complete"

            detail = client.get(f"/api/v1/sessions/{session_id}", headers=headers)
            assert detail.status_code == 200
            detail_body = detail.json()
            assert detail_body["current_phase"] == "pre_strategy"
            assert len(detail_body["messages"]) >= 6
            assert any(
                isinstance(message.get("tool_calls"), list) and message.get("tool_calls")
                for message in detail_body["messages"]
                if message.get("role") == "assistant"
            )


def test_stream_failure_is_gracefully_closed_with_done_event() -> None:
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
        if False:
            yield {}
        raise RuntimeError("peer closed connection without sending complete message body")

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
    assert any(item.get("type") == "stream_start" for item in payloads)
    assert any(item.get("type") == "done" for item in payloads)

    done = next(item for item in payloads if item.get("type") == "done")
    assert isinstance(done.get("stream_error"), str)
    assert done.get("stream_error")

    text = "".join(item.get("delta", "") for item in payloads if item.get("type") == "text_delta")
    assert "interrupted" in text.lower()


def test_pre_strategy_phase_uses_pre_strategy_instructions() -> None:
    call_idx = {"i": 0}
    captured_instructions: list[str | None] = []
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
        captured_instructions.append(instructions)
        captured_tools.append(tools)
        text = _MOCK_RESPONSES[idx] if idx < len(_MOCK_RESPONSES) else "done"
        yield {"type": "response.output_text.delta", "delta": text, "sequence_number": 1}
        yield _make_mock_response_event(text, f"resp_mock_{idx}")

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
            payloads1 = _parse_sse_payloads(turn1.text)
            done1 = next(p for p in payloads1 if p.get("type") == "done")
            session_id = done1["session_id"]

            turn2 = client.post(
                "/api/v1/chat/send-openai-stream",
                headers=headers,
                json={
                    "session_id": session_id,
                    "message": "My risk tolerance is aggressive.",
                },
            )
            assert turn2.status_code == 200

            turn3 = client.post(
                "/api/v1/chat/send-openai-stream",
                headers=headers,
                json={
                    "session_id": session_id,
                    "message": "I target high growth returns.",
                },
            )
            assert turn3.status_code == 200
            payloads3 = _parse_sse_payloads(turn3.text)
            done3 = next(p for p in payloads3 if p.get("type") == "done")
            assert done3["phase"] == "pre_strategy"

            turn4 = client.post(
                "/api/v1/chat/send-openai-stream",
                headers=headers,
                json={
                    "session_id": session_id,
                    "message": "I want to trade US equities on 1h timeframe.",
                },
            )
            assert turn4.status_code == 200
            payloads4 = _parse_sse_payloads(turn4.text)
            done4 = next(p for p in payloads4 if p.get("type") == "done")
            assert done4["phase"] == "pre_strategy"

    assert len(captured_instructions) >= 4
    assert captured_instructions[2] is not None
    assert captured_instructions[3] is not None
    assert "Minsy KYC Agent" in captured_instructions[2]
    assert "Minsy Pre-Strategy Agent" in captured_instructions[3]
    assert "TradingView Knowledge" in captured_instructions[3]
    # Non-symbol turn should not inject market-data MCP tools.
    assert captured_tools[3] is None


def test_new_thread_starts_pre_strategy_when_user_kyc_completed() -> None:
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
        text = _MOCK_RESPONSES[idx]
        yield {"type": "response.output_text.delta", "delta": text, "sequence_number": 1}
        yield _make_mock_response_event(text, f"resp_mock_{idx}")

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
            sid = next(
                p for p in _parse_sse_payloads(turn1.text) if p.get("type") == "done"
            )["session_id"]

            turn2 = client.post(
                "/api/v1/chat/send-openai-stream",
                headers=headers,
                json={"session_id": sid, "message": "My risk tolerance is aggressive."},
            )
            assert turn2.status_code == 200
            turn3 = client.post(
                "/api/v1/chat/send-openai-stream",
                headers=headers,
                json={"session_id": sid, "message": "I target high growth returns."},
            )
            assert turn3.status_code == 200

            new_thread = client.post(
                "/api/v1/chat/new-thread",
                json={},
                headers=headers,
            )
            assert new_thread.status_code == 201
            assert new_thread.json()["phase"] == "pre_strategy"


def test_pre_strategy_flow_completes_and_transitions_to_strategy() -> None:
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
        text = _MOCK_RESPONSES[idx]
        yield {"type": "response.output_text.delta", "delta": text, "sequence_number": 1}
        yield _make_mock_response_event(text, f"resp_mock_{idx}")

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
            sid = next(
                p for p in _parse_sse_payloads(turn1.text) if p.get("type") == "done"
            )["session_id"]

            turn2 = client.post(
                "/api/v1/chat/send-openai-stream",
                headers=headers,
                json={"session_id": sid, "message": "My risk tolerance is aggressive."},
            )
            assert turn2.status_code == 200
            turn3 = client.post(
                "/api/v1/chat/send-openai-stream",
                headers=headers,
                json={"session_id": sid, "message": "I target high growth returns."},
            )
            assert turn3.status_code == 200

            turn4 = client.post(
                "/api/v1/chat/send-openai-stream",
                headers=headers,
                json={"session_id": sid, "message": "US equities"},
            )
            assert turn4.status_code == 200

            turn5 = client.post(
                "/api/v1/chat/send-openai-stream",
                headers=headers,
                json={"session_id": sid, "message": "SPY"},
            )
            assert turn5.status_code == 200

            turn6 = client.post(
                "/api/v1/chat/send-openai-stream",
                headers=headers,
                json={"session_id": sid, "message": "Daily"},
            )
            assert turn6.status_code == 200

            turn7 = client.post(
                "/api/v1/chat/send-openai-stream",
                headers=headers,
                json={"session_id": sid, "message": "Swing days"},
            )
            assert turn7.status_code == 200
            done7 = next(
                p for p in _parse_sse_payloads(turn7.text) if p.get("type") == "done"
            )
            assert done7["phase"] == "pre_strategy"

            turn8 = client.post(
                "/api/v1/chat/send-openai-stream",
                headers=headers,
                json={"session_id": sid, "message": "Swing days"},
            )
            assert turn8.status_code == 200
            done8 = next(
                p for p in _parse_sse_payloads(turn8.text) if p.get("type") == "done"
            )
            assert done8["phase"] == "strategy"
            assert done8["missing_fields"] == []

            detail = client.get(f"/api/v1/sessions/{sid}", headers=headers)
            assert detail.status_code == 200
            artifacts = detail.json()["artifacts"]
            pre_strategy_profile = artifacts["pre_strategy"]["profile"]
            assert pre_strategy_profile["target_market"] == "us_stocks"
            assert pre_strategy_profile["target_instrument"] == "SPY"
            assert pre_strategy_profile["opportunity_frequency_bucket"] == "daily"
            assert pre_strategy_profile["holding_period_bucket"] == "swing_days"


def test_pre_strategy_instrument_options_are_filtered_by_selected_market() -> None:
    responses = [
        _TURN1_RESPONSE,
        _TURN2_RESPONSE,
        _TURN3_RESPONSE,
        (
            "Crypto market noted. Please pick your symbol."
            '<AGENT_STATE_PATCH>{"target_market":"crypto"}</AGENT_STATE_PATCH>'
            '<AGENT_UI_JSON>{"type":"choice_prompt","choice_id":"target_instrument",'
            '"question":"Which symbol do you want to trade?",'
            '"options":[{"id":"AAPL","label":"AAPL"},'
            '{"id":"NVDA","label":"NVDA"},'
            '{"id":"BTCUSD","label":"BTC/USD"},'
            '{"id":"ETHUSD","label":"ETH/USD"}]}</AGENT_UI_JSON>'
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
        text = responses[idx]
        yield {"type": "response.output_text.delta", "delta": text, "sequence_number": 1}
        yield _make_mock_response_event(text, f"resp_mock_{idx}")

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
            sid = next(
                p for p in _parse_sse_payloads(turn1.text) if p.get("type") == "done"
            )["session_id"]

            turn2 = client.post(
                "/api/v1/chat/send-openai-stream",
                headers=headers,
                json={"session_id": sid, "message": "My risk tolerance is aggressive."},
            )
            assert turn2.status_code == 200
            turn3 = client.post(
                "/api/v1/chat/send-openai-stream",
                headers=headers,
                json={"session_id": sid, "message": "I target high growth returns."},
            )
            assert turn3.status_code == 200

            turn4 = client.post(
                "/api/v1/chat/send-openai-stream",
                headers=headers,
                json={"session_id": sid, "message": "crypto"},
            )
            assert turn4.status_code == 200
            payloads4 = _parse_sse_payloads(turn4.text)

            genui_event = next(p for p in payloads4 if p.get("type") == "genui")
            genui_payload = genui_event["payload"]
            option_ids = [item["id"] for item in genui_payload["options"]]

            assert genui_payload["choice_id"] == "target_instrument"
            assert "AAPL" not in option_ids
            assert "NVDA" not in option_ids
            assert set(option_ids).issubset({"BTCUSD", "ETHUSD"})


def test_pre_strategy_can_emit_chart_and_choice_in_same_turn() -> None:
    responses = [
        _TURN1_RESPONSE,
        _TURN2_RESPONSE,
        _TURN3_RESPONSE,
        _TURN4_PRE_STRATEGY_RESPONSE,
        (
            "SPY is available and currently trading around 500 USD. "
            "What opportunity frequency do you expect?"
            '<AGENT_STATE_PATCH>{"target_market":"us_stocks","target_instrument":"SPY"}</AGENT_STATE_PATCH>'
            '<AGENT_UI_JSON>{"type":"tradingview_chart","symbol":"SPY","interval":"D"}</AGENT_UI_JSON>'
            '<AGENT_UI_JSON>{"type":"choice_prompt","choice_id":"opportunity_frequency_bucket",'
            '"question":"What opportunity frequency do you expect?",'
            '"options":[{"id":"few_per_month","label":"Few per month"},'
            '{"id":"few_per_week","label":"Few per week"},'
            '{"id":"daily","label":"Daily"},'
            '{"id":"multiple_per_day","label":"Multiple per day"}]}</AGENT_UI_JSON>'
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
        text = responses[idx]
        yield {"type": "response.output_text.delta", "delta": text, "sequence_number": 1}
        yield _make_mock_response_event(text, f"resp_mock_{idx}")

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
            sid = next(
                p for p in _parse_sse_payloads(turn1.text) if p.get("type") == "done"
            )["session_id"]

            turn2 = client.post(
                "/api/v1/chat/send-openai-stream",
                headers=headers,
                json={"session_id": sid, "message": "My risk tolerance is aggressive."},
            )
            assert turn2.status_code == 200

            turn3 = client.post(
                "/api/v1/chat/send-openai-stream",
                headers=headers,
                json={"session_id": sid, "message": "I target high growth returns."},
            )
            assert turn3.status_code == 200

            turn4 = client.post(
                "/api/v1/chat/send-openai-stream",
                headers=headers,
                json={"session_id": sid, "message": "US equities"},
            )
            assert turn4.status_code == 200

            turn5 = client.post(
                "/api/v1/chat/send-openai-stream",
                headers=headers,
                json={"session_id": sid, "message": "SPY"},
            )
            assert turn5.status_code == 200
            payloads5 = _parse_sse_payloads(turn5.text)

            genui_payloads = [
                item["payload"]
                for item in payloads5
                if item.get("type") == "genui" and isinstance(item.get("payload"), dict)
            ]
            assert len(genui_payloads) == 2
            assert any(item.get("type") == "tradingview_chart" for item in genui_payloads)
            assert any(
                item.get("type") == "choice_prompt"
                and item.get("choice_id") == "opportunity_frequency_bucket"
                for item in genui_payloads
            )

            detail = client.get(f"/api/v1/sessions/{sid}", headers=headers)
            assert detail.status_code == 200
            assistant_messages = [
                message
                for message in detail.json()["messages"]
                if message.get("role") == "assistant"
            ]
            assert assistant_messages
            last_tool_calls = assistant_messages[-1].get("tool_calls") or []
            assert any(
                isinstance(item, dict) and item.get("type") == "tradingview_chart"
                for item in last_tool_calls
            )
            assert any(
                isinstance(item, dict)
                and item.get("type") == "choice_prompt"
                and item.get("choice_id") == "opportunity_frequency_bucket"
                for item in last_tool_calls
            )


def test_transition_emits_phase_change_without_entry_guidance() -> None:
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
        text = _MOCK_RESPONSES[idx]
        yield {"type": "response.output_text.delta", "delta": text, "sequence_number": 1}
        yield _make_mock_response_event(text, f"resp_mock_{idx}")

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
            sid = next(
                p for p in _parse_sse_payloads(turn1.text) if p.get("type") == "done"
            )["session_id"]

            client.post(
                "/api/v1/chat/send-openai-stream",
                headers=headers,
                json={"session_id": sid, "message": "My risk tolerance is aggressive."},
            )

            turn3 = client.post(
                "/api/v1/chat/send-openai-stream",
                headers=headers,
                json={"session_id": sid, "message": "I target high growth returns."},
            )
            assert turn3.status_code == 200
            payloads3 = _parse_sse_payloads(turn3.text)
            full_text = "".join(
                payload.get("delta", "")
                for payload in payloads3
                if payload.get("type") == "text_delta"
            )
            assert "define your strategy scope" not in full_text.lower()

            phase_changes = [p for p in payloads3 if p.get("type") == "phase_change"]
            assert phase_changes
            assert phase_changes[-1].get("from_phase") == "kyc"
            assert phase_changes[-1].get("to_phase") == "pre_strategy"

            done_payload = next(p for p in payloads3 if p.get("type") == "done")
            assert done_payload.get("phase") == "pre_strategy"


def test_runtime_policy_can_replace_tools() -> None:
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
        captured_tools.append(tools)
        text = "Acknowledged."
        yield {"type": "response.output_text.delta", "delta": text, "sequence_number": 1}
        yield _make_mock_response_event(text, "resp_mock_tools")

    custom_tools = [
        {
            "type": "mcp",
            "server_label": "custom_runtime",
            "server_url": "https://example.com/mcp",
            "allowed_tools": ["custom_action"],
            "require_approval": "never",
        }
    ]

    with patch(
        "src.services.openai_stream_service.OpenAIResponsesEventStreamer.stream_events",
        side_effect=_mock_stream_events,
    ):
        with TestClient(app) as client:
            token = _register_and_get_token(client)
            response = client.post(
                "/api/v1/chat/send-openai-stream",
                headers={"Authorization": f"Bearer {token}"},
                json={
                    "message": "hello runtime policy",
                    "runtime_policy": {
                        "tool_mode": "replace",
                        "allowed_tools": custom_tools,
                    },
                },
            )

    assert response.status_code == 200
    assert captured_tools
    assert captured_tools[0] == custom_tools


def test_pre_strategy_normalizes_alias_market_and_symbol_end_to_end() -> None:
    responses = [
        _TURN1_RESPONSE,
        _TURN2_RESPONSE,
        _TURN3_RESPONSE,
        (
            "Great, strategy scope received."
            '<AGENT_STATE_PATCH>{"target_market":"stock","target_instrument":"spy","opportunity_frequency_bucket":"daily","holding_period_bucket":"intraday"}</AGENT_STATE_PATCH>'
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
        text = responses[idx]
        yield {"type": "response.output_text.delta", "delta": text, "sequence_number": 1}
        yield _make_mock_response_event(text, f"resp_mock_alias_{idx}")

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
            sid = next(
                p for p in _parse_sse_payloads(turn1.text) if p.get("type") == "done"
            )["session_id"]

            turn2 = client.post(
                "/api/v1/chat/send-openai-stream",
                headers=headers,
                json={"session_id": sid, "message": "My risk tolerance is aggressive."},
            )
            assert turn2.status_code == 200

            turn3 = client.post(
                "/api/v1/chat/send-openai-stream",
                headers=headers,
                json={"session_id": sid, "message": "I target high growth returns."},
            )
            assert turn3.status_code == 200
            done3 = next(
                p for p in _parse_sse_payloads(turn3.text) if p.get("type") == "done"
            )
            assert done3["phase"] == "pre_strategy"

            turn4 = client.post(
                "/api/v1/chat/send-openai-stream",
                headers=headers,
                json={"session_id": sid, "message": "stock + spy + daily + intraday"},
            )
            assert turn4.status_code == 200
            done4 = next(
                p for p in _parse_sse_payloads(turn4.text) if p.get("type") == "done"
            )
            assert done4["phase"] == "strategy"
            assert done4["missing_fields"] == []

            detail = client.get(f"/api/v1/sessions/{sid}", headers=headers)
            assert detail.status_code == 200
            profile = detail.json()["artifacts"]["pre_strategy"]["profile"]
            assert profile["target_market"] == "us_stocks"
            assert profile["target_instrument"] == "SPY"
            assert profile["opportunity_frequency_bucket"] == "daily"
            assert profile["holding_period_bucket"] == "intraday"


def test_pre_strategy_drops_cross_market_instrument_from_patch() -> None:
    responses = [
        _TURN1_RESPONSE,
        _TURN2_RESPONSE,
        _TURN3_RESPONSE,
        (
            "Market recorded, please pick a valid symbol."
            '<AGENT_STATE_PATCH>{"target_market":"forex","target_instrument":"SPY"}</AGENT_STATE_PATCH>'
            '<AGENT_UI_JSON>{"type":"choice_prompt","choice_id":"target_instrument",'
            '"question":"Pick a forex symbol",'
            '"options":[{"id":"SPY","label":"SPY"},'
            '{"id":"EURUSD","label":"EUR/USD"},'
            '{"id":"USDJPY","label":"USD/JPY"}]}</AGENT_UI_JSON>'
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
        text = responses[idx]
        yield {"type": "response.output_text.delta", "delta": text, "sequence_number": 1}
        yield _make_mock_response_event(text, f"resp_mock_cross_{idx}")

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
            sid = next(
                p for p in _parse_sse_payloads(turn1.text) if p.get("type") == "done"
            )["session_id"]

            turn2 = client.post(
                "/api/v1/chat/send-openai-stream",
                headers=headers,
                json={"session_id": sid, "message": "My risk tolerance is aggressive."},
            )
            assert turn2.status_code == 200

            turn3 = client.post(
                "/api/v1/chat/send-openai-stream",
                headers=headers,
                json={"session_id": sid, "message": "I target high growth returns."},
            )
            assert turn3.status_code == 200

            turn4 = client.post(
                "/api/v1/chat/send-openai-stream",
                headers=headers,
                json={"session_id": sid, "message": "forex + SPY"},
            )
            assert turn4.status_code == 200
            done4 = next(
                p for p in _parse_sse_payloads(turn4.text) if p.get("type") == "done"
            )
            assert done4["phase"] == "pre_strategy"
            assert "target_instrument" in done4["missing_fields"]

            detail = client.get(f"/api/v1/sessions/{sid}", headers=headers)
            assert detail.status_code == 200
            profile = detail.json()["artifacts"]["pre_strategy"]["profile"]
            assert profile["target_market"] == "forex"
            assert "target_instrument" not in profile


def test_session_artifacts_no_legacy_flat_keys() -> None:
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
        text = _MOCK_RESPONSES[idx]
        yield {"type": "response.output_text.delta", "delta": text, "sequence_number": 1}
        yield _make_mock_response_event(text, f"resp_mock_{idx}")

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
            sid = next(
                p for p in _parse_sse_payloads(turn1.text) if p.get("type") == "done"
            )["session_id"]

            client.post(
                "/api/v1/chat/send-openai-stream",
                headers=headers,
                json={"session_id": sid, "message": "My risk tolerance is aggressive."},
            )
            client.post(
                "/api/v1/chat/send-openai-stream",
                headers=headers,
                json={"session_id": sid, "message": "I target high growth returns."},
            )

            detail = client.get(f"/api/v1/sessions/{sid}", headers=headers)
            assert detail.status_code == 200
            artifacts = detail.json()["artifacts"]
            assert "kyc_profile" not in artifacts
            assert "kyc_missing_fields" not in artifacts
            assert "pre_strategy_profile" not in artifacts
            assert "pre_strategy_missing_fields" not in artifacts
