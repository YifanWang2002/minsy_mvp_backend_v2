"""Robustness tests for abnormal pre-strategy inputs and patches."""

from __future__ import annotations

import json
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient

from src.main import app

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


def _register_and_get_token(client: TestClient) -> str:
    email = f"pre_strategy_robust_{uuid4().hex[:10]}@test.com"
    resp = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "pass1234", "name": "Robust User"},
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
    return {
        "type": "response.completed",
        "response": {
            "id": response_id,
            "usage": {"input_tokens": 100, "output_tokens": 50},
        },
    }


def test_invalid_market_patch_is_ignored() -> None:
    responses = [
        _TURN1_RESPONSE,
        _TURN2_RESPONSE,
        _TURN3_RESPONSE,
        (
            "I couldn't parse that market. Please choose from the list."
            '<AGENT_STATE_PATCH>{"target_market":"crypto_perpetuals"}</AGENT_STATE_PATCH>'
            '<AGENT_UI_JSON>{"type":"choice_prompt","choice_id":"target_market",'
            '"question":"Which market do you want to trade?",'
            '"options":[{"id":"us_stocks","label":"US Stocks"},'
            '{"id":"crypto","label":"Crypto"},'
            '{"id":"forex","label":"Forex"},'
            '{"id":"futures","label":"Futures"}]}</AGENT_UI_JSON>'
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
        yield _make_mock_response_event(text, f"resp_robust_market_{idx}")

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
            turn4 = client.post(
                "/api/v1/chat/send-openai-stream",
                headers=headers,
                json={"session_id": sid, "message": "crypto perpetuals"},
            )
            assert turn4.status_code == 200
            done4 = next(
                p for p in _parse_sse_payloads(turn4.text) if p.get("type") == "done"
            )
            assert done4["phase"] == "pre_strategy"
            assert "target_market" in done4["missing_fields"]

            detail = client.get(f"/api/v1/sessions/{sid}", headers=headers)
            assert detail.status_code == 200
            profile = detail.json()["artifacts"]["pre_strategy"]["profile"]
            assert "target_market" not in profile


def test_non_string_patch_values_are_ignored_safely() -> None:
    responses = [
        _TURN1_RESPONSE,
        _TURN2_RESPONSE,
        _TURN3_RESPONSE,
        (
            "Please choose your market."
            '<AGENT_STATE_PATCH>{"target_market":123,"target_instrument":["SPY"],'
            '"opportunity_frequency_bucket":{"x":"daily"},"holding_period_bucket":null}</AGENT_STATE_PATCH>'
            '<AGENT_UI_JSON>{"type":"choice_prompt","choice_id":"target_market",'
            '"question":"Which market do you want to trade?",'
            '"options":[{"id":"us_stocks","label":"US Stocks"},'
            '{"id":"crypto","label":"Crypto"},'
            '{"id":"forex","label":"Forex"},'
            '{"id":"futures","label":"Futures"}]}</AGENT_UI_JSON>'
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
        yield _make_mock_response_event(text, f"resp_robust_types_{idx}")

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
            turn4 = client.post(
                "/api/v1/chat/send-openai-stream",
                headers=headers,
                json={"session_id": sid, "message": "???"},
            )
            assert turn4.status_code == 200
            done4 = next(
                p for p in _parse_sse_payloads(turn4.text) if p.get("type") == "done"
            )
            assert done4["phase"] == "pre_strategy"
            assert set(done4["missing_fields"]).issuperset(
                {
                    "target_market",
                    "target_instrument",
                    "opportunity_frequency_bucket",
                    "holding_period_bucket",
                }
            )


def test_legacy_market_id_is_recovered_by_valid_symbol_inference() -> None:
    responses = [
        _TURN1_RESPONSE,
        _TURN2_RESPONSE,
        _TURN3_RESPONSE,
        (
            "Scope captured."
            '<AGENT_STATE_PATCH>{"target_market":"crypto_perpetuals","target_instrument":"BTCUSD",'
            '"opportunity_frequency_bucket":"daily","holding_period_bucket":"intraday"}</AGENT_STATE_PATCH>'
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
        yield _make_mock_response_event(text, f"resp_robust_legacy_{idx}")

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
            turn4 = client.post(
                "/api/v1/chat/send-openai-stream",
                headers=headers,
                json={"session_id": sid, "message": "crypto_perpetuals + BTCUSD"},
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
            assert profile["target_market"] == "crypto"
            assert profile["target_instrument"] == "BTCUSD"
            assert profile["opportunity_frequency_bucket"] == "daily"
            assert profile["holding_period_bucket"] == "intraday"


def test_malformed_choice_options_do_not_crash_and_keep_phase() -> None:
    responses = [
        _TURN1_RESPONSE,
        _TURN2_RESPONSE,
        _TURN3_RESPONSE,
        (
            "Market set. Pick a valid forex symbol."
            '<AGENT_STATE_PATCH>{"target_market":"forex"}</AGENT_STATE_PATCH>'
            '<AGENT_UI_JSON>{"type":"choice_prompt","choice_id":"target_instrument",'
            '"question":"Pick forex symbol",'
            '"options":[{"id":"SPY","label":"SPY"},"bad",{"label":"No id"},123]}</AGENT_UI_JSON>'
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
        yield _make_mock_response_event(text, f"resp_robust_options_{idx}")

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
            turn4 = client.post(
                "/api/v1/chat/send-openai-stream",
                headers=headers,
                json={"session_id": sid, "message": "forex"},
            )
            assert turn4.status_code == 200
            payloads4 = _parse_sse_payloads(turn4.text)
            done4 = next(p for p in payloads4 if p.get("type") == "done")
            assert done4["phase"] == "pre_strategy"
            assert done4.get("stream_error") in (None, "")

            # Malformed options may be dropped by the genui normalizer.
            # The key contract is: no crash, phase remains healthy.
            genui_payloads = [p for p in payloads4 if p.get("type") == "genui"]
            assert len(genui_payloads) in {0, 1}


def test_structurally_valid_wrong_options_fallback_to_market_symbols() -> None:
    responses = [
        _TURN1_RESPONSE,
        _TURN2_RESPONSE,
        _TURN3_RESPONSE,
        (
            "Market set. Pick a valid forex symbol."
            '<AGENT_STATE_PATCH>{"target_market":"forex"}</AGENT_STATE_PATCH>'
            '<AGENT_UI_JSON>{"type":"choice_prompt","choice_id":"target_instrument",'
            '"question":"Pick forex symbol",'
            '"options":[{"id":"SPY","label":"SPY"},{"id":"AAPL","label":"AAPL"}]}</AGENT_UI_JSON>'
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
        yield _make_mock_response_event(text, f"resp_robust_options_fallback_{idx}")

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
            turn4 = client.post(
                "/api/v1/chat/send-openai-stream",
                headers=headers,
                json={"session_id": sid, "message": "forex"},
            )
            assert turn4.status_code == 200
            payloads4 = _parse_sse_payloads(turn4.text)
            genui_event = next(p for p in payloads4 if p.get("type") == "genui")
            genui_payload = genui_event["payload"]
            assert genui_payload["choice_id"] == "target_instrument"

            option_ids = [item["id"] for item in genui_payload["options"]]
            assert "SPY" not in option_ids
            assert "AAPL" not in option_ids
            assert len(option_ids) >= 2
            assert "EURUSD" in option_ids
            assert all("subtitle" in item for item in genui_payload["options"])
