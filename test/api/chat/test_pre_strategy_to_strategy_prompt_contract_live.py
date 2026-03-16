from __future__ import annotations

import json
import re
import time
from typing import Any

from fastapi.testclient import TestClient

from test._support.live_helpers import parse_sse_payloads


def _extract_latest_choice_prompt(payloads: list[dict[str, Any]]) -> dict[str, Any] | None:
    latest: dict[str, Any] | None = None
    for payload in payloads:
        if payload.get("type") != "genui":
            continue
        raw = payload.get("payload")
        if isinstance(raw, dict) and raw.get("type") == "choice_prompt":
            latest = dict(raw)

    if latest is not None:
        return latest

    full_text = "".join(
        payload.get("delta", "")
        for payload in payloads
        if payload.get("type") == "text_delta"
    )
    matches = list(
        re.finditer(
            r"<\s*AGENT_UI_JSON\s*>([\s\S]*?)</\s*AGENT_UI_JSON\s*>",
            full_text,
            flags=re.IGNORECASE,
        )
    )
    for match in reversed(matches):
        try:
            decoded = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(decoded, dict) and decoded.get("type") == "choice_prompt":
            return decoded
    return None


def _pick_label(prompt: dict[str, Any], *, preferred_id: str | None = None) -> str:
    options = prompt.get("options")
    assert isinstance(options, list) and options, prompt
    if preferred_id is not None:
        for option in options:
            if not isinstance(option, dict):
                continue
            if str(option.get("id", "")).strip() == preferred_id:
                label = str(option.get("label", "")).strip()
                if label:
                    return label
    for option in options:
        if not isinstance(option, dict):
            continue
        label = str(option.get("label", "")).strip()
        if label:
            return label
    raise AssertionError(f"No selectable label found in prompt: {prompt}")


def _preferred_option_id(choice_id: str) -> str | None:
    mapping = {
        "kyc_trading_years_bucket": "years_5_plus",
        "trading_years_bucket": "years_5_plus",
        "kyc_risk_tolerance": "aggressive",
        "risk_tolerance": "aggressive",
        "kyc_return_expectation": "high_growth",
        "return_expectation": "high_growth",
        "target_market": "us_stocks",
        "target_instrument": "SPY",
        "opportunity_frequency_bucket": "daily",
        "holding_period_bucket": "intraday",
        "strategy_family_choice": "trend_continuation",
    }
    return mapping.get(choice_id.strip())


def _stream_turn(
    client: TestClient,
    *,
    headers: dict[str, str],
    message: str,
    session_id: str | None,
    language: str = "zh",
) -> tuple[str, dict[str, Any], dict[str, Any] | None, str]:
    body: dict[str, Any] = {"message": message}
    if session_id is not None:
        body["session_id"] = session_id

    response = client.post(
        f"/api/v1/chat/send-openai-stream?language={language}",
        headers=headers,
        json=body,
    )
    assert response.status_code == 200, response.text

    payloads = parse_sse_payloads(response.text)
    start = next((item for item in payloads if item.get("type") == "stream_start"), None)
    done = next((item for item in payloads if item.get("type") == "done"), None)
    assert start is not None, payloads
    assert done is not None, payloads

    resolved_session_id = str(start["session_id"])
    prompt = _extract_latest_choice_prompt(payloads)
    text = "".join(
        payload.get("delta", "")
        for payload in payloads
        if payload.get("type") == "text_delta"
    )
    if prompt is not None:
        choice_id = str(prompt.get("choice_id", "")).strip()
    else:
        choice_id = "none"
    print(
        f"turn message={message!r} phase={done.get('phase')} "
        f"missing={done.get('missing_fields')} choice_id={choice_id}",
        flush=True,
    )
    return resolved_session_id, dict(done), prompt, text


def test_000_full_kyc_to_strategy_flow_with_real_openai_and_frontend_style_labels_live(
    api_test_client: TestClient,
) -> None:
    register_email = f"prompt_contract_{time.time_ns()}@test.com"
    register = api_test_client.post(
        "/api/v1/auth/register",
        json={
            "email": register_email,
            "password": "test1234",
            "name": "Prompt Contract Live",
        },
    )
    assert register.status_code in {200, 201}, register.text
    register_payload = register.json()
    headers = {"Authorization": f"Bearer {register_payload['access_token']}"}

    session_id, done, prompt, _ = _stream_turn(
        api_test_client,
        headers=headers,
        message="你好，我有超过5年的交易经验。",
        session_id=None,
    )
    assert done["phase"] in {"kyc", "pre_strategy"}, done

    for _ in range(16):
        if done["phase"] == "strategy":
            break

        if prompt is None:
            next_message = "继续"
        else:
            choice_id = str(prompt.get("choice_id", "")).strip()
            next_message = _pick_label(
                prompt,
                preferred_id=_preferred_option_id(choice_id),
            )

        session_id, done, prompt, _ = _stream_turn(
            api_test_client,
            headers=headers,
            message=next_message,
            session_id=session_id,
        )

    assert done["phase"] == "strategy", done
    assert done["missing_fields"] == [], done
