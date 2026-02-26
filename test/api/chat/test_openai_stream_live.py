from __future__ import annotations

from fastapi.testclient import TestClient

from test._support.live_helpers import parse_sse_payloads


def test_000_accessibility_openai_key_and_stream(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    response = api_test_client.post(
        "/api/v1/chat/send-openai-stream?language=zh",
        headers=auth_headers,
        json={
            "message": "我有5年交易经验，请继续问我风险偏好和收益目标问题，不要调用任何工具。",
        },
    )
    assert response.status_code == 200, response.text

    payloads = parse_sse_payloads(response.text)
    assert payloads

    openai_types = [
        payload.get("openai_type")
        for payload in payloads
        if payload.get("type") == "openai_event"
    ]
    assert "response.created" in openai_types
    assert "response.completed" in openai_types

    done_payload = next(item for item in payloads if item.get("type") == "done")
    assert done_payload.get("phase") in {"kyc", "pre_strategy", "strategy", "stress_test", "deployment"}

    text = "".join(item.get("delta", "") for item in payloads if item.get("type") == "text_delta")
    assert len(text.strip()) >= 30


def test_010_openai_stream_completion_quality(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    response = api_test_client.post(
        "/api/v1/chat/send-openai-stream?language=en",
        headers=auth_headers,
        json={
            "message": "In exactly 3 bullet points, summarize what information you still need from me to design a strategy.",
        },
    )
    assert response.status_code == 200, response.text

    payloads = parse_sse_payloads(response.text)
    done_payload = next(item for item in payloads if item.get("type") == "done")
    usage = done_payload.get("usage")
    if isinstance(usage, dict):
        assert int(usage.get("total_tokens", 0)) > 0

    text = "".join(item.get("delta", "") for item in payloads if item.get("type") == "text_delta")
    bullet_lines = [
        line
        for line in text.splitlines()
        if line.strip().startswith(("-", "*", "•", "1.", "2.", "3."))
    ]
    assert len(bullet_lines) >= 3, text
