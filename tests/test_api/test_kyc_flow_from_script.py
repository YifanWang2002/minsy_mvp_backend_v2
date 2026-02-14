"""Smoke-test equivalent of the KYC flow script — hits real OpenAI via /send-openai-stream."""

from __future__ import annotations

import json
from uuid import uuid4

from fastapi.testclient import TestClient

from src.main import app


def _assert_status(response, expected: int, step: str) -> None:
    if response.status_code != expected:
        raise AssertionError(
            f"{step}: expected {expected}, got {response.status_code}, body={response.text}"
        )


def _parse_sse_payloads(raw_text: str) -> list[dict]:
    payloads: list[dict] = []
    for block in raw_text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        for line in block.splitlines():
            if line.startswith("data: "):
                payloads.append(json.loads(line.removeprefix("data: ")))
    return payloads


def _get_done_payload(raw_text: str) -> dict:
    payloads = _parse_sse_payloads(raw_text)
    return next(p for p in payloads if p.get("type") == "done")


def test_kyc_flow_script_equivalent() -> None:
    email = f"kyc_smoke_{uuid4().hex[:10]}@example.com"
    password = "123456"

    with TestClient(app) as client:
        register = client.post(
            "/api/v1/auth/register",
            json={"email": email, "password": password, "name": "KYC Smoke"},
        )
        _assert_status(register, 201, "register")
        register_json = register.json()
        access_token = register_json["access_token"]
        headers = {"Authorization": f"Bearer {access_token}"}

        turn1 = client.post(
            "/api/v1/chat/send-openai-stream?language=zh",
            headers=headers,
            json={"message": "你好，我有5年交易经验。"},
        )
        _assert_status(turn1, 200, "chat turn1")
        done1 = _get_done_payload(turn1.text)
        session_id = done1["session_id"]

        turn2 = client.post(
            "/api/v1/chat/send-openai-stream?language=zh",
            headers=headers,
            json={"session_id": session_id, "message": "我的风险偏好是aggressive。"},
        )
        _assert_status(turn2, 200, "chat turn2")

        turn3 = client.post(
            "/api/v1/chat/send-openai-stream?language=zh",
            headers=headers,
            json={"session_id": session_id, "message": "我追求高回报，目标年化30%。"},
        )
        _assert_status(turn3, 200, "chat turn3")
        done3 = _get_done_payload(turn3.text)
        # Real model output is non-deterministic: after KYC completion it may
        # remain in pre_strategy to collect market/symbol, or already move forward.
        assert done3["phase"] in {"pre_strategy", "strategy"}

        me = client.get("/api/v1/auth/me", headers=headers)
        _assert_status(me, 200, "auth/me")
        assert me.json()["kyc_status"] == "complete"

        session_detail = client.get(f"/api/v1/sessions/{session_id}", headers=headers)
        _assert_status(session_detail, 200, "session detail")
