"""Frontend-like probe for KYC choice widget rendering via SSE.

Usage:
    uv run python -m src.scripts.genui_frontend_probe --language en
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from uuid import uuid4

from fastapi.testclient import TestClient

from src.main import app


def _parse_sse_events(raw_text: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    current_event = "message"
    for block in raw_text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        for line in block.splitlines():
            if line.startswith("event: "):
                current_event = line.removeprefix("event: ").strip()
            elif line.startswith("data: "):
                try:
                    payload = json.loads(line.removeprefix("data: ").strip())
                except json.JSONDecodeError:
                    continue
                events.append((current_event, payload))
    return events


def _extract_done(events: list[tuple[str, dict]]) -> dict:
    for _, payload in reversed(events):
        if payload.get("type") == "done":
            return payload
    return {}


def _choice_visible_to_frontend(events: list[tuple[str, dict]]) -> tuple[bool, str]:
    genui_payloads = [
        payload.get("payload")
        for _, payload in events
        if payload.get("type") == "genui"
    ]
    valid_genui = [p for p in genui_payloads if isinstance(p, dict) and p.get("type") == "choice_prompt"]
    if valid_genui:
        return True, "via `genui` event"

    full_text = "".join(
        payload.get("delta", "")
        for _, payload in events
        if payload.get("type") == "text_delta"
    )
    has_tagged_json = bool(
        re.search(r"<AGENT_UI_JSON>[\s\S]*?</AGENT_UI_JSON>", full_text)
    )
    if has_tagged_json:
        return True, "via tagged JSON in text stream"

    return False, "no `genui` and no `<AGENT_UI_JSON>` block"


def run(language: str) -> int:
    turns_en = [
        "I have over five years of trading experience.",
        "My risk tolerance is aggressive.",
        "My return target is high growth.",
        "US equities.",
        "SPY.",
        "Daily opportunities.",
        "Swing style.",
    ]
    turns_zh = [
        "我有超过五年的交易经验。",
        "我的风险偏好是激进型。",
        "我的回报目标是高增长。",
        "美股。",
        "SPY。",
        "每天都可以。",
        "偏向波段持仓几天。",
    ]
    turns = turns_zh if language == "zh" else turns_en

    with TestClient(app) as client:
        email = f"genui_probe_{uuid4().hex[:8]}@test.com"
        reg = client.post(
            "/api/v1/auth/register",
            json={"email": email, "password": "test1234", "name": "GenUI Probe"},
        )
        if reg.status_code != 201:
            print(f"[FAIL] register: {reg.status_code} {reg.text}")
            return 1

        token = reg.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        session_id: str | None = None
        saw_choice_in_kyc = False
        saw_choice_in_pre_strategy = False

        print(f"[INFO] registered user={email}")

        for idx, user_msg in enumerate(turns, start=1):
            body: dict[str, str] = {"message": user_msg}
            if session_id:
                body["session_id"] = session_id

            resp = client.post(
                f"/api/v1/chat/send-openai-stream?language={language}",
                headers=headers,
                json=body,
            )
            if resp.status_code != 200:
                print(f"[FAIL] turn={idx} status={resp.status_code} body={resp.text}")
                return 1

            events = _parse_sse_events(resp.text)
            done = _extract_done(events)
            session_id = done.get("session_id", session_id)
            phase = done.get("phase", "?")
            kyc_status = done.get("kyc_status", "?")
            missing_fields = done.get("missing_fields", [])

            visible, reason = _choice_visible_to_frontend(events)
            if phase == "kyc" and visible:
                saw_choice_in_kyc = True
            if phase == "pre_strategy" and visible:
                saw_choice_in_pre_strategy = True

            print(
                f"[TURN {idx}] phase={phase} kyc={kyc_status} missing={missing_fields} "
                f"choice_visible={visible} ({reason})"
            )

        if not saw_choice_in_kyc:
            print(
                "[FAIL] Did not observe any frontend-renderable choice prompt during KYC turns."
            )
            return 2

        if not saw_choice_in_pre_strategy:
            print(
                "[FAIL] Did not observe any frontend-renderable choice prompt during pre_strategy turns."
            )
            return 3

        print("[PASS] Observed frontend-renderable choice prompt in KYC and pre_strategy flows.")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe frontend choice-widget visibility in SSE.")
    parser.add_argument("--language", default="en", help="ISO 639-1 language code, e.g. en/zh")
    args = parser.parse_args()
    return run(args.language)


if __name__ == "__main__":
    sys.exit(main())
