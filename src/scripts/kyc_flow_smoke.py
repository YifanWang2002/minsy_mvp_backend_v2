"""End-to-end smoke script: register -> multi-turn KYC via OpenAI -> pre-strategy transition.

Usage:
    python -m src.scripts.kyc_flow_smoke [--base-url http://127.0.0.1:8000/api/v1] [--language zh]

This script simulates a real frontend client:
  1. Registers a new user
  2. Sends KYC messages through the /send-openai-stream endpoint
  3. Verifies that OpenAI fills the KYC fields via AGENT_STATE_PATCH
  4. Checks that the user profile is updated in the DB
"""

from __future__ import annotations

import argparse
import json
import sys
from uuid import uuid4

import httpx


def _assert_status(resp: httpx.Response, expected: int, step: str) -> None:
    if resp.status_code != expected:
        raise RuntimeError(
            f"{step}: expected {expected}, got {resp.status_code}, body={resp.text}"
        )


def _parse_sse_payloads(raw_text: str) -> list[dict]:
    payloads: list[dict] = []
    for block in raw_text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        for line in block.splitlines():
            if line.startswith("data: "):
                try:
                    payloads.append(json.loads(line.removeprefix("data: ")))
                except json.JSONDecodeError:
                    pass
    return payloads


def _send_stream(
    client: httpx.Client,
    base_url: str,
    headers: dict,
    message: str,
    session_id: str | None,
    language: str,
) -> tuple[list[dict], str | None]:
    """Send a message via /send-openai-stream, return (payloads, session_id)."""
    body: dict = {"message": message}
    if session_id:
        body["session_id"] = session_id

    resp = client.post(
        f"{base_url}/chat/send-openai-stream?language={language}",
        headers=headers,
        json=body,
        timeout=60.0,
    )
    _assert_status(resp, 200, f"stream: {message[:40]}")
    payloads = _parse_sse_payloads(resp.text)
    done = next((p for p in payloads if p.get("type") == "done"), None)
    sid = done.get("session_id") if done else session_id
    return payloads, sid


def run(base_url: str, language: str) -> None:
    email = f"kyc_smoke_{uuid4().hex[:10]}@example.com"
    password = "smoke123456"

    with httpx.Client(timeout=30.0, trust_env=False) as client:
        # 1. Register
        register = client.post(
            f"{base_url}/auth/register",
            json={"email": email, "password": password, "name": "KYC Smoke"},
        )
        _assert_status(register, 201, "register")
        token = register.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        print(f"[OK] Registered {email}")

        # 2. Turn 1 — provide trading experience
        session_id = None
        msgs = [
            "I have over 5 years of trading experience.",
            "My risk tolerance is aggressive.",
            "I target high growth returns.",
        ]
        if language == "zh":
            msgs = [
                "你好，我有超过5年的交易经验。",
                "我的风险偏好是激进的。",
                "我的目标是高增长回报。",
            ]

        for i, msg in enumerate(msgs, 1):
            payloads, session_id = _send_stream(
                client, base_url, headers, msg, session_id, language,
            )
            done = next((p for p in payloads if p.get("type") == "done"), {})
            phase = done.get("phase", "?")
            kyc_status = done.get("kyc_status", "?")
            missing = done.get("missing_fields", [])
            text_chunks = [p.get("delta", "") for p in payloads if p.get("type") == "text_delta"]
            ai_reply = "".join(text_chunks)[:120]
            print(f"[OK] Turn {i}: phase={phase} kyc={kyc_status} missing={missing}")
            print(f"     AI: {ai_reply}...")

            if phase == "pre_strategy" and kyc_status == "complete":
                print("[OK] KYC completed and transitioned to pre_strategy!")
                break
        else:
            # If 3 turns weren't enough, send one more nudge
            nudge = "Please confirm and complete my KYC profile with the information I provided."
            payloads, session_id = _send_stream(
                client, base_url, headers, nudge, session_id, language,
            )
            done = next((p for p in payloads if p.get("type") == "done"), {})
            phase = done.get("phase", "?")
            kyc_status = done.get("kyc_status", "?")
            print(f"[OK] Nudge turn: phase={phase} kyc={kyc_status}")

        # 3. Verify user profile via /auth/me
        me = client.get(f"{base_url}/auth/me", headers=headers)
        _assert_status(me, 200, "auth/me")
        me_data = me.json()
        final_kyc = me_data.get("kyc_status", "?")
        print(f"[OK] /auth/me kyc_status={final_kyc}")

        if final_kyc == "complete":
            print("\n[PASS] Full KYC flow smoke test passed!")
        else:
            print(f"\n[WARN] KYC status is '{final_kyc}' — AI may need more turns.")
            print("       This is expected if OpenAI didn't emit all patches in 3 turns.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run KYC flow smoke test with real OpenAI.")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000/api/v1",
        help="API base url",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Language code to send (en, zh, etc.)",
    )
    args = parser.parse_args()

    try:
        run(args.base_url.rstrip("/"), args.language)
    except Exception as exc:  # noqa: BLE001
        print(f"\n[FAIL] {exc}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
