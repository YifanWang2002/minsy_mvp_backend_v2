"""Verbose end-to-end KYC integration test — prints every detail for human review.

Run with:  uv run python -m pytest tests/test_api/test_kyc_e2e_verbose.py -v -s
"""

from __future__ import annotations

import json
from uuid import uuid4

from fastapi.testclient import TestClient

from src.main import app

SEPARATOR = "=" * 80
THIN_SEP = "-" * 60


def _parse_sse_events(raw_text: str) -> list[tuple[str, dict]]:
    """Return list of (event_name, payload) tuples."""
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
                    data = json.loads(line.removeprefix("data: "))
                    events.append((current_event, data))
                except json.JSONDecodeError:
                    pass
    return events


def test_full_kyc_flow_with_real_openai_verbose() -> None:
    """Multi-turn KYC conversation with real OpenAI, printing all details."""

    with TestClient(app) as client:
        # ── Register ──
        email = f"e2e_{uuid4().hex[:8]}@test.com"
        reg = client.post(
            "/api/v1/auth/register",
            json={"email": email, "password": "test1234", "name": "E2E Tester"},
        )
        assert reg.status_code == 201
        token = reg.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        print(f"\n{SEPARATOR}")
        print(f"REGISTERED: {email}")
        print(SEPARATOR)

        session_id: str | None = None
        language = "zh"

        turns = [
            "你好，我有超过5年的交易经验。",
            "我的风险偏好是激进的。",
            "我追求高增长回报。",
        ]

        for turn_idx, user_msg in enumerate(turns, 1):
            print(f"\n{SEPARATOR}")
            print(f"TURN {turn_idx}")
            print(SEPARATOR)
            print(f">>> USER: {user_msg}")
            print(THIN_SEP)

            body: dict = {"message": user_msg}
            if session_id:
                body["session_id"] = session_id

            resp = client.post(
                f"/api/v1/chat/send-openai-stream?language={language}",
                headers=headers,
                json=body,
            )
            assert resp.status_code == 200

            events = _parse_sse_events(resp.text)

            # Collect text deltas
            full_text = ""
            genui_payload = None
            done_payload = None

            print(f"\n  [SSE Events received: {len(events)}]")
            print()

            for evt_name, evt_data in events:
                evt_type = evt_data.get("type", "?")

                if evt_type == "stream_start":
                    sid = evt_data.get("session_id", "?")
                    phase = evt_data.get("phase", "?")
                    print(f"  STREAM_START  session={sid}  phase={phase}")
                    if not session_id:
                        session_id = sid

                elif evt_type == "text_delta":
                    delta = evt_data.get("delta", "")
                    full_text += delta
                    # Print each delta inline
                    print(f"  TEXT_DELTA: \"{delta}\"")

                elif evt_type == "openai_event":
                    otype = evt_data.get("openai_type", "?")
                    # Only print key lifecycle events, not every delta
                    if otype in (
                        "response.created",
                        "response.in_progress",
                        "response.completed",
                        "response.output_text.done",
                    ):
                        print(f"  OPENAI_EVENT: {otype}")

                elif evt_type == "genui":
                    genui_payload = evt_data.get("payload")
                    print(f"  GENUI: {json.dumps(genui_payload, ensure_ascii=False, indent=4)}")

                elif evt_type == "phase_change":
                    fr = evt_data.get("from_phase", "?")
                    to = evt_data.get("to_phase", "?")
                    print(f"  PHASE_CHANGE: {fr} -> {to}")

                elif evt_type == "done":
                    done_payload = evt_data
                    print(f"  DONE: {json.dumps(done_payload, ensure_ascii=False, indent=4)}")

                elif evt_type == "mcp_event":
                    otype = evt_data.get("openai_type", "?")
                    print(f"  MCP_EVENT: {otype}")

            # ── Summary for this turn ──
            print(f"\n{THIN_SEP}")
            print("<<< AI FULL TEXT:")
            print(f"    {full_text}")
            print(THIN_SEP)

            if done_payload:
                phase = done_payload.get("phase", "?")
                kyc_status = done_payload.get("kyc_status", "?")
                missing = done_payload.get("missing_fields", [])
                print(f"  phase={phase}  kyc_status={kyc_status}  missing_fields={missing}")

            if done_payload and done_payload.get("kyc_status") == "complete":
                print(f"\n  *** KYC COMPLETED at turn {turn_idx} ***")
                break

        # ── Verify DB state ──
        print(f"\n{SEPARATOR}")
        print("VERIFICATION")
        print(SEPARATOR)

        me = client.get("/api/v1/auth/me", headers=headers)
        assert me.status_code == 200
        me_data = me.json()
        print(f"  /auth/me kyc_status = {me_data.get('kyc_status', '?')}")
        print(f"  /auth/me full response = {json.dumps(me_data, ensure_ascii=False, indent=4)}")

        if session_id:
            detail = client.get(f"/api/v1/sessions/{session_id}", headers=headers)
            assert detail.status_code == 200
            detail_data = detail.json()
            print(f"\n  Session phase = {detail_data.get('current_phase', '?')}")
            artifacts = detail_data.get("artifacts", {})
            kyc_block = artifacts.get("kyc", {})
            kyc_profile = kyc_block.get("profile", {}) if isinstance(kyc_block, dict) else {}
            print(f"  KYC Profile in artifacts = {json.dumps(kyc_profile, ensure_ascii=False, indent=4)}")
            missing = kyc_block.get("missing_fields", []) if isinstance(kyc_block, dict) else []
            print(f"  Missing fields = {missing}")
            msgs = detail_data.get("messages", [])
            print(f"  Total messages in session = {len(msgs)}")
            for m in msgs:
                role = m.get("role", "?")
                content = m.get("content", "")[:120]
                print(f"    [{role}] {content}...")

        print(f"\n{SEPARATOR}")
        print("TEST COMPLETE")
        print(SEPARATOR)
