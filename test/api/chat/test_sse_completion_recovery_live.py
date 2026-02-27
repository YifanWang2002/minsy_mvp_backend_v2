from __future__ import annotations

import json
import time
from collections.abc import Iterable, Iterator
from typing import Any
from uuid import uuid4

import httpx

from test._support.live_helpers import parse_sse_payloads

_BASE_URL = "http://127.0.0.1:8000"
_DEFAULT_TIMEOUT = httpx.Timeout(connect=10.0, read=360.0, write=60.0, pool=60.0)


def _build_client() -> httpx.Client:
    return httpx.Client(base_url=_BASE_URL, timeout=_DEFAULT_TIMEOUT, trust_env=False)


def _login_headers(
    client: httpx.Client,
    seeded_user_credentials: tuple[str, str],
) -> dict[str, str]:
    email, password = seeded_user_credentials
    response = client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": password},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    access_token = str(payload["access_token"])
    return {"Authorization": f"Bearer {access_token}"}


def _create_thread(client: httpx.Client, auth_headers: dict[str, str]) -> str:
    response = client.post(
        "/api/v1/chat/new-thread",
        headers=auth_headers,
        json={"metadata": {"source": "pytest-sse-recovery"}},
    )
    assert response.status_code == 201, response.text
    return str(response.json()["session_id"])


def _iter_sse_payloads_from_lines(lines: Iterable[str | bytes]) -> Iterator[dict[str, Any]]:
    current_event = "message"
    current_data: list[str] = []

    def _flush() -> Iterator[dict[str, Any]]:
        nonlocal current_event, current_data
        if not current_data:
            return
        raw = "\n".join(current_data)
        current_event = current_event or "message"
        current_data = []
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return
        if isinstance(payload, dict):
            payload["_sse_event"] = current_event
            yield payload

    for raw_line in lines:
        line = raw_line.decode("utf-8", errors="ignore") if isinstance(raw_line, bytes) else raw_line
        line = line.rstrip("\r")

        if not line:
            yield from _flush()
            current_event = "message"
            continue

        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            current_event = line[6:].strip() or "message"
            continue
        if line.startswith("data:"):
            current_data.append(line[5:].strip())

    yield from _flush()


def _find_message_by_id(messages: list[dict[str, Any]], message_id: str) -> dict[str, Any] | None:
    for message in messages:
        if str(message.get("id")) == message_id:
            return message
    return None


def _wait_for_turn_completion(
    client: httpx.Client,
    auth_headers: dict[str, str],
    *,
    session_id: str,
    turn_id: str,
    timeout_seconds: int = 240,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    deadline = time.monotonic() + float(timeout_seconds)
    while time.monotonic() < deadline:
        try:
            detail_response = client.get(
                f"/api/v1/sessions/{session_id}",
                headers=auth_headers,
            )
        except httpx.HTTPError:
            time.sleep(2)
            continue

        if detail_response.status_code != 200:
            time.sleep(2)
            continue
        detail = detail_response.json()

        stream_recovery = detail.get("stream_recovery")
        if not isinstance(stream_recovery, dict):
            time.sleep(2)
            continue

        if str(stream_recovery.get("turn_id", "")).strip() != turn_id:
            time.sleep(2)
            continue

        state = str(stream_recovery.get("state", "")).strip().lower()
        if state != "completed":
            time.sleep(2)
            continue

        assistant_message_id = str(stream_recovery.get("assistant_message_id", "")).strip()
        if not assistant_message_id:
            time.sleep(2)
            continue

        messages = detail.get("messages")
        if not isinstance(messages, list):
            time.sleep(2)
            continue

        assistant_message = _find_message_by_id(messages, assistant_message_id)
        if assistant_message is None:
            time.sleep(2)
            continue

        return detail, stream_recovery, assistant_message

    raise AssertionError(
        f"Timed out waiting for completed stream_recovery turn_id={turn_id} session_id={session_id}"
    )


def test_000_sse_done_and_session_recovery_contract(
    compose_stack: list[dict[str, Any]],
    seeded_user_credentials: tuple[str, str],
) -> None:
    _ = compose_stack

    with _build_client() as client:
        auth_headers = _login_headers(client, seeded_user_credentials)
        session_id = _create_thread(client, auth_headers)
        turn_id = f"it_turn_{uuid4().hex[:24]}"

        response = client.post(
            "/api/v1/chat/send-openai-stream?language=zh",
            headers=auth_headers,
            json={
                "session_id": session_id,
                "client_turn_id": turn_id,
                "message": "请用中文给出5条风控建议，每条至少20字，不要调用工具。",
            },
        )
        assert response.status_code == 200, response.text

        payloads = parse_sse_payloads(response.text)
        assert payloads, "SSE payloads should not be empty"

        stream_start = next(item for item in payloads if item.get("type") == "stream_start")
        assert str(stream_start.get("turn_id", "")).strip() == turn_id
        assert str(stream_start.get("session_id", "")).strip() == session_id

        done_payload = next(item for item in payloads if item.get("type") == "done")
        assert str(done_payload.get("turn_id", "")).strip() == turn_id

        assistant_message_id = str(done_payload.get("assistant_message_id", "")).strip()
        assert assistant_message_id

        detail_response = client.get(
            f"/api/v1/sessions/{session_id}",
            headers=auth_headers,
        )
        assert detail_response.status_code == 200, detail_response.text
        detail = detail_response.json()

        stream_recovery = detail.get("stream_recovery")
        assert isinstance(stream_recovery, dict)
        assert str(stream_recovery.get("turn_id", "")).strip() == turn_id
        assert str(stream_recovery.get("state", "")).strip().lower() == "completed"
        assert str(stream_recovery.get("assistant_message_id", "")).strip() == assistant_message_id
        assert str(stream_recovery.get("user_message_id", "")).strip()
        assert str(stream_recovery.get("started_at", "")).strip()
        assert str(stream_recovery.get("updated_at", "")).strip()

        messages = detail.get("messages")
        assert isinstance(messages, list)

        assistant_message = _find_message_by_id(messages, assistant_message_id)
        assert isinstance(assistant_message, dict)
        assert assistant_message.get("role") == "assistant"
        assert str(assistant_message.get("content", "")).strip()


def test_010_stream_disconnect_still_persists_assistant_message(
    compose_stack: list[dict[str, Any]],
    seeded_user_credentials: tuple[str, str],
) -> None:
    _ = compose_stack

    with _build_client() as client:
        auth_headers = _login_headers(client, seeded_user_credentials)
        session_id = _create_thread(client, auth_headers)
        turn_id = f"it_disconnect_{uuid4().hex[:22]}"

        payload = {
            "session_id": session_id,
            "client_turn_id": turn_id,
            "message": (
                "请用中文输出5条编号建议，每条至少15字，并解释风险收益权衡，不要调用工具。"
            ),
        }

        saw_stream_start = False
        saw_text_delta_before_disconnect = False

        with client.stream(
            "POST",
            "/api/v1/chat/send-openai-stream?language=zh",
            headers=auth_headers,
            json=payload,
        ) as response:
            assert response.status_code == 200
            for event in _iter_sse_payloads_from_lines(response.iter_lines()):
                event_type = str(event.get("type", "")).strip()
                if event_type == "stream_start":
                    saw_stream_start = True
                    assert str(event.get("turn_id", "")).strip() == turn_id
                    continue
                if event_type == "text_delta":
                    delta = str(event.get("delta", "")).strip()
                    if delta:
                        saw_text_delta_before_disconnect = True
                        break

        assert saw_stream_start, "Expected stream_start before disconnect"
        assert saw_text_delta_before_disconnect, "Expected at least one text_delta before disconnect"

        detail, stream_recovery, assistant_message = _wait_for_turn_completion(
            client,
            auth_headers,
            session_id=session_id,
            turn_id=turn_id,
            timeout_seconds=240,
        )

        assert str(stream_recovery.get("turn_id", "")).strip() == turn_id
        assert str(stream_recovery.get("state", "")).strip().lower() == "completed"
        assert str(stream_recovery.get("assistant_message_id", "")).strip() == str(
            assistant_message.get("id", "")
        ).strip()

        content = str(assistant_message.get("content", "")).strip()
        assert content, "Assistant content should be persisted after disconnect"

        # Ensure /sessions detail carries stream_recovery contract fields.
        assert isinstance(detail.get("stream_recovery"), dict)
        assert str(detail["stream_recovery"].get("started_at", "")).strip()
        assert str(detail["stream_recovery"].get("updated_at", "")).strip()
