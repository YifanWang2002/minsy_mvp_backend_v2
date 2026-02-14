from __future__ import annotations

import json
from collections.abc import AsyncIterator
from uuid import uuid4

import pytest

from src.agents.handler_registry import init_all_artifacts
from src.agents.orchestrator import ChatOrchestrator
from src.api.schemas.requests import ChatSendRequest
from src.models.session import Session
from src.models.user import User


class _CaptureStreamer:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def stream_events(
        self,
        *,
        model: str,
        input_text: str,
        instructions: str | None = None,
        previous_response_id: str | None = None,
        tools: list[dict[str, object]] | None = None,
        tool_choice: dict[str, object] | None = None,
        reasoning: dict[str, object] | None = None,
    ) -> AsyncIterator[dict[str, object]]:
        self.calls.append(
            {
                "model": model,
                "input_text": input_text,
                "instructions": instructions,
                "tools": tools,
                "tool_choice": tool_choice,
                "reasoning": reasoning,
            }
        )
        yield {"type": "response.output_text.delta", "delta": "ok", "sequence_number": 1}
        yield {
            "type": "response.completed",
            "response": {
                "id": f"resp_{uuid4().hex}",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        }


def _parse_sse_payload(chunk: str) -> dict[str, object] | None:
    data_lines = [
        line.removeprefix("data:").lstrip()
        for line in chunk.splitlines()
        if line.startswith("data:")
    ]
    if not data_lines:
        return None
    raw = "\n".join(data_lines).strip()
    if not raw:
        return None
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        return None
    return parsed


@pytest.mark.asyncio
async def test_strategy_phase_auto_policy_limits_tools_to_validation(db_session) -> None:
    user = User(email=f"orchestrator_rt_{uuid4().hex}@example.com", password_hash="hash", name="rt")
    db_session.add(user)
    await db_session.flush()

    session = Session(
        user_id=user.id,
        current_phase="strategy",
        status="active",
        artifacts=init_all_artifacts(),
        metadata_={},
    )
    db_session.add(session)
    await db_session.flush()

    streamer = _CaptureStreamer()
    orchestrator = ChatOrchestrator(db_session)

    payload = ChatSendRequest(session_id=session.id, message="please draft schema")
    async for _ in orchestrator.handle_message_stream(user, payload, streamer, language="en"):
        pass

    assert streamer.calls
    tools = streamer.calls[0]["tools"]
    assert isinstance(tools, list) and tools

    strategy_tools = next(
        tool
        for tool in tools
        if isinstance(tool, dict) and tool.get("server_label") == "strategy"
    )
    assert strategy_tools["allowed_tools"] == ["strategy_validate_dsl"]


@pytest.mark.asyncio
async def test_stress_test_feedback_stage_exposes_strategy_and_backtest_tools(db_session) -> None:
    user = User(email=f"orchestrator_rt_{uuid4().hex}@example.com", password_hash="hash", name="rt")
    db_session.add(user)
    await db_session.flush()

    artifacts = init_all_artifacts()
    artifacts["stress_test"]["profile"] = {
        "strategy_id": str(uuid4()),
        "backtest_job_id": str(uuid4()),
        "backtest_status": "done",
    }
    artifacts["stress_test"]["missing_fields"] = []

    session = Session(
        user_id=user.id,
        current_phase="stress_test",
        status="active",
        artifacts=artifacts,
        metadata_={},
    )
    db_session.add(session)
    await db_session.flush()

    streamer = _CaptureStreamer()
    orchestrator = ChatOrchestrator(db_session)

    payload = ChatSendRequest(session_id=session.id, message="analyze result and improve")
    async for _ in orchestrator.handle_message_stream(user, payload, streamer, language="en"):
        pass

    assert streamer.calls
    tools = streamer.calls[0]["tools"]
    assert isinstance(tools, list) and tools

    labels = {
        tool.get("server_label")
        for tool in tools
        if isinstance(tool, dict)
    }
    assert "strategy" in labels
    assert "backtest" in labels


@pytest.mark.asyncio
async def test_stop_criteria_placeholder_emits_hint_after_ten_turns(db_session) -> None:
    user = User(email=f"orchestrator_rt_{uuid4().hex}@example.com", password_hash="hash", name="rt")
    db_session.add(user)
    await db_session.flush()

    session = Session(
        user_id=user.id,
        current_phase="strategy",
        status="active",
        artifacts=init_all_artifacts(),
        metadata_={},
    )
    db_session.add(session)
    await db_session.flush()

    streamer = _CaptureStreamer()
    orchestrator = ChatOrchestrator(db_session)

    seen_stop_hint = False
    for idx in range(10):
        payload = ChatSendRequest(session_id=session.id, message=f"turn-{idx}")
        async for chunk in orchestrator.handle_message_stream(user, payload, streamer, language="en"):
            envelope = _parse_sse_payload(chunk)
            if not isinstance(envelope, dict):
                continue
            if envelope.get("type") != "text_delta":
                continue
            delta = envelope.get("delta")
            if isinstance(delta, str) and "many iterations" in delta:
                seen_stop_hint = True

    await db_session.refresh(session)
    stop_meta = (session.metadata_ or {}).get("stop_criteria_placeholder", {})
    assert seen_stop_hint is True
    assert isinstance(stop_meta, dict)
    assert stop_meta.get("enabled") is True
