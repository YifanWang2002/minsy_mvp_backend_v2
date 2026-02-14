from __future__ import annotations

import json
from collections.abc import AsyncIterator
from uuid import uuid4

import pytest

from src.agents.handler_registry import init_all_artifacts
from src.agents.orchestrator import ChatOrchestrator
from src.api.schemas.requests import ChatSendRequest
from src.engine.strategy import EXAMPLE_PATH, load_strategy_payload, upsert_strategy_dsl
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


class _SequencedStreamer:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.calls: list[dict[str, object]] = []
        self.turn = 0

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
                "previous_response_id": previous_response_id,
                "tools": tools,
                "tool_choice": tool_choice,
                "reasoning": reasoning,
            }
        )
        index = min(self.turn, len(self.outputs) - 1)
        self.turn += 1
        output = self.outputs[index]
        yield {"type": "response.output_text.delta", "delta": output, "sequence_number": 1}
        yield {
            "type": "response.completed",
            "response": {
                "id": f"resp_{uuid4().hex}",
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
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
async def test_strategy_phase_with_strategy_id_exposes_strategy_and_backtest_tools(db_session) -> None:
    user = User(email=f"orchestrator_rt_{uuid4().hex}@example.com", password_hash="hash", name="rt")
    db_session.add(user)
    await db_session.flush()

    artifacts = init_all_artifacts()
    artifacts["strategy"]["profile"] = {"strategy_id": str(uuid4())}
    artifacts["strategy"]["missing_fields"] = []

    session = Session(
        user_id=user.id,
        current_phase="strategy",
        status="active",
        artifacts=artifacts,
        metadata_={},
    )
    db_session.add(session)
    await db_session.flush()

    streamer = _CaptureStreamer()
    orchestrator = ChatOrchestrator(db_session)

    payload = ChatSendRequest(session_id=session.id, message="run first backtest and iterate")
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
async def test_strategy_phase_message_strategy_id_unlocks_artifact_tools(db_session) -> None:
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

    dsl_payload = load_strategy_payload(EXAMPLE_PATH)
    created = await upsert_strategy_dsl(
        db_session,
        session_id=session.id,
        dsl_payload=dsl_payload,
        auto_commit=False,
    )
    await db_session.commit()
    await db_session.refresh(session)
    strategy_id = str(created.strategy.id)

    streamer = _CaptureStreamer()
    orchestrator = ChatOrchestrator(db_session)

    payload = ChatSendRequest(
        session_id=session.id,
        message=f"Please update strategy_id {strategy_id} and rerun backtest.",
    )
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

    await db_session.refresh(session)
    strategy_profile = (
        (session.artifacts or {})
        .get("strategy", {})
        .get("profile", {})
    )
    assert strategy_profile.get("strategy_id") == strategy_id


@pytest.mark.asyncio
async def test_strategy_phase_ignores_foreign_strategy_id_in_message(db_session) -> None:
    user_a = User(email=f"orchestrator_rt_a_{uuid4().hex}@example.com", password_hash="hash", name="rt-a")
    user_b = User(email=f"orchestrator_rt_b_{uuid4().hex}@example.com", password_hash="hash", name="rt-b")
    db_session.add_all([user_a, user_b])
    await db_session.flush()

    session_a = Session(
        user_id=user_a.id,
        current_phase="strategy",
        status="active",
        artifacts=init_all_artifacts(),
        metadata_={},
    )
    session_b = Session(
        user_id=user_b.id,
        current_phase="strategy",
        status="active",
        artifacts=init_all_artifacts(),
        metadata_={},
    )
    db_session.add_all([session_a, session_b])
    await db_session.flush()

    dsl_payload = load_strategy_payload(EXAMPLE_PATH)
    created = await upsert_strategy_dsl(
        db_session,
        session_id=session_b.id,
        dsl_payload=dsl_payload,
        auto_commit=False,
    )
    await db_session.commit()
    foreign_strategy_id = str(created.strategy.id)

    streamer = _CaptureStreamer()
    orchestrator = ChatOrchestrator(db_session)

    payload = ChatSendRequest(
        session_id=session_a.id,
        message=f"Please patch strategy_id {foreign_strategy_id}.",
    )
    async for _ in orchestrator.handle_message_stream(user_a, payload, streamer, language="en"):
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
async def test_legacy_stress_test_session_is_redirected_to_strategy_phase(db_session) -> None:
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
    done_payload: dict[str, object] | None = None
    async for chunk in orchestrator.handle_message_stream(user, payload, streamer, language="en"):
        envelope = _parse_sse_payload(chunk)
        if not isinstance(envelope, dict):
            continue
        if envelope.get("type") == "done":
            done_payload = envelope

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
    assert isinstance(done_payload, dict)
    assert done_payload.get("phase") == "strategy"
    await db_session.refresh(session)
    assert session.current_phase == "strategy"
    strategy_profile = ((session.artifacts or {}).get("strategy", {}) or {}).get("profile", {})
    assert strategy_profile.get("strategy_id") == artifacts["stress_test"]["profile"]["strategy_id"]


@pytest.mark.asyncio
async def test_strategy_phase_auto_wraps_first_full_dsl_into_strategy_genui(db_session) -> None:
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

    dsl_payload = load_strategy_payload(EXAMPLE_PATH)
    dsl_text = json.dumps(dsl_payload, ensure_ascii=False)
    streamer = _SequencedStreamer([f"Draft generated.\n```json\n{dsl_text}\n```"])
    orchestrator = ChatOrchestrator(db_session)

    payload = ChatSendRequest(session_id=session.id, message="draft strategy dsl")
    genui_payloads: list[dict[str, object]] = []
    async for chunk in orchestrator.handle_message_stream(user, payload, streamer, language="en"):
        envelope = _parse_sse_payload(chunk)
        if not isinstance(envelope, dict):
            continue
        if envelope.get("type") != "genui":
            continue
        candidate = envelope.get("payload")
        if isinstance(candidate, dict):
            genui_payloads.append(candidate)

    assert genui_payloads
    strategy_card = next(
        item for item in genui_payloads if item.get("type") == "strategy_card"
    )
    assert isinstance(strategy_card.get("dsl_json"), dict)
    assert strategy_card.get("source") == "auto_detected_first_dsl"


@pytest.mark.asyncio
async def test_strategy_phase_auto_wraps_validate_draft_id_into_strategy_ref_genui(db_session) -> None:
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

    strategy_draft_id = str(uuid4())

    class _McpOnlyStreamer:
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
            del model, input_text, instructions, previous_response_id, tools, tool_choice, reasoning
            validate_output = {
                "category": "strategy",
                "tool": "strategy_validate_dsl",
                "ok": True,
                "errors": [],
                "dsl_version": "1.0.0",
                "strategy_draft_id": strategy_draft_id,
            }
            yield {
                "type": "response.output_item.done",
                "sequence_number": 1,
                "item": {
                    "type": "mcp_call",
                    "id": "call_strategy_validate",
                    "name": "strategy_validate_dsl",
                    "status": "completed",
                    "arguments": {"session_id": str(session.id)},
                    "output": json.dumps(validate_output),
                },
            }
            yield {
                "type": "response.completed",
                "response": {
                    "id": f"resp_{uuid4().hex}",
                    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                },
            }

    streamer = _McpOnlyStreamer()
    orchestrator = ChatOrchestrator(db_session)

    payload = ChatSendRequest(session_id=session.id, message="draft strategy dsl")
    genui_payloads: list[dict[str, object]] = []
    async for chunk in orchestrator.handle_message_stream(user, payload, streamer, language="en"):
        envelope = _parse_sse_payload(chunk)
        if not isinstance(envelope, dict):
            continue
        if envelope.get("type") != "genui":
            continue
        candidate = envelope.get("payload")
        if isinstance(candidate, dict):
            genui_payloads.append(candidate)

    assert genui_payloads
    strategy_ref = next(
        item for item in genui_payloads if item.get("type") == "strategy_ref"
    )
    assert strategy_ref.get("strategy_draft_id") == strategy_draft_id
    assert strategy_ref.get("source") == "strategy_validate_dsl"


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


@pytest.mark.asyncio
async def test_phase_transition_resets_previous_response_id(db_session) -> None:
    user = User(email=f"orchestrator_rt_{uuid4().hex}@example.com", password_hash="hash", name="rt")
    db_session.add(user)
    await db_session.flush()

    session = Session(
        user_id=user.id,
        current_phase="kyc",
        status="active",
        artifacts=init_all_artifacts(),
        metadata_={},
        previous_response_id="resp_should_be_cleared",
    )
    db_session.add(session)
    await db_session.flush()

    orchestrator = ChatOrchestrator(db_session)
    await orchestrator._transition_phase(
        session=session,
        to_phase="pre_strategy",
        trigger="test",
        metadata={"reason": "test_transition"},
    )

    assert session.current_phase == "pre_strategy"
    assert session.previous_response_id is None


@pytest.mark.asyncio
async def test_phase_carryover_memory_is_injected_once_after_transition(db_session) -> None:
    user = User(email=f"orchestrator_carry_{uuid4().hex}@example.com", password_hash="hash", name="rt")
    db_session.add(user)
    await db_session.flush()

    session = Session(
        user_id=user.id,
        current_phase="kyc",
        status="active",
        artifacts=init_all_artifacts(),
        metadata_={},
    )
    db_session.add(session)
    await db_session.flush()

    streamer = _SequencedStreamer(
        outputs=[
            (
                "已记录。"
                "<AGENT_STATE_PATCH>"
                '{"trading_years_bucket":"years_5_plus","risk_tolerance":"aggressive","return_expectation":"high_growth"}'
                "</AGENT_STATE_PATCH>"
            ),
            "继续告诉我你的交易市场。",
            "继续告诉我你的交易标的。",
        ]
    )
    orchestrator = ChatOrchestrator(db_session)

    payload1 = ChatSendRequest(
        session_id=session.id,
        message="我的经验 years_5_plus，风险 aggressive，收益 high_growth。",
    )
    async for _ in orchestrator.handle_message_stream(user, payload1, streamer, language="zh"):
        pass

    await db_session.refresh(session)
    assert session.current_phase == "pre_strategy"
    meta_after_transition = dict(session.metadata_ or {})
    carryover_meta = meta_after_transition.get("phase_carryover_memory")
    assert isinstance(carryover_meta, dict)
    assert carryover_meta.get("target_phase") == "pre_strategy"

    payload2 = ChatSendRequest(session_id=session.id, message="目标市场 target_market=us_stocks。")
    async for _ in orchestrator.handle_message_stream(user, payload2, streamer, language="zh"):
        pass

    assert len(streamer.calls) >= 2
    second_input_text = str(streamer.calls[1].get("input_text", ""))
    assert "[PHASE CARRYOVER MEMORY]" in second_input_text
    assert "from_phase: kyc" in second_input_text
    assert "我的经验 years_5_plus" in second_input_text

    await db_session.refresh(session)
    meta_after_consume = dict(session.metadata_ or {})
    assert "phase_carryover_memory" not in meta_after_consume

    payload3 = ChatSendRequest(session_id=session.id, message="继续。")
    async for _ in orchestrator.handle_message_stream(user, payload3, streamer, language="zh"):
        pass

    assert len(streamer.calls) >= 3
    third_input_text = str(streamer.calls[2].get("input_text", ""))
    assert "[PHASE CARRYOVER MEMORY]" not in third_input_text
