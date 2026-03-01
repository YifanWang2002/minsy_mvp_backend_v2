from __future__ import annotations

from uuid import uuid4

from apps.api.orchestration.constants import (
    _INSTRUCTION_CONTEXT_META_KEY,
    _INSTRUCTION_REFRESH_EVERY_PHASE_TURNS,
)
from apps.api.orchestration.stream_handler import StreamHandlerMixin
from packages.infra.db.models.session import Session


class _DummyStreamHandler(StreamHandlerMixin):
    pass


def _build_session(*, previous_response_id: str | None, metadata: dict | None = None) -> Session:
    return Session(
        user_id=uuid4(),
        current_phase="strategy",
        status="active",
        previous_response_id=previous_response_id,
        artifacts={},
        metadata_=metadata or {},
    )


def test_000_instruction_context_roundtrip() -> None:
    handler = _DummyStreamHandler()
    session = _build_session(previous_response_id="resp_123")

    handler._write_instruction_context(
        session=session,
        phase="strategy",
        phase_stage=" schema_only ",
        phase_turn_count=3,
        instructions_sent=True,
    )

    phase, stage = handler._read_instruction_context(session)
    assert phase == "strategy"
    assert stage == "schema_only"
    assert session.metadata_[_INSTRUCTION_CONTEXT_META_KEY]["phase_turn_count"] == 3
    assert session.metadata_[_INSTRUCTION_CONTEXT_META_KEY]["instructions_sent"] is True


def test_010_should_send_instructions_for_new_chain() -> None:
    handler = _DummyStreamHandler()
    session = _build_session(previous_response_id=None)

    assert handler._should_send_instructions(
        session=session,
        phase="strategy",
        phase_stage="schema_only",
        phase_turn_count=1,
    )


def test_020_should_not_send_when_stage_unchanged() -> None:
    handler = _DummyStreamHandler()
    session = _build_session(previous_response_id="resp_abc")
    handler._write_instruction_context(
        session=session,
        phase="strategy",
        phase_stage="schema_only",
        phase_turn_count=1,
        instructions_sent=True,
    )

    assert not handler._should_send_instructions(
        session=session,
        phase="strategy",
        phase_stage="schema_only",
        phase_turn_count=2,
    )


def test_030_should_send_when_stage_changes() -> None:
    handler = _DummyStreamHandler()
    session = _build_session(previous_response_id="resp_abc")
    handler._write_instruction_context(
        session=session,
        phase="strategy",
        phase_stage="schema_only",
        phase_turn_count=1,
        instructions_sent=True,
    )

    assert handler._should_send_instructions(
        session=session,
        phase="strategy",
        phase_stage="artifact_ops",
        phase_turn_count=3,
    )


def test_040_should_send_on_periodic_refresh() -> None:
    handler = _DummyStreamHandler()
    session = _build_session(previous_response_id="resp_abc")
    handler._write_instruction_context(
        session=session,
        phase="strategy",
        phase_stage="artifact_ops",
        phase_turn_count=_INSTRUCTION_REFRESH_EVERY_PHASE_TURNS - 1,
        instructions_sent=False,
    )

    assert handler._should_send_instructions(
        session=session,
        phase="strategy",
        phase_stage="artifact_ops",
        phase_turn_count=_INSTRUCTION_REFRESH_EVERY_PHASE_TURNS,
    )
