from __future__ import annotations

from src.util.chat_debug_trace import (
    build_chat_debug_trace,
    get_chat_debug_trace,
    reset_chat_debug_trace,
    set_chat_debug_trace,
)


def test_trace_default_disabled_without_header() -> None:
    trace = build_chat_debug_trace(
        default_enabled=False,
        header_value=None,
        requested_trace_id=None,
    )
    assert trace.enabled is False
    assert trace.trace_id is None


def test_trace_header_can_enable_when_default_disabled() -> None:
    trace = build_chat_debug_trace(
        default_enabled=False,
        header_value="1",
        requested_trace_id="trace_abc_123",
    )
    assert trace.enabled is True
    assert trace.trace_id == "trace_abc_123"


def test_trace_header_can_disable_when_default_enabled() -> None:
    trace = build_chat_debug_trace(
        default_enabled=True,
        header_value="off",
        requested_trace_id="trace_from_header",
    )
    assert trace.enabled is False
    assert trace.trace_id is None


def test_invalid_trace_id_falls_back_to_generated_value() -> None:
    trace = build_chat_debug_trace(
        default_enabled=True,
        header_value=None,
        requested_trace_id="invalid trace id with spaces",
    )
    assert trace.enabled is True
    assert isinstance(trace.trace_id, str)
    assert trace.trace_id.startswith("trace_")


def test_set_and_reset_trace_context() -> None:
    trace = build_chat_debug_trace(
        default_enabled=True,
        header_value="1",
        requested_trace_id="trace_context_test",
    )
    token = set_chat_debug_trace(trace)
    try:
        current = get_chat_debug_trace()
        assert current is not None
        assert current.trace_id == "trace_context_test"
    finally:
        reset_chat_debug_trace(token)
    assert get_chat_debug_trace() is None
