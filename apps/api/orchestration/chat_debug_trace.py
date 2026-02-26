"""Request-scoped chat tracing helpers for end-to-end debugging."""

from __future__ import annotations

import contextvars
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from packages.infra.observability.logger import logger

CHAT_TRACE_HEADER_ENABLED = "x-minsy-debug-trace"
CHAT_TRACE_HEADER_ID = "x-minsy-debug-trace-id"
CHAT_TRACE_HEADER_MODE = "x-minsy-debug-trace-mode"
CHAT_TRACE_RESPONSE_HEADER_ID = "X-Minsy-Debug-Trace-Id"
CHAT_TRACE_MODE_VERBOSE = "verbose"
CHAT_TRACE_MODE_COMPACT = "compact"

_TRUE_VALUES = {"1", "true", "on", "yes", "y"}
_FALSE_VALUES = {"0", "false", "off", "no", "n"}
_TRACE_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,96}$")

_trace_ctx: contextvars.ContextVar[ChatDebugTrace | None] = contextvars.ContextVar(
    "chat_debug_trace",
    default=None,
)


def _parse_bool_override(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if not normalized:
        return None
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    return None


def _sanitize_trace_id(value: str | None) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized:
        return None
    if _TRACE_ID_PATTERN.match(normalized) is None:
        return None
    return normalized


def _parse_trace_mode(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized == CHAT_TRACE_MODE_VERBOSE:
        return CHAT_TRACE_MODE_VERBOSE
    if normalized == CHAT_TRACE_MODE_COMPACT:
        return CHAT_TRACE_MODE_COMPACT
    return None


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value

    if isinstance(value, UUID):
        return str(value)

    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()

    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}

    if isinstance(value, list | tuple | set):
        return [_to_jsonable(item) for item in value]

    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump(mode="json", exclude_none=True, warnings=False)
            return _to_jsonable(dumped)
        except Exception:  # noqa: BLE001
            return str(value)

    return str(value)


@dataclass(slots=True, frozen=True)
class ChatDebugTrace:
    enabled: bool
    trace_id: str | None = None
    mode: str = CHAT_TRACE_MODE_VERBOSE

    def record(self, *, stage: str, payload: dict[str, Any] | None = None) -> None:
        if not self.enabled or not self.trace_id:
            return
        safe_payload = _to_jsonable(payload or {})
        payload_text = json.dumps(safe_payload, ensure_ascii=False, separators=(",", ":"))
        logger.info(
            "chat_trace trace_id=%s stage=%s at=%s payload=%s",
            self.trace_id,
            stage,
            datetime.now(UTC).isoformat(),
            payload_text,
            extra={"markup": False},
        )


def build_chat_debug_trace(
    *,
    default_enabled: bool,
    default_mode: str,
    header_value: str | None,
    requested_trace_id: str | None,
    requested_mode: str | None,
) -> ChatDebugTrace:
    resolved_mode = _parse_trace_mode(requested_mode) or _parse_trace_mode(default_mode)
    if resolved_mode is None:
        resolved_mode = CHAT_TRACE_MODE_VERBOSE

    override = _parse_bool_override(header_value)
    enabled = default_enabled if override is None else override
    if not enabled:
        return ChatDebugTrace(enabled=False, mode=resolved_mode)

    trace_id = _sanitize_trace_id(requested_trace_id) or f"trace_{uuid4().hex}"
    return ChatDebugTrace(enabled=True, trace_id=trace_id, mode=resolved_mode)


def set_chat_debug_trace(trace: ChatDebugTrace | None) -> contextvars.Token[ChatDebugTrace | None]:
    if trace is None or not trace.enabled:
        return _trace_ctx.set(None)
    return _trace_ctx.set(trace)


def reset_chat_debug_trace(token: contextvars.Token[ChatDebugTrace | None]) -> None:
    _trace_ctx.reset(token)


def get_chat_debug_trace() -> ChatDebugTrace | None:
    return _trace_ctx.get()


def record_chat_debug_trace(stage: str, payload: dict[str, Any] | None = None) -> None:
    trace = get_chat_debug_trace()
    if trace is None:
        return
    trace.record(stage=stage, payload=payload)
