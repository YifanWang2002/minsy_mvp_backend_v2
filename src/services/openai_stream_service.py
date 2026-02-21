"""OpenAI Responses API streaming helper for agent conversational responses."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any, Protocol

import httpx
from openai import APIConnectionError, APIError, APITimeoutError, AsyncOpenAI

from src.config import settings
from src.observability.sentry_setup import capture_exception_with_context
from src.util.logger import logger


def _to_jsonable(value: Any) -> Any:
    """Recursively normalize arbitrary values into JSON-serializable shapes."""
    if value is None or isinstance(value, str | int | float | bool):
        return value

    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}

    if isinstance(value, list | tuple | set):
        return [_to_jsonable(item) for item in value]

    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump(mode="json", exclude_none=True, warnings=False)
            return _to_jsonable(dumped)
        except Exception:  # noqa: BLE001
            return {"type": str(getattr(value, "type", "unknown"))}

    if hasattr(value, "isoformat"):
        try:
            return str(value.isoformat())
        except Exception:  # noqa: BLE001
            return str(value)

    return str(value)


def _coerce_event_payload(event: Any) -> dict[str, Any]:
    """Convert SDK stream event into a plain serializable dict payload."""
    raw_payload = _to_jsonable(event)
    if isinstance(raw_payload, dict):
        payload = raw_payload
    else:
        payload = {"type": str(getattr(event, "type", "unknown"))}

    event_type = payload.get("type")
    if not isinstance(event_type, str) or not event_type:
        payload["type"] = str(getattr(event, "type", "unknown"))
    return payload


def _is_setup_only_event(event_type: str) -> bool:
    return event_type in {"response.created", "response.in_progress"}


def _is_text_output_event(event_type: str) -> bool:
    return event_type in {"response.output_text.delta", "response.output_text.done"}


def _truncate_text(value: Any, *, max_chars: int) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}...<truncated>"


def _extract_status_code(exc: Exception) -> int | None:
    raw = getattr(exc, "status_code", None)
    if isinstance(raw, int):
        return raw
    return None


def _extract_request_id(exc: Exception) -> str | None:
    raw = getattr(exc, "request_id", None)
    if isinstance(raw, str) and raw.strip():
        return raw.strip()

    response = getattr(exc, "response", None)
    if isinstance(response, httpx.Response):
        value = response.headers.get("x-request-id")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _iter_error_nodes(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, dict):
        return []

    nodes: list[dict[str, Any]] = []
    seen: set[int] = set()
    current: Any = value
    for _ in range(5):
        if not isinstance(current, dict):
            break
        marker = id(current)
        if marker in seen:
            break
        seen.add(marker)
        nodes.append(current)
        nested = current.get("error")
        if isinstance(nested, dict):
            current = nested
            continue
        break
    return nodes


def _first_non_empty_field(nodes: list[dict[str, Any]], *keys: str) -> str | None:
    for node in reversed(nodes):
        for key in keys:
            if key not in node:
                continue
            value = _truncate_text(node.get(key), max_chars=700)
            if value is not None:
                return value
    return None


def _normalize_error_body(exc: Exception) -> Any:
    raw = getattr(exc, "body", None)
    if raw is None:
        return None
    return _to_jsonable(raw)


def _summarize_exception_text(*, exc: Exception, upstream_message: str | None) -> str:
    base = _truncate_text(str(exc), max_chars=800) or "Upstream stream interrupted."
    if (
        upstream_message
        and (
            base.lower().startswith("error code:")
            or base.lower() in {"connection error.", "request timed out."}
        )
    ):
        return upstream_message
    return upstream_message or base


def _is_mcp_list_tools_error_text(text: str) -> bool:
    lowered = text.lower()
    if "mcp" not in lowered:
        return False
    return any(
        marker in lowered
        for marker in (
            "tool list",
            "list tools",
            "mcp_list_tools",
            "retrieving tool list",
        )
    )


def _classify_stream_error(
    *,
    exc: Exception,
    combined_text: str,
    status_code: int | None,
) -> str:
    if _is_retryable_stream_error(exc):
        return "network"
    if _is_mcp_list_tools_error_text(combined_text):
        return "mcp_list_tools_fetch_failed"
    if isinstance(exc, APIError):
        if status_code == 429:
            return "upstream_rate_limit"
        if isinstance(status_code, int) and status_code >= 500:
            return "upstream_server_error"
        if isinstance(status_code, int) and status_code >= 400:
            return "upstream_client_error"
        return "upstream_api_error"
    return "unknown"


def _build_stream_error_diagnostics(exc: Exception) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    status_code = _extract_status_code(exc)
    request_id = _extract_request_id(exc)
    body = _normalize_error_body(exc)
    nodes = _iter_error_nodes(body)

    upstream_message = _first_non_empty_field(nodes, "message", "detail", "description")
    upstream_code = _first_non_empty_field(nodes, "code")
    upstream_type = _first_non_empty_field(nodes, "type")
    upstream_param = _first_non_empty_field(nodes, "param")

    parts = [str(exc)]
    if upstream_message:
        parts.append(upstream_message)
    if upstream_code:
        parts.append(upstream_code)
    if upstream_type:
        parts.append(upstream_type)
    combined_text = " | ".join(part for part in parts if part).strip()
    category = _classify_stream_error(
        exc=exc,
        combined_text=combined_text,
        status_code=status_code,
    )
    diagnostics["category"] = category

    if status_code is not None:
        diagnostics["status_code"] = status_code
    if request_id is not None:
        diagnostics["request_id"] = request_id
    if upstream_message is not None:
        diagnostics["upstream_message"] = upstream_message
    if upstream_code is not None:
        diagnostics["upstream_error_code"] = upstream_code
    if upstream_type is not None:
        diagnostics["upstream_error_type"] = upstream_type
    if upstream_param is not None:
        diagnostics["upstream_error_param"] = upstream_param

    if isinstance(exc, APIError):
        if isinstance(exc.code, str) and exc.code.strip():
            diagnostics.setdefault("upstream_error_code", exc.code.strip())
        if isinstance(exc.type, str) and exc.type.strip():
            diagnostics.setdefault("upstream_error_type", exc.type.strip())
        if isinstance(exc.param, str) and exc.param.strip():
            diagnostics.setdefault("upstream_error_param", exc.param.strip())

    if body is not None:
        try:
            raw = json.dumps(body, ensure_ascii=False, default=str)
        except TypeError:
            raw = str(body)
        snippet = _truncate_text(raw, max_chars=1200)
        if snippet is not None:
            diagnostics["upstream_body_excerpt"] = snippet

    return diagnostics


def _build_stream_error_payload(
    *,
    exc: Exception,
    retryable: bool,
    attempt: int,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    upstream_message = diagnostics.get("upstream_message")
    preferred_message = _summarize_exception_text(
        exc=exc,
        upstream_message=upstream_message if isinstance(upstream_message, str) else None,
    )
    status_code = diagnostics.get("status_code")
    if (
        isinstance(status_code, int)
        and f"{status_code}" not in preferred_message
        and diagnostics.get("category") != "network"
    ):
        preferred_message = f"{preferred_message} (status={status_code})"

    payload: dict[str, Any] = {
        "class": type(exc).__name__,
        "message": preferred_message,
        "retryable": retryable,
        "attempt": attempt,
    }
    if diagnostics:
        payload["diagnostics"] = diagnostics
    return payload


def _build_stream_kwargs(
    *,
    model: str,
    input_text: str,
    instructions: str | None,
    previous_response_id: str | None,
    tools: list[dict[str, Any]] | None,
    tool_choice: dict[str, Any] | None,
    reasoning: dict[str, Any] | None,
) -> dict[str, Any]:
    stream_kwargs: dict[str, Any] = {
        "model": model,
        "input": input_text,
    }
    optional_fields: dict[str, Any] = {
        "instructions": instructions,
        "previous_response_id": previous_response_id,
        "tools": tools,
        "tool_choice": tool_choice,
        "reasoning": reasoning,
    }
    for key, value in optional_fields.items():
        if value is not None:
            stream_kwargs[key] = value
    return stream_kwargs


def _should_retry_stream_error(
    *,
    retryable: bool,
    saw_text_output_event: bool,
    attempt: int,
    max_attempts: int,
) -> bool:
    # If user-facing text has not started yet, retry transient failures.
    return retryable and not saw_text_output_event and attempt < max_attempts


class ResponsesEventStreamer(Protocol):
    """Protocol for streaming raw Responses API events."""

    async def stream_events(
        self,
        *,
        model: str,
        input_text: str,
        instructions: str | None = None,
        previous_response_id: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | None = None,
        reasoning: dict[str, Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield response stream events as plain dicts."""


class OpenAIResponsesEventStreamer:
    """Stream raw OpenAI Responses API events."""

    def __init__(self) -> None:
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)

    async def stream_events(
        self,
        *,
        model: str,
        input_text: str,
        instructions: str | None = None,
        previous_response_id: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | None = None,
        reasoning: dict[str, Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        stream_kwargs = _build_stream_kwargs(
            model=model,
            input_text=input_text,
            instructions=instructions,
            previous_response_id=previous_response_id,
            tools=tools,
            tool_choice=tool_choice,
            reasoning=reasoning,
        )

        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            saw_text_output_event = False
            try:
                async with self.client.responses.stream(**stream_kwargs) as stream:
                    async for event in stream:
                        payload = _coerce_event_payload(event)
                        event_type = str(payload.get("type", "unknown"))
                        if _is_text_output_event(event_type):
                            saw_text_output_event = True
                        yield payload
                return
            except Exception as exc:  # noqa: BLE001
                diagnostics = _build_stream_error_diagnostics(exc)
                retryable_network = _is_retryable_stream_error(exc)
                retryable_mcp = _is_retryable_mcp_listing_error(
                    exc,
                    diagnostics=diagnostics,
                )
                retryable = retryable_network or retryable_mcp
                # If user-facing text has not started yet, retry both transient network
                # failures and MCP list-tools fetch failures even if setup events appeared.
                can_retry = _should_retry_stream_error(
                    retryable=retryable,
                    saw_text_output_event=saw_text_output_event,
                    attempt=attempt,
                    max_attempts=max_attempts,
                )
                if can_retry:
                    await asyncio.sleep(0.35 * attempt)
                    continue

                error_payload = _build_stream_error_payload(
                    exc=exc,
                    retryable=retryable,
                    attempt=attempt,
                    diagnostics=diagnostics,
                )
                logger.warning(
                    (
                        "openai.stream_error class=%s category=%s retryable=%s "
                        "attempt=%s status_code=%s request_id=%s message=%s"
                    ),
                    type(exc).__name__,
                    diagnostics.get("category"),
                    retryable,
                    attempt,
                    diagnostics.get("status_code"),
                    diagnostics.get("request_id"),
                    error_payload.get("message"),
                )
                capture_exception_with_context(
                    exc,
                    tags={
                        "source": "openai_stream",
                        "category": diagnostics.get("category"),
                        "status_code": diagnostics.get("status_code"),
                        "request_id": diagnostics.get("request_id"),
                    },
                    extras={
                        "attempt": attempt,
                        "retryable": retryable,
                        "diagnostics": diagnostics,
                        "error_payload": error_payload,
                    },
                )
                yield {
                    "type": "response.stream_error",
                    "error": error_payload,
                }
                return


def _is_retryable_stream_error(exc: Exception) -> bool:
    if isinstance(
        exc,
        (
            APIConnectionError,
            APITimeoutError,
            httpx.RemoteProtocolError,
            httpx.ReadError,
            httpx.ReadTimeout,
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.NetworkError,
        ),
    ):
        return True

    text = str(exc).lower()
    return any(
        keyword in text
        for keyword in (
            "incomplete chunked read",
            "peer closed connection",
            "connection reset",
            "connection aborted",
        )
    )


def _is_retryable_mcp_listing_error(
    exc: Exception,
    *,
    diagnostics: dict[str, Any] | None = None,
) -> bool:
    if not isinstance(exc, APIError):
        return False

    resolved_diagnostics = diagnostics or _build_stream_error_diagnostics(exc)
    if resolved_diagnostics.get("category") != "mcp_list_tools_fetch_failed":
        return False

    status_code = _extract_status_code(exc)
    if status_code is None:
        return True
    return status_code in {408, 409, 424, 429, 500, 502, 503, 504}
