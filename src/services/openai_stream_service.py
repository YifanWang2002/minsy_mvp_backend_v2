"""OpenAI Responses API streaming helper for agent conversational responses."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any, Protocol

import httpx
from openai import APIConnectionError, APIError, APITimeoutError, AsyncOpenAI

from src.config import settings


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
        stream_kwargs: dict[str, Any] = {
            "model": model,
            "input": input_text,
        }
        if instructions is not None:
            stream_kwargs["instructions"] = instructions
        if previous_response_id is not None:
            stream_kwargs["previous_response_id"] = previous_response_id
        if tools is not None:
            stream_kwargs["tools"] = tools
        if tool_choice is not None:
            stream_kwargs["tool_choice"] = tool_choice
        if reasoning is not None:
            stream_kwargs["reasoning"] = reasoning

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
                retryable_network = _is_retryable_stream_error(exc)
                retryable_mcp = _is_retryable_mcp_listing_error(exc)
                retryable = retryable_network or retryable_mcp
                # If user-facing text has not started yet, retry both transient network
                # failures and MCP list-tools fetch failures even if setup events appeared.
                can_retry = retryable and not saw_text_output_event and attempt < max_attempts
                if can_retry:
                    await asyncio.sleep(0.35 * attempt)
                    continue

                yield {
                    "type": "response.stream_error",
                    "error": {
                        "class": type(exc).__name__,
                        "message": str(exc),
                        "retryable": retryable,
                        "attempt": attempt,
                    },
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


def _is_retryable_mcp_listing_error(exc: Exception) -> bool:
    if not isinstance(exc, APIError):
        return False
    text = str(exc).lower()
    if "error retrieving tool list from mcp server" not in text:
        return False

    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        return True
    return status_code in {408, 409, 424, 429, 500, 502, 503, 504}
