"""Orchestrator mixin extracted from legacy implementation."""

from __future__ import annotations

from .shared import *  # noqa: F403


class StreamHandlerMixin:
    async def _stream_openai_and_collect(
        self,
        *,
        session: Session,
        streamer: ResponsesEventStreamer,
        preparation: _TurnPreparation,
        stream_state: _TurnStreamState,
    ) -> AsyncIterator[str]:
        resolved_model = preparation.prompt.model or settings.openai_response_model
        stream_request_kwargs: dict[str, Any] = {
            "model": resolved_model,
            "input_text": preparation.prompt.enriched_input,
            "instructions": preparation.prompt.instructions,
            "previous_response_id": session.previous_response_id,
            "tools": preparation.tools,
            "tool_choice": preparation.prompt.tool_choice,
            "reasoning": preparation.prompt.reasoning,
        }
        stream_state.request_model = resolved_model
        trace_stream_request_kwargs = self._redact_stream_request_kwargs_for_trace(
            stream_request_kwargs
        )
        record_chat_debug_trace(
            "orchestrator_to_openai_request",
            {
                "session_id": str(session.id),
                "phase": preparation.phase_before,
                **trace_stream_request_kwargs,
            },
        )
        trace = get_chat_debug_trace()
        compact_trace = (
            trace is not None
            and trace.enabled
            and isinstance(trace.mode, str)
            and trace.mode.strip().lower() == CHAT_TRACE_MODE_COMPACT
        )
        try:
            timeout_seconds = self._resolve_stream_timeout_seconds()
            async with asyncio.timeout(timeout_seconds):
                async for event in streamer.stream_events(**stream_request_kwargs):
                    event_type = str(event.get("type", "unknown"))
                    if not (compact_trace and self._is_delta_trace_token(event_type)):
                        record_chat_debug_trace(
                            "openai_to_orchestrator_event",
                            {
                                "session_id": str(session.id),
                                "phase": preparation.phase_before,
                                "event": event,
                            },
                        )
                    seq = event.get("sequence_number")
                    stream_state.mcp_fallback_counter = self._collect_mcp_records_from_event(
                        event_type=event_type,
                        event=event,
                        sequence_number=seq,
                        records=stream_state.mcp_call_records,
                        order=stream_state.mcp_call_order,
                        fallback_counter=stream_state.mcp_fallback_counter,
                    )
                    yield self._sse(
                        "openai_event",
                        {
                            "type": "openai_event",
                            "openai_type": event_type,
                            "sequence_number": seq,
                            "payload": event,
                        },
                    )
                    if event_type == "response.stream_error":
                        err = event.get("error")
                        if isinstance(err, dict):
                            stream_state.stream_error_detail = copy.deepcopy(err)
                            msg = err.get("message")
                            if isinstance(msg, str) and msg.strip():
                                stream_state.stream_error_message = msg.strip()
                            if not stream_state.stream_error_message:
                                diagnostics = err.get("diagnostics")
                                if isinstance(diagnostics, dict):
                                    fallback_message = diagnostics.get("upstream_message")
                                    if (
                                        isinstance(fallback_message, str)
                                        and fallback_message.strip()
                                    ):
                                        stream_state.stream_error_message = (
                                            fallback_message.strip()
                                        )
                        if not stream_state.stream_error_message:
                            stream_state.stream_error_message = (
                                "Upstream stream interrupted."
                            )
                        continue
                    if event_type == "response.output_text.delta":
                        delta = event.get("delta")
                        if isinstance(delta, str) and delta:
                            stream_state.full_text += delta
                            yield self._sse(
                                "stream",
                                {
                                    "type": "text_delta",
                                    "delta": delta,
                                    "sequence_number": seq,
                                },
                            )
                            stream_state.text_delta_emitted = True
                    elif event_type == "response.output_text.done":
                        done_text = event.get("text")
                        if (
                            isinstance(done_text, str)
                            and done_text
                            and not stream_state.full_text.strip()
                        ):
                            stream_state.full_text = done_text
                            yield self._sse(
                                "stream",
                                {
                                    "type": "text_delta",
                                    "delta": done_text,
                                    "sequence_number": seq,
                                },
                            )
                            stream_state.text_delta_emitted = True
                    if "mcp_" in event_type:
                        yield self._sse(
                            "stream",
                            {
                                "type": "mcp_event",
                                "openai_type": event_type,
                                "sequence_number": seq,
                                "payload": event,
                            },
                        )
                    if event_type == "response.completed":
                        response_obj = event.get("response")
                        if isinstance(response_obj, dict):
                            usage = response_obj.get("usage")
                            if isinstance(usage, dict):
                                stream_state.completed_usage = usage
                            response_model = response_obj.get("model")
                            if isinstance(response_model, str) and response_model.strip():
                                stream_state.completed_model = response_model.strip()
                            resp_id = response_obj.get("id")
                            if isinstance(resp_id, str) and resp_id:
                                session.previous_response_id = resp_id
        except TimeoutError:
            stream_state.stream_error_message = (
                "OpenAI stream timed out before completion. "
                f"(>{int(timeout_seconds)}s)"
            )
            stream_state.stream_error_detail = {
                "class": "TimeoutError",
                "message": stream_state.stream_error_message,
                "diagnostics": {"category": "orchestrator_stream_timeout"},
            }
        except Exception as exc:  # noqa: BLE001
            stream_state.stream_error_message = f"{type(exc).__name__}: {exc}"
            stream_state.stream_error_detail = {
                "class": type(exc).__name__,
                "message": str(exc),
                "diagnostics": {"category": "orchestrator_stream_runtime_error"},
            }

    @staticmethod
    def _resolve_stream_timeout_seconds() -> float:
        # Keep package-level monkeypatch hooks working after package split.
        from apps.api import orchestration as orchestrator_mod

        raw_value = getattr(
            orchestrator_mod,
            "_OPENAI_STREAM_HARD_TIMEOUT_SECONDS",
            _OPENAI_STREAM_HARD_TIMEOUT_SECONDS,
        )
        try:
            timeout_seconds = float(raw_value)
        except (TypeError, ValueError):
            return _OPENAI_STREAM_HARD_TIMEOUT_SECONDS
        if timeout_seconds <= 0:
            return _OPENAI_STREAM_HARD_TIMEOUT_SECONDS
        return timeout_seconds

    def _sse(self, event: str, payload: dict[str, Any]) -> str:
        trace = get_chat_debug_trace()
        compact_trace = (
            trace is not None
            and trace.enabled
            and isinstance(trace.mode, str)
            and trace.mode.strip().lower() == CHAT_TRACE_MODE_COMPACT
        )
        if not (
            compact_trace
            and (
                self._is_delta_trace_token(event)
                or self._is_delta_trace_token(payload.get("type"))
                or self._is_delta_trace_token(payload.get("openai_type"))
                or (
                    isinstance(payload.get("payload"), dict)
                    and self._is_delta_trace_token(payload.get("payload", {}).get("type"))
                )
            )
        ):
            record_chat_debug_trace(
                "orchestrator_to_frontend_sse",
                {
                    "event": event,
                    "payload": payload,
                },
            )
        return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

    @staticmethod
    def _is_delta_trace_token(value: Any) -> bool:
        if not isinstance(value, str):
            return False
        return "delta" in value.strip().lower()
