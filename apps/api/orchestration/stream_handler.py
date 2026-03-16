"""Orchestrator mixin extracted from legacy implementation."""

from __future__ import annotations

import base64

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
        resolved_reasoning = (
            dict(preparation.prompt.reasoning)
            if isinstance(preparation.prompt.reasoning, dict)
            else None
        )
        resolved_max_output_tokens = (
            int(preparation.prompt.max_output_tokens)
            if isinstance(preparation.prompt.max_output_tokens, int)
            and preparation.prompt.max_output_tokens > 0
            else None
        )
        resolved_response_verbosity = (
            preparation.prompt.response_verbosity.strip().lower()
            if isinstance(preparation.prompt.response_verbosity, str)
            and preparation.prompt.response_verbosity.strip()
            else None
        )
        resolved_input_payload = self._build_openai_input_payload(
            preparation=preparation,
        )
        stream_request_kwargs: dict[str, Any] = {
            "model": resolved_model,
            "input_text": preparation.prompt.enriched_input,
            "input_payload": resolved_input_payload,
            "instructions": preparation.prompt.instructions,
            "max_output_tokens": resolved_max_output_tokens,
            "previous_response_id": session.previous_response_id,
            "tools": preparation.tools,
            "tool_choice": preparation.prompt.tool_choice,
            "reasoning": resolved_reasoning,
            "response_verbosity": resolved_response_verbosity,
        }
        stream_state.request_model = resolved_model
        stream_state.request_reasoning_effort = (
            str(resolved_reasoning.get("effort", "")).strip().lower()
            if isinstance(resolved_reasoning, dict)
            else None
        ) or None
        stream_state.request_response_verbosity = resolved_response_verbosity
        stream_state.request_max_output_tokens = resolved_max_output_tokens
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

    def _build_openai_input_payload(
        self,
        *,
        preparation: _TurnPreparation,
    ) -> list[dict[str, Any]] | None:
        if preparation.phase_before != Phase.PRE_STRATEGY.value:
            return None

        pre_raw = preparation.artifacts.get(Phase.PRE_STRATEGY.value)
        if not isinstance(pre_raw, dict):
            return None

        missing_raw = pre_raw.get("missing_fields")
        if not isinstance(missing_raw, list):
            return None
        missing = [str(item).strip() for item in missing_raw if str(item).strip()]
        if missing != ["strategy_family_choice"]:
            return None

        runtime_raw = pre_raw.get("runtime")
        if not isinstance(runtime_raw, dict):
            return None
        if str(runtime_raw.get("regime_snapshot_status", "")).strip().lower() != "ready":
            return None

        snapshot_id = str(runtime_raw.get("regime_snapshot_id", "")).strip()
        if not snapshot_id:
            return None

        chart_pack = self._load_pre_strategy_chart_data(
            snapshot_id=snapshot_id,
        )
        if chart_pack is None:
            return None
        image_data_url, alt_text = chart_pack

        content: list[dict[str, Any]] = [
            {
                "type": "input_text",
                "text": preparation.prompt.enriched_input,
            },
            {
                "type": "input_image",
                "image_url": image_data_url,
            },
        ]
        if isinstance(alt_text, str) and alt_text.strip():
            content.insert(
                1,
                {
                    "type": "input_text",
                    "text": (
                        "[AUTO_GENERATED_REGIME_CHART_CONTEXT]\n"
                        f"{alt_text.strip()}"
                    ),
                },
            )
        return [{"role": "user", "content": content}]

    def _load_pre_strategy_chart_data(
        self,
        *,
        snapshot_id: str,
    ) -> tuple[str, str] | None:
        try:
            from apps.mcp.domains.market_data.tools import (
                pre_strategy_render_candlestick,
            )
        except Exception as exc:  # noqa: BLE001
            log_agent(
                "orchestrator",
                (
                    "pre_strategy_image_attach_import_failed "
                    f"error={type(exc).__name__}"
                ),
            )
            return None

        try:
            rendered = pre_strategy_render_candlestick(
                snapshot_id=snapshot_id,
                timeframe="primary",
                bars=min(int(settings.pre_strategy_regime_image_max_bars), 180),
            )
        except Exception as exc:  # noqa: BLE001
            log_agent(
                "orchestrator",
                (
                    "pre_strategy_image_attach_render_failed "
                    f"snapshot_id={snapshot_id} error={type(exc).__name__}"
                ),
            )
            return None

        if not isinstance(rendered, list) or not rendered:
            return None

        image_data_url: str | None = None
        alt_text = ""
        for item in rendered:
            if image_data_url is None:
                image_data_url = self._coerce_image_data_url(item)
            if not alt_text and isinstance(item, str) and item.strip():
                alt_text = item.strip()

        if not image_data_url:
            return None
        return image_data_url, alt_text

    @staticmethod
    def _coerce_image_data_url(value: Any) -> str | None:
        raw_data: bytes | None = None
        if hasattr(value, "data"):
            candidate = getattr(value, "data", None)
            if isinstance(candidate, bytes):
                raw_data = candidate
            elif isinstance(candidate, str) and candidate.strip():
                try:
                    raw_data = base64.b64decode(candidate.strip(), validate=True)
                except Exception:  # noqa: BLE001
                    raw_data = None
        elif isinstance(value, dict):
            candidate = value.get("data")
            if isinstance(candidate, bytes):
                raw_data = candidate
            elif isinstance(candidate, str) and candidate.strip():
                try:
                    raw_data = base64.b64decode(candidate.strip(), validate=True)
                except Exception:  # noqa: BLE001
                    raw_data = None

        if not raw_data:
            return None

        raw_format = (
            getattr(value, "_format", None)
            or getattr(value, "format", None)
            or (value.get("format") if isinstance(value, dict) else None)
            or "png"
        )
        fmt = str(raw_format).strip().lower() or "png"
        if fmt == "jpg":
            fmt = "jpeg"
        encoded = base64.b64encode(raw_data).decode("ascii")
        return f"data:image/{fmt};base64,{encoded}"

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
