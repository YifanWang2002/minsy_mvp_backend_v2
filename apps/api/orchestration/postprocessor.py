"""Orchestrator mixin extracted from legacy implementation."""

from __future__ import annotations

from .shared import *  # noqa: F403


class PostProcessorMixin:
    async def _post_process_turn(
        self,
        *,
        session: Session,
        user: User,
        payload: ChatSendRequest,
        language: str,
        preparation: _TurnPreparation,
        stream_state: _TurnStreamState,
    ) -> _TurnPostProcessResult:
        cleaned_text, genui_payloads, raw_patches = self._extract_wrapped_payloads(
            stream_state.full_text
        )
        record_chat_debug_trace(
            "openai_to_orchestrator_full_text",
            {
                "session_id": str(session.id),
                "phase": preparation.phase_before,
                "full_text": stream_state.full_text,
                "cleaned_text": cleaned_text,
                "stream_error": stream_state.stream_error_message,
                "stream_error_detail": stream_state.stream_error_detail,
            },
        )
        assistant_text = cleaned_text.strip()
        if stream_state.stream_error_message and not assistant_text:
            assistant_text = (
                "The upstream AI stream was interrupted before completion. "
                f"Reason: {stream_state.stream_error_message}. Please retry this step."
            )

        final_mcp_tool_calls = self._build_persistable_mcp_tool_calls(
            records=stream_state.mcp_call_records,
            order=stream_state.mcp_call_order,
        )
        selected_genui_payloads = normalize_genui_payloads(
            genui_payloads,
            allow_passthrough_unregistered=True,
        )
        selected_genui_payloads = self._maybe_auto_wrap_strategy_genui(
            phase=preparation.phase_before,
            artifacts=preparation.artifacts,
            assistant_text=assistant_text,
            existing_genui=selected_genui_payloads,
            mcp_tool_calls=final_mcp_tool_calls,
        )
        selected_genui_payloads = await self._maybe_backfill_strategy_ref_from_validate_call(
            phase=preparation.phase_before,
            artifacts=preparation.artifacts,
            existing_genui=selected_genui_payloads,
            mcp_tool_calls=final_mcp_tool_calls,
            session_id=session.id,
            user_id=user.id,
        )
        selected_genui_payloads = self._maybe_auto_wrap_backtest_charts_genui(
            phase=preparation.phase_before,
            existing_genui=selected_genui_payloads,
            mcp_tool_calls=final_mcp_tool_calls,
        )

        result = await preparation.handler.post_process(
            preparation.ctx, raw_patches, self.db
        )
        session.artifacts = result.artifacts
        if result.completed:
            # Completed turns should not keep stale selectable prompts, but
            # non-choice UI payloads (for example chart refs) are still valid
            # for transcript rendering.
            selected_genui_payloads = [
                payload
                for payload in selected_genui_payloads
                if str(payload.get("type", "")).strip().lower() != "choice_prompt"
            ]

        post_process_ctx = PhaseContext(
            user_id=preparation.ctx.user_id,
            session_artifacts=result.artifacts,
            session_id=preparation.ctx.session_id,
            language=preparation.ctx.language,
            runtime_policy=preparation.ctx.runtime_policy,
        )
        filtered_genui_payloads: list[dict[str, Any]] = []
        for genui_payload in selected_genui_payloads:
            filtered = preparation.handler.filter_genui(genui_payload, post_process_ctx)
            if filtered is not None:
                filtered_genui_payloads.append(filtered)
        filtered_genui_payloads = self._ensure_required_choice_prompt_payload(
            handler=preparation.handler,
            ctx=post_process_ctx,
            missing_fields=result.missing_fields,
            genui_payloads=filtered_genui_payloads,
        )
        filtered_genui_payloads = self._ensure_pre_strategy_chart_payload(
            phase_before=preparation.phase_before,
            artifacts=result.artifacts,
            genui_payloads=filtered_genui_payloads,
            instrument_before=preparation.pre_strategy_instrument_before,
        )
        persisted_tool_calls = [*filtered_genui_payloads, *final_mcp_tool_calls]

        transitioned = False
        if result.completed and result.next_phase:
            await self._transition_phase(
                session=session,
                to_phase=result.next_phase,
                trigger="ai_output",
                metadata={
                    "reason": result.transition_reason or "phase_completed",
                    "source": "orchestrator",
                    "context": {"phase_before": preparation.phase_before},
                    "recorded_at": datetime.now(UTC).isoformat(),
                },
            )
            transitioned = True

        assistant_text = self._resolve_assistant_text_fallbacks(
            assistant_text=assistant_text,
            filtered_genui_payloads=filtered_genui_payloads,
            full_text=stream_state.full_text,
        )

        kyc_status = result.phase_status.get("kyc_status")
        if kyc_status is None:
            kyc_status = await self._fetch_kyc_status(user.id)

        assistant_text, stop_criteria_delta = self._maybe_apply_stop_criteria_placeholder(
            session=session,
            phase=preparation.phase_before,
            phase_turn_count=preparation.phase_turn_count,
            language=language,
            assistant_text=assistant_text,
        )
        if not assistant_text.strip():
            assistant_text = self._build_empty_turn_fallback_text(
                phase=preparation.phase_before,
                missing_fields=result.missing_fields,
                language=language,
            )
            log_agent(
                "orchestrator",
                (
                    f"session={session.id} phase={preparation.phase_before} "
                    "empty_model_output_fallback"
                ),
            )
        if transitioned:
            await self._store_phase_carryover_memory(
                session=session,
                from_phase=preparation.phase_before,
                to_phase=session.current_phase,
                user_message=payload.message,
                assistant_message=assistant_text,
            )

        return _TurnPostProcessResult(
            assistant_text=assistant_text,
            cleaned_text=cleaned_text,
            filtered_genui_payloads=filtered_genui_payloads,
            persisted_tool_calls=persisted_tool_calls,
            missing_fields=result.missing_fields,
            kyc_status=kyc_status,
            transitioned=transitioned,
            stop_criteria_delta=stop_criteria_delta,
        )

    def _resolve_assistant_text_fallbacks(
        self,
        *,
        assistant_text: str,
        filtered_genui_payloads: list[dict[str, Any]],
        full_text: str,
    ) -> str:
        if assistant_text:
            return assistant_text
        if not filtered_genui_payloads or not full_text.strip():
            return assistant_text

        fallback_choice = next(
            (
                payload
                for payload in reversed(filtered_genui_payloads)
                if payload.get("type") == "choice_prompt"
            ),
            None,
        )
        question = fallback_choice.get("question") if fallback_choice is not None else None
        if isinstance(question, str) and question.strip():
            return question.strip()
        return assistant_text

    async def _emit_tail_events_and_persist(
        self,
        *,
        session: Session,
        preparation: _TurnPreparation,
        stream_state: _TurnStreamState,
        post_process_result: _TurnPostProcessResult,
    ) -> AsyncIterator[str]:
        turn_usage = build_turn_usage_snapshot(
            raw_usage=stream_state.completed_usage,
            model=stream_state.completed_model or stream_state.request_model,
            response_id=session.previous_response_id,
            at=datetime.now(UTC),
            pricing=settings.openai_pricing_json,
            cost_tracking_enabled=settings.openai_cost_tracking_enabled,
        )
        usage_payload = turn_usage or stream_state.completed_usage or None
        session_openai_cost_totals: dict[str, Any] | None = None
        if settings.openai_cost_tracking_enabled and turn_usage is not None:
            next_metadata, totals = merge_session_openai_cost_metadata(
                session.metadata_,
                turn_usage,
            )
            session.metadata_ = next_metadata
            session_openai_cost_totals = totals

        self.db.add(
            Message(
                session_id=session.id,
                role="assistant",
                content=post_process_result.assistant_text,
                phase=preparation.phase_before,
                response_id=session.previous_response_id,
                tool_calls=post_process_result.persisted_tool_calls or None,
                token_usage=usage_payload,
            )
        )
        session.last_activity_at = datetime.now(UTC)
        title_payload = await refresh_session_title(db=self.db, session=session)
        await self.db.commit()
        await self.db.refresh(session)

        for genui_payload in post_process_result.filtered_genui_payloads:
            yield self._sse("stream", {"type": "genui", "payload": genui_payload})

        if post_process_result.transitioned:
            yield self._sse(
                "stream",
                {
                    "type": "phase_change",
                    "from_phase": preparation.phase_before,
                    "to_phase": session.current_phase,
                },
            )

        stop_criteria_delta = post_process_result.stop_criteria_delta
        if isinstance(stop_criteria_delta, str) and stop_criteria_delta.strip():
            if stream_state.text_delta_emitted and post_process_result.cleaned_text.strip():
                yield self._sse(
                    "stream",
                    {
                        "type": "text_delta",
                        "delta": stop_criteria_delta,
                    },
                )
            else:
                stop_criteria_delta = None

        should_emit_terminal_text = (
            post_process_result.assistant_text.strip()
            and (
                not stream_state.text_delta_emitted
                or not post_process_result.cleaned_text.strip()
            )
        )
        if should_emit_terminal_text:
            yield self._sse(
                "stream",
                {
                    "type": "text_delta",
                    "delta": post_process_result.assistant_text,
                },
            )
            stream_state.text_delta_emitted = True

        yield self._sse(
            "stream",
            {
                "type": "done",
                "session_id": str(session.id),
                "phase": session.current_phase,
                "status": session.status,
                "kyc_status": post_process_result.kyc_status,
                "missing_fields": post_process_result.missing_fields,
                "session_title": title_payload.title,
                "session_title_record": title_payload.record,
                "usage": usage_payload,
                "session_openai_cost": session_openai_cost_totals,
                "stream_error": stream_state.stream_error_message,
                "stream_error_detail": stream_state.stream_error_detail,
            },
        )

    def _extract_wrapped_payloads(
        self,
        text: str,
    ) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
        """Parse AGENT_UI_JSON and AGENT_STATE_PATCH blocks from AI text.

        Returns (cleaned_text, genui_payloads, raw_patch_payloads).
        """
        genui_payloads = self._extract_json_by_tag(text, _AGENT_UI_TAG)
        patch_payloads = self._extract_json_by_tag(text, _AGENT_STATE_PATCH_TAG)

        cleaned = self._strip_tag_blocks(text, _AGENT_UI_TAG)
        cleaned = self._strip_tag_blocks(cleaned, _AGENT_STATE_PATCH_TAG)
        cleaned = self._strip_session_state_echo(cleaned)
        cleaned = self._strip_mcp_pseudo_tool_tags(cleaned)

        return cleaned, genui_payloads, patch_payloads

    def _extract_json_by_tag(self, text: str, tag: str) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        pattern = self._build_tag_pattern(tag)
        for matched in pattern.findall(text):
            try:
                data = json.loads(matched.strip())
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                payloads.append(data)
        return payloads

    def _strip_tag_blocks(self, text: str, tag: str) -> str:
        return self._build_tag_pattern(tag).sub("", text)

    def _strip_session_state_echo(self, text: str) -> str:
        # Remove full [SESSION STATE] ... [/SESSION STATE] echoes.
        output = re.sub(
            r"\[\s*SESSION\s+STATE\s*\][\s\S]*?\[\s*/\s*SESSION\s+STATE\s*\]",
            "",
            text,
            flags=re.IGNORECASE,
        )
        # Remove truncated echoes like a lone "[SESSION STATE]" line and attached
        # state bullets (when model starts echoing but stops mid-block).
        output = re.sub(
            r"\[\s*SESSION\s+STATE\s*\]\s*\n(?:\s*-\s*[^\n]*\n)*",
            "",
            output,
            flags=re.IGNORECASE,
        )
        # Remove any remaining standalone marker lines.
        output = re.sub(
            r"(?im)^\s*\[\s*/?\s*SESSION\s+STATE\s*\]\s*$\n?",
            "",
            output,
        )
        return output

    def _strip_mcp_pseudo_tool_tags(self, text: str) -> str:
        """Strip hallucinated pseudo-tool XML tags like <mcp_xxx>{...}</mcp_xxx>."""
        output = re.sub(
            r"<\s*(mcp_[a-z0-9_:\-]+)\s*>[\s\S]*?<\s*/\s*\1\s*>",
            "",
            text,
            flags=re.IGNORECASE,
        )

        open_matches = list(
            re.finditer(r"<\s*mcp_[a-z0-9_:\-]+\s*>", output, flags=re.IGNORECASE)
        )
        if open_matches:
            open_idx = open_matches[-1].start()
            tail = output[open_idx:]
            has_close = re.search(
                r"<\s*/\s*mcp_[a-z0-9_:\-]+\s*>",
                tail,
                flags=re.IGNORECASE,
            )
            if has_close is None:
                output = output[:open_idx]

        output = re.sub(
            r"</?\s*mcp_[a-z0-9_:\-]+\s*>",
            "",
            output,
            flags=re.IGNORECASE,
        )
        return output

    @staticmethod
    def _build_tag_pattern(tag: str) -> re.Pattern[str]:
        escaped_tag = re.escape(tag)
        return re.compile(
            rf"<\s*{escaped_tag}\s*>(.*?)<\s*/\s*{escaped_tag}\s*>",
            flags=re.DOTALL | re.IGNORECASE,
        )

    def _ensure_required_choice_prompt_payload(
        self,
        *,
        handler: Any,
        ctx: PhaseContext,
        missing_fields: list[str],
        genui_payloads: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not missing_fields:
            return genui_payloads
        if any(payload.get("type") == "choice_prompt" for payload in genui_payloads):
            return genui_payloads

        fallback_builder = getattr(handler, "build_fallback_choice_prompt", None)
        if not callable(fallback_builder):
            return genui_payloads

        try:
            candidate = fallback_builder(
                missing_fields=list(missing_fields),
                ctx=ctx,
            )
        except Exception as exc:  # noqa: BLE001
            phase_name = getattr(handler, "phase_name", "unknown")
            log_agent(
                "orchestrator",
                f"session={ctx.session_id} phase={phase_name} "
                f"fallback_choice_prompt_error={type(exc).__name__}",
            )
            return genui_payloads

        if not isinstance(candidate, dict):
            return genui_payloads

        normalized = normalize_genui_payloads(
            [candidate],
            allow_passthrough_unregistered=True,
        )
        if not normalized:
            return genui_payloads

        filtered = handler.filter_genui(normalized[0], ctx)
        if not isinstance(filtered, dict):
            return genui_payloads

        phase_name = getattr(handler, "phase_name", "unknown")
        choice_id = filtered.get("choice_id")
        log_agent(
            "orchestrator",
            f"session={ctx.session_id} phase={phase_name} "
            f"injected_fallback_choice_prompt choice_id={choice_id}",
        )
        return [*genui_payloads, filtered]

    def _ensure_pre_strategy_chart_payload(
        self,
        *,
        phase_before: str,
        artifacts: dict[str, Any],
        genui_payloads: list[dict[str, Any]],
        instrument_before: str | None,
    ) -> list[dict[str, Any]]:
        if phase_before != Phase.PRE_STRATEGY.value:
            return genui_payloads
        if any(
            payload.get("type") == "tradingview_chart" for payload in genui_payloads
        ):
            return genui_payloads
        if not any(
            payload.get("type") == "choice_prompt" for payload in genui_payloads
        ):
            return genui_payloads

        pre_strategy_data = artifacts.get(Phase.PRE_STRATEGY.value)
        if not isinstance(pre_strategy_data, dict):
            return genui_payloads
        profile = pre_strategy_data.get("profile")
        if not isinstance(profile, dict):
            return genui_payloads
        instrument = profile.get("target_instrument")
        if not isinstance(instrument, str) or not instrument.strip():
            return genui_payloads
        instrument = instrument.strip()
        if instrument_before == instrument:
            return genui_payloads
        market = profile.get("target_market")
        if not isinstance(market, str) or not market.strip():
            return genui_payloads

        symbol = get_tradingview_symbol_for_market_instrument(
            market=market,
            instrument=instrument,
        )
        if not isinstance(symbol, str) or not symbol.strip():
            return genui_payloads

        chart_payload = {
            "type": "tradingview_chart",
            "symbol": symbol,
            "interval": "D",
        }
        return [chart_payload, *genui_payloads]
