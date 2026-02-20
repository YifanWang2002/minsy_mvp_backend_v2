"""Orchestrator mixin extracted from legacy implementation."""

from __future__ import annotations

from .shared import *  # noqa: F403


class GenUiWrapperMixin:
    def _maybe_auto_wrap_strategy_genui(
        self,
        *,
        phase: str,
        artifacts: dict[str, Any],
        assistant_text: str,
        existing_genui: list[dict[str, Any]],
        mcp_tool_calls: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if phase != Phase.STRATEGY.value:
            return existing_genui
        if self._contains_strategy_genui(existing_genui):
            return existing_genui

        strategy_profile = self._extract_phase_profile(
            artifacts=artifacts,
            phase=Phase.STRATEGY.value,
        )
        strategy_id = strategy_profile.get("strategy_id")
        if isinstance(strategy_id, str) and strategy_id.strip():
            return existing_genui

        strategy_draft_id = self._extract_strategy_draft_id_from_mcp_calls(
            mcp_tool_calls
        )
        if strategy_draft_id is not None:
            wrapped_ref = {
                "type": _STRATEGY_REF_GENUI_TYPE,
                "strategy_draft_id": strategy_draft_id,
                "source": "strategy_validate_dsl",
                "display_mode": "draft",
            }
            return self._append_genui_if_new(
                existing_genui=existing_genui, candidate=wrapped_ref
            )

        dsl_payload = self._try_extract_strategy_dsl_from_text(assistant_text)
        if not isinstance(dsl_payload, dict):
            return existing_genui
        wrapped = {
            "type": _STRATEGY_CARD_GENUI_TYPE,
            "dsl_json": dsl_payload,
            "source": "auto_detected_first_dsl",
            "display_mode": "draft",
        }
        return self._append_genui_if_new(
            existing_genui=existing_genui, candidate=wrapped
        )

    @staticmethod
    def _append_genui_if_new(
        *,
        existing_genui: list[dict[str, Any]],
        candidate: dict[str, Any],
    ) -> list[dict[str, Any]]:
        dedupe_key = json.dumps(candidate, ensure_ascii=False, sort_keys=True)
        existing_keys = {
            json.dumps(item, ensure_ascii=False, sort_keys=True)
            for item in existing_genui
        }
        if dedupe_key in existing_keys:
            return existing_genui
        return [*existing_genui, candidate]

    async def _maybe_backfill_strategy_ref_from_validate_call(
        self,
        *,
        phase: str,
        artifacts: dict[str, Any],
        existing_genui: list[dict[str, Any]],
        mcp_tool_calls: list[dict[str, Any]],
        session_id: UUID,
        user_id: UUID,
    ) -> list[dict[str, Any]]:
        if phase != Phase.STRATEGY.value:
            return existing_genui

        strategy_profile = self._extract_phase_profile(
            artifacts=artifacts,
            phase=Phase.STRATEGY.value,
        )
        strategy_id = strategy_profile.get("strategy_id")
        if isinstance(strategy_id, str) and strategy_id.strip():
            return existing_genui

        strategy_ref_index = self._find_strategy_ref_payload_index(existing_genui)
        if strategy_ref_index is not None:
            existing_ref = existing_genui[strategy_ref_index]
            existing_ref_draft_id = self._coerce_uuid_text(
                existing_ref.get("strategy_draft_id")
            )
            if existing_ref_draft_id is not None:
                return existing_genui

        latest = self._extract_latest_successful_strategy_validate_call(mcp_tool_calls)
        if latest is None:
            return existing_genui
        validate_call, output_payload = latest

        # Normal path already handled by the regular wrapper.
        existing_draft_id = self._coerce_uuid_text(output_payload.get("strategy_draft_id"))
        if existing_draft_id is None:
            data_payload = self._coerce_json_object(output_payload.get("data"))
            if isinstance(data_payload, dict):
                existing_draft_id = self._coerce_uuid_text(data_payload.get("strategy_draft_id"))
        if existing_draft_id is not None:
            return existing_genui

        dsl_payload = self._extract_validate_dsl_payload_from_arguments(
            validate_call.get("arguments")
        )
        if not isinstance(dsl_payload, dict):
            return existing_genui

        validation = validate_strategy_payload(dsl_payload)
        if not validation.is_valid:
            return existing_genui

        try:
            draft = await create_strategy_draft(
                user_id=user_id,
                session_id=session_id,
                dsl_json=dsl_payload,
            )
        except Exception as exc:  # noqa: BLE001
            log_agent(
                "orchestrator",
                (
                    f"session={session_id} strategy_ref_backfill_failed "
                    f"{type(exc).__name__}: {exc}"
                ),
            )
            return existing_genui

        self._inject_strategy_draft_into_validate_call_output(
            validate_call=validate_call,
            output_payload=output_payload,
            strategy_draft_id=str(draft.strategy_draft_id),
            draft_expires_at=draft.expires_at.isoformat(),
            draft_ttl_seconds=int(draft.ttl_seconds),
        )
        if strategy_ref_index is not None:
            updated = list(existing_genui)
            next_payload = dict(updated[strategy_ref_index])
            next_payload["strategy_draft_id"] = str(draft.strategy_draft_id)
            next_payload["source"] = "strategy_validate_dsl_backfill"
            if not str(next_payload.get("display_mode", "")).strip():
                next_payload["display_mode"] = "draft"
            updated[strategy_ref_index] = next_payload
            return updated

        wrapped_ref = {
            "type": _STRATEGY_REF_GENUI_TYPE,
            "strategy_draft_id": str(draft.strategy_draft_id),
            "source": "strategy_validate_dsl_backfill",
            "display_mode": "draft",
        }
        return self._append_genui_if_new(
            existing_genui=existing_genui,
            candidate=wrapped_ref,
        )

    @staticmethod
    def _find_strategy_ref_payload_index(payloads: list[dict[str, Any]]) -> int | None:
        for index, payload in enumerate(payloads):
            payload_type = payload.get("type")
            if not isinstance(payload_type, str):
                continue
            if payload_type.strip().lower() == _STRATEGY_REF_GENUI_TYPE:
                return index
        return None

    def _extract_latest_successful_strategy_validate_call(
        self,
        mcp_tool_calls: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        for item in reversed(mcp_tool_calls):
            if not isinstance(item, dict):
                continue
            if str(item.get("type", "")).strip().lower() != "mcp_call":
                continue
            if str(item.get("name", "")).strip() != "strategy_validate_dsl":
                continue
            if str(item.get("status", "")).strip().lower() != "success":
                continue
            output_payload = self._coerce_json_object(item.get("output"))
            if not isinstance(output_payload, dict):
                output_payload = self._coerce_json_object(item.get("result"))
            if not isinstance(output_payload, dict):
                continue
            if output_payload.get("ok") is not True:
                continue
            return item, output_payload
        return None

    def _extract_validate_dsl_payload_from_arguments(
        self,
        arguments: Any,
    ) -> dict[str, Any] | None:
        payload = self._coerce_json_object(arguments)
        if not isinstance(payload, dict):
            return None
        raw_dsl = payload.get("dsl_json")
        if isinstance(raw_dsl, dict):
            return dict(raw_dsl)
        if isinstance(raw_dsl, str):
            parsed = self._coerce_json_object(raw_dsl)
            if isinstance(parsed, dict):
                return parsed
        return None

    def _inject_strategy_draft_into_validate_call_output(
        self,
        *,
        validate_call: dict[str, Any],
        output_payload: dict[str, Any],
        strategy_draft_id: str,
        draft_expires_at: str,
        draft_ttl_seconds: int,
    ) -> None:
        next_payload = dict(output_payload)
        next_payload["strategy_draft_id"] = strategy_draft_id
        next_payload["draft_expires_at"] = draft_expires_at
        next_payload["draft_ttl_seconds"] = draft_ttl_seconds

        existing_output = validate_call.get("output")
        if isinstance(existing_output, dict):
            validate_call["output"] = next_payload
            return
        validate_call["output"] = json.dumps(next_payload, ensure_ascii=False)

    def _extract_strategy_draft_id_from_mcp_calls(
        self,
        mcp_tool_calls: list[dict[str, Any]],
    ) -> str | None:
        for item in reversed(mcp_tool_calls):
            if not isinstance(item, dict):
                continue
            if str(item.get("type", "")).strip().lower() != "mcp_call":
                continue
            if str(item.get("name", "")).strip() != "strategy_validate_dsl":
                continue
            if str(item.get("status", "")).strip().lower() != "success":
                continue

            output_payload = self._coerce_json_object(item.get("output"))
            if not isinstance(output_payload, dict):
                continue
            if output_payload.get("ok") is not True:
                continue
            draft_id = self._coerce_uuid_text(output_payload.get("strategy_draft_id"))
            if draft_id is None:
                data_payload = self._coerce_json_object(output_payload.get("data"))
                if isinstance(data_payload, dict):
                    draft_id = self._coerce_uuid_text(
                        data_payload.get("strategy_draft_id")
                    )
            if draft_id is None:
                continue
            return draft_id
        return None

    @staticmethod
    def _coerce_json_object(value: Any) -> dict[str, Any] | None:
        if isinstance(value, dict):
            return dict(value)
        if not isinstance(value, str):
            return None
        text = value.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            return parsed
        return None

    @staticmethod
    def _contains_strategy_genui(payloads: list[dict[str, Any]]) -> bool:
        for payload in payloads:
            payload_type = payload.get("type")
            if not isinstance(payload_type, str):
                continue
            normalized = payload_type.strip().lower()
            if normalized in {_STRATEGY_CARD_GENUI_TYPE, _STRATEGY_REF_GENUI_TYPE}:
                return True
        return False

    @staticmethod
    def _contains_backtest_charts_genui(payloads: list[dict[str, Any]]) -> bool:
        for payload in payloads:
            payload_type = payload.get("type")
            if not isinstance(payload_type, str):
                continue
            if payload_type.strip().lower() == _BACKTEST_CHARTS_GENUI_TYPE:
                return True
        return False

    def _maybe_auto_wrap_backtest_charts_genui(
        self,
        *,
        phase: str,
        existing_genui: list[dict[str, Any]],
        mcp_tool_calls: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if phase != Phase.STRATEGY.value:
            return existing_genui
        if self._contains_backtest_charts_genui(existing_genui):
            return existing_genui

        resolved = self._extract_backtest_done_from_mcp_calls(mcp_tool_calls)
        if resolved is None:
            return existing_genui

        job_id, strategy_id, source = resolved
        payload: dict[str, Any] = {
            "type": _BACKTEST_CHARTS_GENUI_TYPE,
            "job_id": job_id,
            "charts": list(_BACKTEST_DEFAULT_CHARTS),
            "sampling": "eod",
            "max_points": 365,
            "source": source,
        }
        if strategy_id is not None:
            payload["strategy_id"] = strategy_id
        return self._append_genui_if_new(
            existing_genui=existing_genui,
            candidate=payload,
        )

    def _extract_backtest_done_from_mcp_calls(
        self,
        mcp_tool_calls: list[dict[str, Any]],
    ) -> tuple[str, str | None, str] | None:
        for item in reversed(mcp_tool_calls):
            if not isinstance(item, dict):
                continue
            if str(item.get("type", "")).strip().lower() != "mcp_call":
                continue

            name = str(item.get("name", "")).strip()
            if name not in _BACKTEST_COMPLETED_TOOL_NAMES:
                continue
            if str(item.get("status", "")).strip().lower() != "success":
                continue

            output_payload = self._coerce_json_object(item.get("output"))
            if not isinstance(output_payload, dict):
                continue
            if output_payload.get("ok") is not True:
                continue
            if str(output_payload.get("status", "")).strip().lower() != "done":
                continue

            job_id = self._coerce_uuid_text(output_payload.get("job_id"))
            if job_id is None:
                continue
            strategy_id = self._coerce_uuid_text(output_payload.get("strategy_id"))
            return job_id, strategy_id, name
        return None

    def _try_extract_strategy_dsl_from_text(self, text: str) -> dict[str, Any] | None:
        if not isinstance(text, str):
            return None
        body = text.strip()
        if not body:
            return None

        candidates = self._extract_json_candidates_from_text(body)
        for raw_candidate in candidates:
            try:
                parsed = json.loads(raw_candidate)
            except json.JSONDecodeError:
                continue
            if not isinstance(parsed, dict):
                continue
            if not self._looks_like_strategy_dsl(parsed):
                continue

            validation = validate_strategy_payload(parsed)
            if validation.is_valid:
                return parsed
        return None

    @staticmethod
    def _looks_like_strategy_dsl(payload: dict[str, Any]) -> bool:
        required_keys = ("strategy", "universe", "timeframe", "trade")
        for key in required_keys:
            if key not in payload:
                return False
        return isinstance(payload.get("strategy"), dict) and isinstance(
            payload.get("trade"), dict
        )

    @staticmethod
    def _extract_json_candidates_from_text(text: str) -> list[str]:
        candidates: list[str] = []
        seen: set[str] = set()

        def _add(candidate: str) -> None:
            normalized = candidate.strip()
            if not normalized:
                return
            if normalized in seen:
                return
            seen.add(normalized)
            candidates.append(normalized)

        for match in re.finditer(
            r"```(?:json)?\s*([\s\S]*?)```",
            text,
            flags=re.IGNORECASE,
        ):
            block = match.group(1)
            if isinstance(block, str):
                _add(block)

        depth = 0
        in_string = False
        escaped = False
        start_index: int | None = None
        max_candidates = 40

        for idx, ch in enumerate(text):
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue

            if ch == "{":
                if depth == 0:
                    start_index = idx
                depth += 1
                continue

            if ch == "}":
                if depth <= 0:
                    continue
                depth -= 1
                if depth == 0 and start_index is not None:
                    _add(text[start_index : idx + 1])
                    if len(candidates) >= max_candidates:
                        break
                    start_index = None

        return candidates
