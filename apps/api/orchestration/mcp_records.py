"""Orchestrator mixin extracted from legacy implementation."""

from __future__ import annotations

from .shared import *  # noqa: F403


class McpRecordsMixin:
    def _collect_mcp_records_from_event(
        self,
        *,
        event_type: str,
        event: dict[str, Any],
        sequence_number: Any,
        records: dict[str, dict[str, Any]],
        order: list[str],
        fallback_counter: int,
    ) -> int:
        candidates: list[dict[str, Any]] = []

        if event_type in {"response.output_item.added", "response.output_item.done"}:
            raw_item = event.get("item")
            if isinstance(raw_item, dict):
                candidates.append(raw_item)

        if event_type == "response.completed":
            response_obj = event.get("response")
            if isinstance(response_obj, dict):
                output_items = response_obj.get("output")
                if isinstance(output_items, list):
                    for output_item in output_items:
                        if isinstance(output_item, dict):
                            candidates.append(output_item)

        if "mcp_call" in event_type:
            candidates.append(event)

        for candidate in candidates:
            fallback_counter = self._upsert_mcp_record(
                candidate=candidate,
                event_type=event_type,
                sequence_number=sequence_number,
                records=records,
                order=order,
                fallback_counter=fallback_counter,
            )
        return fallback_counter

    def _upsert_mcp_record(
        self,
        *,
        candidate: dict[str, Any],
        event_type: str,
        sequence_number: Any,
        records: dict[str, dict[str, Any]],
        order: list[str],
        fallback_counter: int,
    ) -> int:
        event_type_norm = event_type.strip().lower()
        payload_type_raw = candidate.get("type")
        payload_type = (
            payload_type_raw.strip().lower()
            if isinstance(payload_type_raw, str)
            else ""
        )

        if payload_type == "mcp_list_tools" or "mcp_list_tools" in event_type_norm:
            return fallback_counter

        is_payload_mcp_call = payload_type == "mcp_call"
        is_mcp_call_event = "mcp_call" in event_type_norm
        if not is_payload_mcp_call and not is_mcp_call_event:
            return fallback_counter

        call_id, fallback_counter = self._resolve_mcp_record_id(
            candidate=candidate,
            sequence_number=sequence_number,
            fallback_counter=fallback_counter,
        )

        existing = records.get(call_id, {})
        next_status = self._normalize_mcp_status(existing.get("status")) or "running"
        status_from_payload = self._normalize_mcp_status(candidate.get("status"))
        status_from_event = self._status_from_mcp_event_type(event_type_norm)
        merged_status = status_from_payload or status_from_event
        error_text = self._normalize_mcp_error(candidate.get("error"))
        if error_text:
            next_status = "failure"
        elif merged_status == "failure":
            next_status = "failure"
        elif merged_status in {"running", "success"} and next_status != "failure":
            next_status = merged_status

        arguments = candidate.get("arguments")
        if arguments is None:
            arguments = candidate.get("args")
        next_arguments = (
            arguments if arguments is not None else existing.get("arguments")
        )

        output = candidate.get("output")
        if output is None:
            output = candidate.get("result")
        if output is None:
            output = candidate.get("response")
        next_output = output if output is not None else existing.get("output")
        output_failure_error = self._extract_mcp_output_failure_error(next_output)
        if output_failure_error:
            next_status = "failure"

        tool_name = candidate.get("name")
        if isinstance(tool_name, str):
            cleaned_name = tool_name.strip()
        else:
            cleaned_name = ""
        if not cleaned_name:
            existing_name = existing.get("name")
            cleaned_name = (
                existing_name.strip()
                if isinstance(existing_name, str) and existing_name.strip()
                else "mcp_call"
            )

        next_error = (
            error_text
            or output_failure_error
            or self._normalize_mcp_error(existing.get("error"))
        )

        next_record: dict[str, Any] = {
            "type": "mcp_call",
            "id": call_id,
            "call_id": call_id,
            "name": cleaned_name,
            "status": next_status,
        }
        if next_arguments is not None:
            next_record["arguments"] = next_arguments
        if next_output is not None:
            next_record["output"] = next_output
        if next_error:
            next_record["error"] = next_error
            next_record["status"] = "failure"

        records[call_id] = next_record
        if call_id not in order:
            order.append(call_id)
        return fallback_counter

    @staticmethod
    def _resolve_mcp_record_id(
        *,
        candidate: dict[str, Any],
        sequence_number: Any,
        fallback_counter: int,
    ) -> tuple[str, int]:
        for key in ("call_id", "item_id", "output_item_id", "id"):
            raw = candidate.get(key)
            if raw is None:
                continue
            text = str(raw).strip()
            if text:
                return text, fallback_counter

        output_index = candidate.get("output_index")
        if output_index is not None:
            output_text = str(output_index).strip()
            if output_text:
                return f"output_{output_text}", fallback_counter

        if sequence_number is not None:
            seq_text = str(sequence_number).strip()
            if seq_text:
                return f"seq_{seq_text}", fallback_counter

        next_counter = fallback_counter + 1
        return f"mcp_call_{next_counter}", next_counter

    @staticmethod
    def _normalize_mcp_status(value: Any) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip().lower()
        if not normalized:
            return None
        if normalized in {"in_progress", "running"}:
            return "running"
        if normalized in {"completed", "success"}:
            return "success"
        if normalized in {"failed", "error", "failure"}:
            return "failure"
        return None

    @staticmethod
    def _status_from_mcp_event_type(event_type: str) -> str | None:
        normalized = event_type.strip().lower()
        if normalized.endswith(".in_progress"):
            return "running"
        if normalized.endswith(".completed"):
            return "success"
        if normalized.endswith(".failed"):
            return "failure"
        return None

    @staticmethod
    def _normalize_mcp_error(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            text = value.strip()
            return text or None
        try:
            text = json.dumps(value, ensure_ascii=False)
        except TypeError:
            text = str(value)
        text = text.strip()
        return text or None

    def _extract_mcp_output_failure_error(self, value: Any) -> str | None:
        payload = self._coerce_json_object(value)
        if not isinstance(payload, dict):
            return None

        has_error = payload.get("error") is not None
        output_failed = payload.get("ok") is False or has_error
        if not output_failed:
            return None

        nested_error = self._normalize_mcp_error(payload.get("error"))
        if nested_error:
            return nested_error

        for key in ("message", "detail", "description", "code"):
            text = self._normalize_mcp_error(payload.get(key))
            if text:
                return text
        return "MCP tool response marked as failed."

    def _build_persistable_mcp_tool_calls(
        self,
        *,
        records: dict[str, dict[str, Any]],
        order: list[str],
    ) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for call_id in order:
            record = records.get(call_id)
            if not isinstance(record, dict):
                continue

            status = self._normalize_mcp_status(record.get("status"))
            error = self._normalize_mcp_error(record.get("error"))
            if error:
                status = "failure"

            if status not in {"success", "failure"}:
                continue

            name_raw = record.get("name")
            name = (
                str(name_raw).strip()
                if name_raw is not None and str(name_raw).strip()
                else "mcp_call"
            )
            item: dict[str, Any] = {
                "type": "mcp_call",
                "id": call_id,
                "call_id": call_id,
                "name": name,
                "status": status,
            }
            if "arguments" in record:
                item["arguments"] = record["arguments"]
            if "output" in record:
                item["output"] = record["output"]
            if error:
                item["error"] = error
            output.append(item)

        # Suppress transient retry failures when the same tool+arguments
        # succeeds later in the same assistant turn.
        success_signatures: set[tuple[str, str]] = set()
        success_names: set[str] = set()
        for item in output:
            if item.get("status") != "success":
                continue
            name = str(item.get("name", "")).strip()
            success_names.add(name)
            success_signatures.add(
                (
                    name,
                    self._stable_json_signature(item.get("arguments")),
                )
            )

        filtered: list[dict[str, Any]] = []
        for item in output:
            status = str(item.get("status", "")).strip().lower()
            signature = (
                str(item.get("name", "")).strip(),
                self._stable_json_signature(item.get("arguments")),
            )
            if status == "failure" and self._is_retryable_mcp_failure(item) and (
                signature in success_signatures or signature[0] in success_names
            ):
                continue
            filtered.append(item)
        return filtered

    def _is_retryable_mcp_failure(self, item: dict[str, Any]) -> bool:
        retryable_hints = (
            "timeout",
            "timed out",
            "connection reset",
            "connection aborted",
            "remoteprotocolerror",
            "session terminated",
            "status=408",
            "status=424",
            "status=429",
            "status=500",
            "status=502",
            "status=503",
            "status=504",
            "temporarily unavailable",
            "try again",
        )

        parts: list[str] = []
        error_text = self._normalize_mcp_error(item.get("error"))
        if error_text:
            parts.append(error_text)

        output_payload = self._coerce_json_object(item.get("output"))
        if isinstance(output_payload, dict):
            nested_error = self._normalize_mcp_error(output_payload.get("error"))
            if nested_error:
                parts.append(nested_error)
            for key in ("message", "detail", "description"):
                text = self._normalize_mcp_error(output_payload.get(key))
                if text:
                    parts.append(text)

        blob = " ".join(part.lower() for part in parts if part)
        if not blob:
            return False
        return any(hint in blob for hint in retryable_hints)

    @staticmethod
    def _stable_json_signature(value: Any) -> str:
        if value is None:
            return ""
        try:
            return json.dumps(
                value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
        except TypeError:
            return str(value)
