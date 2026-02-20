"""Orchestrator mixin extracted from legacy implementation."""

from __future__ import annotations

from .shared import *  # noqa: F403


class CarryoverMixin:
    def _consume_phase_carryover_memory(
        self, *, session: Session, phase: str
    ) -> str | None:
        metadata = dict(session.metadata_ or {})
        raw = metadata.get(_PHASE_CARRYOVER_META_KEY)
        if not isinstance(raw, dict):
            return None

        target_phase = raw.get("target_phase")
        block = raw.get("block")
        if not isinstance(target_phase, str) or not isinstance(block, str):
            metadata.pop(_PHASE_CARRYOVER_META_KEY, None)
            session.metadata_ = metadata
            return None

        if target_phase != phase:
            metadata.pop(_PHASE_CARRYOVER_META_KEY, None)
            session.metadata_ = metadata
            return None

        metadata.pop(_PHASE_CARRYOVER_META_KEY, None)
        session.metadata_ = metadata
        return block

    async def _store_phase_carryover_memory(
        self,
        *,
        session: Session,
        from_phase: str,
        to_phase: str,
        user_message: str,
        assistant_message: str,
    ) -> None:
        block = await self._build_phase_carryover_block(
            session_id=session.id,
            from_phase=from_phase,
            user_message=user_message,
            assistant_message=assistant_message,
        )
        if not isinstance(block, str) or not block.strip():
            return

        metadata = dict(session.metadata_ or {})
        metadata[_PHASE_CARRYOVER_META_KEY] = {
            "target_phase": to_phase,
            "from_phase": from_phase,
            "created_at": datetime.now(UTC).isoformat(),
            "block": block,
        }
        session.metadata_ = metadata

    async def _build_phase_carryover_block(
        self,
        *,
        session_id: UUID,
        from_phase: str,
        user_message: str,
        assistant_message: str,
    ) -> str | None:
        stmt = (
            select(Message.role, Message.content)
            .where(
                Message.session_id == session_id,
                Message.phase == from_phase,
                Message.role.in_(("user", "assistant")),
            )
            .order_by(Message.created_at.desc(), Message.id.desc())
            .limit(_PHASE_CARRYOVER_MAX_TURNS * 2)
        )
        rows = (await self.db.execute(stmt)).all()

        entries: list[tuple[str, str]] = []
        for role, content in reversed(rows):
            normalized = self._normalize_carryover_utterance(content)
            if normalized:
                entries.append((str(role), normalized))

        current_user = self._normalize_carryover_utterance(user_message)
        current_assistant = self._normalize_carryover_utterance(assistant_message)
        if current_user:
            entries.append(("user", current_user))
        if current_assistant:
            entries.append(("assistant", current_assistant))

        deduped: list[tuple[str, str]] = []
        for role, content in entries:
            if deduped and deduped[-1] == (role, content):
                continue
            deduped.append((role, content))

        tail = deduped[-(_PHASE_CARRYOVER_MAX_TURNS * 2) :]
        if not tail:
            return None

        lines = [
            f"[{_PHASE_CARRYOVER_TAG}]",
            "- note: previous phase dialogue snippets, reference only",
            f"- from_phase: {from_phase}",
        ]
        for role, content in tail:
            lines.append(f"- {role}: {content}")
        lines.append(f"[/{_PHASE_CARRYOVER_TAG}]")
        return "\n".join(lines) + "\n\n"

    def _normalize_carryover_utterance(self, text: Any) -> str:
        if not isinstance(text, str):
            return ""
        cleaned = re.sub(r"\s+", " ", text).strip()
        if not cleaned:
            return ""
        if len(cleaned) > _PHASE_CARRYOVER_MAX_CHARS_PER_UTTERANCE:
            trimmed = cleaned[:_PHASE_CARRYOVER_MAX_CHARS_PER_UTTERANCE].rstrip()
            cleaned = f"{trimmed}..."
        return cleaned

    def _increment_phase_turn_count(self, *, session: Session, phase: str) -> int:
        metadata = dict(session.metadata_ or {})
        raw_counts = metadata.get("phase_turn_counts")
        counts = dict(raw_counts) if isinstance(raw_counts, dict) else {}

        current = counts.get(phase, 0)
        try:
            current_value = int(current)
        except (TypeError, ValueError):
            current_value = 0

        next_value = max(0, current_value) + 1
        counts[phase] = next_value
        metadata["phase_turn_counts"] = counts
        session.metadata_ = metadata
        return next_value
