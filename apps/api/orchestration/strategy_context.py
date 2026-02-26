"""Orchestrator mixin extracted from legacy implementation."""

from __future__ import annotations

from .shared import *  # noqa: F403


class StrategyContextMixin:
    async def _enforce_strategy_only_boundary(self, *, session: Session) -> None:
        """Redirect legacy stress-test sessions back into strategy phase.

        Product boundary today keeps performance iteration inside strategy.
        If an old session is still parked in ``stress_test``, we migrate it
        before prompt/tool selection so the model never executes stress phase
        instructions.
        """
        if session.current_phase != Phase.STRESS_TEST.value:
            return

        artifacts = self._ensure_phase_keyed(copy.deepcopy(session.artifacts or {}))
        strategy_block = artifacts.setdefault(
            Phase.STRATEGY.value,
            {"profile": {}, "missing_fields": ["strategy_id"]},
        )
        stress_block = artifacts.setdefault(
            Phase.STRESS_TEST.value,
            {
                "profile": {},
                "missing_fields": ["strategy_id", "backtest_job_id", "backtest_status"],
            },
        )

        strategy_profile_raw = strategy_block.get("profile")
        strategy_profile = (
            dict(strategy_profile_raw) if isinstance(strategy_profile_raw, dict) else {}
        )
        stress_profile_raw = stress_block.get("profile")
        stress_profile = (
            dict(stress_profile_raw) if isinstance(stress_profile_raw, dict) else {}
        )

        resolved_strategy_id = self._coerce_uuid_text(
            strategy_profile.get("strategy_id")
        )
        if resolved_strategy_id is None:
            resolved_strategy_id = self._coerce_uuid_text(
                stress_profile.get("strategy_id")
            )
        if resolved_strategy_id is not None:
            strategy_profile["strategy_id"] = resolved_strategy_id
            raw_missing = strategy_block.get("missing_fields")
            if isinstance(raw_missing, list):
                strategy_block["missing_fields"] = [
                    normalized
                    for item in raw_missing
                    if (normalized := str(item).strip()) and normalized != "strategy_id"
                ]
            else:
                strategy_block["missing_fields"] = []

        strategy_block["profile"] = strategy_profile
        stress_block["profile"] = stress_profile
        session.artifacts = artifacts

        await self._transition_phase(
            session=session,
            to_phase=Phase.STRATEGY.value,
            trigger="system",
            metadata={"reason": "stress_test_disabled_redirect_to_strategy"},
        )

    async def _hydrate_strategy_context(
        self,
        *,
        session: Session,
        user_id: UUID,
        phase: str,
        artifacts: dict[str, Any],
        user_message: str,
    ) -> None:
        if phase not in {Phase.STRATEGY.value, Phase.STRESS_TEST.value}:
            return

        strategy_block = artifacts.setdefault(
            Phase.STRATEGY.value,
            {"profile": {}, "missing_fields": ["strategy_id"]},
        )
        strategy_profile_raw = strategy_block.get("profile")
        strategy_profile = (
            dict(strategy_profile_raw) if isinstance(strategy_profile_raw, dict) else {}
        )

        stress_block = artifacts.setdefault(
            Phase.STRESS_TEST.value,
            {
                "profile": {},
                "missing_fields": ["strategy_id", "backtest_job_id", "backtest_status"],
            },
        )
        stress_profile_raw = stress_block.get("profile")
        stress_profile = (
            dict(stress_profile_raw) if isinstance(stress_profile_raw, dict) else {}
        )

        resolved_strategy_id = self._coerce_uuid_text(
            strategy_profile.get("strategy_id")
        )
        if resolved_strategy_id is None:
            resolved_strategy_id = self._coerce_uuid_text(
                stress_profile.get("strategy_id")
            )
        if resolved_strategy_id is None:
            metadata = dict(session.metadata_ or {})
            resolved_strategy_id = self._coerce_uuid_text(metadata.get("strategy_id"))

        if resolved_strategy_id is None:
            for candidate in self._extract_uuid_candidates(text=user_message):
                if await self._strategy_belongs_to_user(
                    user_id=user_id, strategy_id=candidate
                ):
                    resolved_strategy_id = candidate
                    break
        if resolved_strategy_id is None:
            resolved_strategy_id = await self._resolve_latest_session_strategy_id(
                user_id=user_id,
                session_id=session.id,
            )

        strategy_block["profile"] = strategy_profile
        stress_block["profile"] = stress_profile
        if resolved_strategy_id is None:
            return

        strategy_profile["strategy_id"] = resolved_strategy_id
        if "strategy_id" not in stress_profile:
            stress_profile["strategy_id"] = resolved_strategy_id

        raw_strategy_missing = strategy_block.get("missing_fields")
        if isinstance(raw_strategy_missing, list):
            strategy_block["missing_fields"] = [
                normalized
                for item in raw_strategy_missing
                if (normalized := str(item).strip()) and normalized != "strategy_id"
            ]
        else:
            strategy_block["missing_fields"] = []

        raw_stress_missing = stress_block.get("missing_fields")
        if isinstance(raw_stress_missing, list):
            stress_block["missing_fields"] = [
                normalized
                for item in raw_stress_missing
                if (normalized := str(item).strip()) and normalized != "strategy_id"
            ]

    async def _strategy_belongs_to_user(
        self,
        *,
        user_id: UUID,
        strategy_id: str,
    ) -> bool:
        try:
            strategy_uuid = UUID(strategy_id)
        except ValueError:
            return False

        owned = await self.db.scalar(
            select(Strategy.id).where(
                Strategy.id == strategy_uuid,
                Strategy.user_id == user_id,
            )
        )
        return owned is not None

    async def _resolve_latest_session_strategy_id(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
    ) -> str | None:
        strategy_uuid = await self.db.scalar(
            select(Strategy.id)
            .where(
                Strategy.user_id == user_id,
                Strategy.session_id == session_id,
            )
            .order_by(
                Strategy.updated_at.desc(),
                Strategy.created_at.desc(),
            )
            .limit(1)
        )
        if strategy_uuid is None:
            return None
        return self._coerce_uuid_text(str(strategy_uuid))

    @staticmethod
    def _coerce_uuid_text(value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        text = value.strip()
        if not text:
            return None
        try:
            return str(UUID(text))
        except ValueError:
            return None

    @staticmethod
    def _extract_uuid_candidates(*, text: str) -> list[str]:
        if not isinstance(text, str):
            return []

        output: list[str] = []
        seen: set[str] = set()
        for candidate in _UUID_CANDIDATE_PATTERN.findall(text):
            normalized = StrategyContextMixin._coerce_uuid_text(candidate)
            if normalized is None or normalized in seen:
                continue
            seen.add(normalized)
            output.append(normalized)
        return output
