"""Strategy phase handler."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from src.agents.handler_protocol import (
    PhaseContext,
    PostProcessResult,
    PromptPieces,
)
from src.agents.phases import Phase
from src.agents.skills.strategy_skills import (
    REQUIRED_FIELDS,
    build_strategy_dynamic_state,
    build_strategy_static_instructions,
)
from src.config import settings


class StrategyHandler:
    """Implements strategy design + DSL persistence phase."""

    @property
    def phase_name(self) -> str:
        return Phase.STRATEGY.value

    @property
    def required_fields(self) -> list[str]:
        return list(REQUIRED_FIELDS)

    @property
    def valid_values(self) -> dict[str, set[str]]:
        return {}

    def build_prompt(
        self,
        ctx: PhaseContext,
        user_message: str,
    ) -> PromptPieces:
        phase_data = ctx.session_artifacts.get(Phase.STRATEGY.value, {})
        profile = dict(phase_data.get("profile", {}))
        pre_strategy_data = ctx.session_artifacts.get(Phase.PRE_STRATEGY.value, {})
        pre_profile = dict(pre_strategy_data.get("profile", {}))
        missing = self._compute_missing(profile)

        instructions = build_strategy_static_instructions(
            language=ctx.language,
            phase_stage=ctx.runtime_policy.phase_stage,
        )
        state_block = build_strategy_dynamic_state(
            missing_fields=missing,
            collected_fields=profile,
            pre_strategy_fields=pre_profile,
        )
        return PromptPieces(
            instructions=instructions,
            enriched_input=state_block + user_message,
            tools=_build_strategy_tools(),
            model=settings.openai_response_model,
            reasoning={"effort": "low"},
        )

    async def post_process(
        self,
        ctx: PhaseContext,
        raw_patches: list[dict[str, Any]],
        db: AsyncSession,
    ) -> PostProcessResult:
        del db
        artifacts = ctx.session_artifacts
        phase_data = artifacts.setdefault(
            Phase.STRATEGY.value,
            {"profile": {}, "missing_fields": list(REQUIRED_FIELDS)},
        )
        profile = dict(phase_data.get("profile", {}))

        for patch in raw_patches:
            validated = self._validate_patch(patch)
            if validated:
                profile.update(validated)

        missing = self._compute_missing(profile)
        phase_data["profile"] = profile
        phase_data["missing_fields"] = missing

        return PostProcessResult(
            artifacts=artifacts,
            missing_fields=missing,
            completed=False,
        )

    def filter_genui(
        self,
        payload: dict[str, Any],
        ctx: PhaseContext,
    ) -> dict[str, Any] | None:
        return payload

    def init_artifacts(self) -> dict[str, Any]:
        return {"profile": {}, "missing_fields": list(REQUIRED_FIELDS)}

    def build_phase_entry_guidance(self, ctx: PhaseContext) -> str | None:
        if ctx.language == "zh":
            return "进入策略阶段：先产出完整 DSL 供前端确认，确认保存后再基于 strategy_id 做回测与迭代。"
        return (
            "Entering strategy phase: draft a full DSL for frontend confirmation first, "
            "then use strategy_id for backtesting and iteration."
        )

    def _compute_missing(self, profile: dict[str, Any]) -> list[str]:
        return [field for field in REQUIRED_FIELDS if not _has_value(profile.get(field))]

    def _validate_patch(self, patch: dict[str, Any]) -> dict[str, str]:
        output: dict[str, str] = {}
        raw_strategy_id = patch.get("strategy_id")
        if isinstance(raw_strategy_id, str) and raw_strategy_id.strip():
            value = raw_strategy_id.strip()
            try:
                UUID(value)
            except ValueError:
                return output
            output["strategy_id"] = value
        return output


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _build_strategy_tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "mcp",
            "server_label": "strategy",
            "server_url": settings.strategy_mcp_server_url,
            "allowed_tools": [
                "strategy_validate_dsl",
                "strategy_upsert_dsl",
                "strategy_get_dsl",
                "strategy_list_tunable_params",
                "strategy_patch_dsl",
                "strategy_list_versions",
                "strategy_get_version_dsl",
                "strategy_diff_versions",
                "strategy_rollback_dsl",
                "get_indicator_detail",
                "get_indicator_catalog",
            ],
            "require_approval": "never",
        }
    ]
