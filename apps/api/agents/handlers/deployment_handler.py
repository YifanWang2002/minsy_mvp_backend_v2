"""Deployment phase handler."""

from __future__ import annotations

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.agents.handler_protocol import (
    PhaseContext,
    PostProcessResult,
    PromptPieces,
)
from apps.api.agents.phases import Phase
from apps.api.agents.skills.deployment_skills import (
    REQUIRED_FIELDS,
    VALID_STATUS_VALUES,
    build_deployment_dynamic_state,
    build_deployment_static_instructions,
)
from packages.shared_settings.schema.settings import settings


class DeploymentHandler:
    """Implements deployment readiness/confirmation phase."""

    @property
    def phase_name(self) -> str:
        return Phase.DEPLOYMENT.value

    @property
    def required_fields(self) -> list[str]:
        return list(REQUIRED_FIELDS)

    @property
    def valid_values(self) -> dict[str, set[str]]:
        return {"deployment_status": set(VALID_STATUS_VALUES)}

    def build_prompt(
        self,
        ctx: PhaseContext,
        user_message: str,
    ) -> PromptPieces:
        phase_data = ctx.session_artifacts.get(Phase.DEPLOYMENT.value, {})
        profile = dict(phase_data.get("profile", {}))
        runtime_state = dict(phase_data.get("runtime", {}))

        instructions = build_deployment_static_instructions(language=ctx.language)
        state_block = build_deployment_dynamic_state(
            collected_fields=profile,
            deployment_state=runtime_state,
        )
        return PromptPieces(
            instructions=instructions,
            enriched_input=state_block + user_message,
            model=settings.openai_response_model,
            reasoning={"effort": "none"},
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
            Phase.DEPLOYMENT.value,
            {"profile": {}, "missing_fields": list(REQUIRED_FIELDS), "runtime": {}},
        )
        profile = dict(phase_data.get("profile", {}))

        for patch in raw_patches:
            validated = self._validate_patch(patch)
            if validated:
                profile.update(validated)

        missing = self._compute_missing(profile)
        status = str(profile.get("deployment_status", "")).strip().lower()
        completed = False
        next_phase: str | None = None
        reason: str | None = None

        if status == "blocked":
            completed = True
            next_phase = Phase.STRATEGY.value
            reason = "deployment_blocked_back_to_strategy"
            missing = []

        phase_data["profile"] = profile
        phase_data["missing_fields"] = missing

        return PostProcessResult(
            artifacts=artifacts,
            missing_fields=missing,
            completed=completed,
            next_phase=next_phase,
            transition_reason=reason,
            phase_status={"deployment_status": status or "ready"},
        )

    def filter_genui(
        self,
        payload: dict[str, Any],
        ctx: PhaseContext,
    ) -> dict[str, Any] | None:
        return payload

    def init_artifacts(self) -> dict[str, Any]:
        return {"profile": {}, "missing_fields": list(REQUIRED_FIELDS), "runtime": {}}

    def build_phase_entry_guidance(self, ctx: PhaseContext) -> str | None:
        if ctx.language == "zh":
            return "进入部署阶段：确认 ready/deployed/blocked 状态并完成交付。"
        return "Entering deployment phase: confirm ready/deployed/blocked status and finalize handoff."

    def _compute_missing(self, profile: dict[str, Any]) -> list[str]:
        return [field for field in REQUIRED_FIELDS if not _has_value(profile.get(field))]

    def _validate_patch(self, patch: dict[str, Any]) -> dict[str, str]:
        output: dict[str, str] = {}
        raw = patch.get("deployment_status")
        if isinstance(raw, str):
            value = raw.strip().lower()
            if value in VALID_STATUS_VALUES:
                output["deployment_status"] = value
        return output


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True
