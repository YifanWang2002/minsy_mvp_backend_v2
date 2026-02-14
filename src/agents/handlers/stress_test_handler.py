"""Stress-test phase handler."""

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
from src.agents.skills.stress_test_skills import (
    REQUIRED_FIELDS,
    VALID_DECISION_VALUES,
    VALID_STATUS_VALUES,
    build_stress_test_dynamic_state,
    build_stress_test_static_instructions,
)
from src.config import settings


class StressTestHandler:
    """Legacy stress-test phase handler.

    Current product boundary keeps all performance iteration in strategy.
    Legacy sessions may still carry this phase in persisted rows.
    """

    @property
    def phase_name(self) -> str:
        return Phase.STRESS_TEST.value

    @property
    def required_fields(self) -> list[str]:
        return list(REQUIRED_FIELDS)

    @property
    def valid_values(self) -> dict[str, set[str]]:
        return {
            "backtest_status": set(VALID_STATUS_VALUES),
            "stress_test_decision": set(VALID_DECISION_VALUES),
        }

    def build_prompt(
        self,
        ctx: PhaseContext,
        user_message: str,
    ) -> PromptPieces:
        phase_data = ctx.session_artifacts.get(Phase.STRESS_TEST.value, {})
        profile = dict(phase_data.get("profile", {}))
        strategy_profile = dict(
            (ctx.session_artifacts.get(Phase.STRATEGY.value, {}) or {}).get("profile", {})
        )
        if "strategy_id" not in profile and isinstance(strategy_profile.get("strategy_id"), str):
            profile["strategy_id"] = strategy_profile["strategy_id"]

        instructions = build_stress_test_static_instructions(
            language=ctx.language,
            phase_stage=ctx.runtime_policy.phase_stage,
        )
        state_block = build_stress_test_dynamic_state(
            collected_fields=profile,
            session_id=str(ctx.session_id) if ctx.session_id is not None else None,
        )
        tool_choice = _build_stress_test_tool_choice(profile=profile)
        return PromptPieces(
            instructions=instructions,
            enriched_input=state_block + user_message,
            tools=_build_backtest_tools(),
            tool_choice=tool_choice,
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
            Phase.STRESS_TEST.value,
            {"profile": {}, "missing_fields": list(REQUIRED_FIELDS)},
        )
        profile = dict(phase_data.get("profile", {}))

        strategy_profile = dict(
            (artifacts.get(Phase.STRATEGY.value, {}) or {}).get("profile", {})
        )
        if "strategy_id" not in profile and isinstance(strategy_profile.get("strategy_id"), str):
            profile["strategy_id"] = strategy_profile["strategy_id"]

        for patch in raw_patches:
            validated = self._validate_patch(patch)
            if validated:
                profile.update(validated)

        missing = self._compute_missing(profile)
        status = str(profile.get("backtest_status", "")).strip().lower()
        decision = str(profile.get("stress_test_decision", "")).strip().lower()
        completed = False
        next_phase: str | None = None
        reason: str | None = None

        if status == "done" and not missing:
            completed = True
            next_phase = Phase.STRATEGY.value
            reason = "stress_test_legacy_done_back_to_strategy"
        elif status == "failed":
            completed = True
            next_phase = Phase.STRATEGY.value
            reason = "stress_test_legacy_failed_back_to_strategy"
            missing = []

        phase_data["profile"] = profile
        phase_data["missing_fields"] = missing

        result = PostProcessResult(
            artifacts=artifacts,
            missing_fields=missing,
            completed=completed,
            next_phase=next_phase,
            transition_reason=reason,
            phase_status={
                "backtest_status": status or "pending",
                "stress_test_decision": decision or "hold",
            },
        )
        return result

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
            return "当前版本不启用 stress_test 阶段：将回到策略阶段继续回测与迭代。"
        return (
            "Stress-test phase is currently disabled in this version: "
            "the flow returns to strategy for backtest iteration."
        )

    def _compute_missing(self, profile: dict[str, Any]) -> list[str]:
        return [field for field in REQUIRED_FIELDS if not _has_value(profile.get(field))]

    def _validate_patch(self, patch: dict[str, Any]) -> dict[str, str]:
        output: dict[str, str] = {}
        for field in ("strategy_id", "backtest_job_id"):
            raw_value = patch.get(field)
            if not isinstance(raw_value, str) or not raw_value.strip():
                continue
            value = raw_value.strip()
            try:
                UUID(value)
            except ValueError:
                continue
            output[field] = value

        raw_status = patch.get("backtest_status")
        if isinstance(raw_status, str):
            status = raw_status.strip().lower()
            if status in VALID_STATUS_VALUES:
                output["backtest_status"] = status

        raw_error_code = patch.get("backtest_error_code")
        if isinstance(raw_error_code, str) and raw_error_code.strip():
            output["backtest_error_code"] = raw_error_code.strip()

        raw_decision = patch.get("stress_test_decision")
        if isinstance(raw_decision, str):
            decision = raw_decision.strip().lower()
            if decision in VALID_DECISION_VALUES:
                output["stress_test_decision"] = decision
        return output


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _build_backtest_tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "mcp",
            "server_label": "backtest",
            "server_url": settings.backtest_mcp_server_url,
            "allowed_tools": [
                "backtest_create_job",
                "backtest_get_job",
                "backtest_entry_hour_pnl_heatmap",
                "backtest_entry_weekday_pnl",
                "backtest_monthly_return_table",
                "backtest_holding_period_pnl_bins",
                "backtest_long_short_breakdown",
                "backtest_exit_reason_breakdown",
                "backtest_underwater_curve",
                "backtest_rolling_metrics",
            ],
            "require_approval": "never",
        }
    ]


def _build_stress_test_tool_choice(*, profile: dict[str, Any]) -> dict[str, str] | None:
    raw_status = profile.get("backtest_status")
    status = raw_status.strip().lower() if isinstance(raw_status, str) else ""

    has_strategy_id = isinstance(profile.get("strategy_id"), str) and bool(
        str(profile.get("strategy_id")).strip()
    )
    has_job_id = isinstance(profile.get("backtest_job_id"), str) and bool(
        str(profile.get("backtest_job_id")).strip()
    )

    if has_job_id and status in {"pending", "running"}:
        return {
            "type": "mcp",
            "server_label": "backtest",
            "name": "backtest_get_job",
        }

    if not has_job_id and has_strategy_id:
        return {
            "type": "mcp",
            "server_label": "backtest",
            "name": "backtest_create_job",
        }

    return None
