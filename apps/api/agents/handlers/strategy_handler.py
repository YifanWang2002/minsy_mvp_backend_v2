"""Strategy phase handler."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.agents.handler_protocol import (
    PhaseContext,
    PostProcessResult,
    PromptPieces,
)
from apps.api.agents.phases import Phase
from apps.api.agents.skills.strategy_skills import (
    REQUIRED_FIELDS,
    build_strategy_dynamic_state,
    build_strategy_static_instructions,
)
from apps.api.i18n import is_zh_locale
from apps.api.agents.deployment_defaults import (
    hydrate_deployment_profile_defaults,
)
from packages.domain.trading.services.trading_preference_service import (
    TradingPreferenceService,
)
from packages.shared_settings.schema.settings import settings


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
        stress_data = ctx.session_artifacts.get(Phase.STRESS_TEST.value, {})
        stress_profile = dict(stress_data.get("profile", {}))
        if "strategy_id" not in profile and isinstance(
            stress_profile.get("strategy_id"), str
        ):
            profile["strategy_id"] = stress_profile["strategy_id"]
        pre_strategy_data = ctx.session_artifacts.get(Phase.PRE_STRATEGY.value, {})
        pre_profile = dict(pre_strategy_data.get("profile", {}))
        pre_runtime = dict(pre_strategy_data.get("runtime", {}))
        missing = self._compute_missing(profile)

        instructions = build_strategy_static_instructions(
            language=ctx.language,
            phase_stage=ctx.runtime_policy.phase_stage,
            prompt_profile=ctx.turn_context.get("strategy_prompt_profile"),
        )
        state_block = build_strategy_dynamic_state(
            missing_fields=missing,
            collected_fields=profile,
            pre_strategy_fields=pre_profile,
            pre_strategy_runtime=pre_runtime,
            session_id=str(ctx.session_id),
            choice_selection=ctx.turn_context.get("choice_selection"),
            trade_snapshot_request=ctx.turn_context.get("trade_snapshot_request"),
            pending_trade_patch=profile.get("pending_trade_patch"),
        )
        return PromptPieces(
            instructions=instructions,
            enriched_input=state_block + user_message,
            tools=_build_strategy_tools(),
        )

    async def post_process(
        self,
        ctx: PhaseContext,
        raw_patches: list[dict[str, Any]],
        db: AsyncSession,
    ) -> PostProcessResult:
        artifacts = ctx.session_artifacts
        phase_data = artifacts.setdefault(
            Phase.STRATEGY.value,
            {"profile": {}, "missing_fields": list(REQUIRED_FIELDS)},
        )
        profile = dict(phase_data.get("profile", {}))
        stress_data = artifacts.get(Phase.STRESS_TEST.value, {})
        stress_profile = (
            dict(stress_data.get("profile", {}))
            if isinstance(stress_data, dict)
            else {}
        )
        if "strategy_id" not in profile and isinstance(
            stress_profile.get("strategy_id"), str
        ):
            profile["strategy_id"] = stress_profile["strategy_id"]

        strategy_confirmed_this_turn = False
        for patch in raw_patches:
            validated = self._validate_patch(patch)
            if validated:
                if validated.get("strategy_confirmed") is True:
                    strategy_confirmed_this_turn = True
                for key, value in validated.items():
                    if key == "pending_trade_patch":
                        if value is None:
                            profile.pop("pending_trade_patch", None)
                        else:
                            profile["pending_trade_patch"] = value
                        continue
                    profile[key] = value

        pending_trade_patch = profile.get("pending_trade_patch")
        strategy_id = str(profile.get("strategy_id", "")).strip()
        if isinstance(pending_trade_patch, dict):
            pending_strategy_id = str(
                pending_trade_patch.get("strategy_id", "")
            ).strip()
            if (
                strategy_id
                and pending_strategy_id
                and pending_strategy_id != strategy_id
            ):
                profile.pop("pending_trade_patch", None)

        missing = self._compute_missing(profile)
        completed = False
        next_phase: str | None = None
        reason: str | None = None

        if strategy_confirmed_this_turn and not missing:
            confirmed_at = datetime.now(UTC).isoformat()
            profile["strategy_confirmed"] = True
            profile["strategy_last_confirmed_at"] = confirmed_at
            await self._prepare_deployment_artifacts(
                artifacts=artifacts,
                strategy_profile=profile,
                confirmed_at=confirmed_at,
                db=db,
                user_id=ctx.user_id,
            )
            completed = True
            next_phase = Phase.DEPLOYMENT.value
            reason = "strategy_ai_confirmed_to_deployment"

        phase_data["profile"] = profile
        phase_data["missing_fields"] = missing

        return PostProcessResult(
            artifacts=artifacts,
            missing_fields=missing,
            completed=completed,
            next_phase=next_phase,
            transition_reason=reason,
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
        if is_zh_locale(ctx.language):
            return "进入策略阶段：先验证 DSL 并生成 strategy_draft_id 供前端渲染确认，确认保存后继续在本阶段完成回测与迭代。"
        return (
            "Entering strategy phase: validate a full DSL and hand off strategy_draft_id "
            "for frontend confirmation first, then keep backtesting and iteration in this same phase with strategy_id."
        )

    def _compute_missing(self, profile: dict[str, Any]) -> list[str]:
        return [
            field for field in REQUIRED_FIELDS if not _has_value(profile.get(field))
        ]

    def _validate_patch(self, patch: dict[str, Any]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        raw_strategy_id = patch.get("strategy_id")
        if isinstance(raw_strategy_id, str) and raw_strategy_id.strip():
            value = raw_strategy_id.strip()
            try:
                UUID(value)
            except ValueError:
                return output
            output["strategy_id"] = value

        strategy_confirmed = _coerce_bool(patch.get("strategy_confirmed"))
        if strategy_confirmed is not None:
            output["strategy_confirmed"] = strategy_confirmed

        if _coerce_bool(patch.get("clear_pending_trade_patch")) is True:
            output["pending_trade_patch"] = None

        if "pending_trade_patch" in patch:
            raw_pending = patch.get("pending_trade_patch")
            if raw_pending is None:
                output["pending_trade_patch"] = None
            else:
                normalized_pending = self._normalize_pending_trade_patch(raw_pending)
                if normalized_pending is not None:
                    output["pending_trade_patch"] = normalized_pending
        return output

    def _normalize_pending_trade_patch(
        self,
        payload: Any,
    ) -> dict[str, Any] | None:
        if not isinstance(payload, dict):
            return None

        strategy_id = payload.get("strategy_id")
        if not isinstance(strategy_id, str) or not strategy_id.strip():
            return None
        strategy_id = strategy_id.strip()
        try:
            UUID(strategy_id)
        except ValueError:
            return None

        patch_ops_raw = payload.get("patch_ops")
        if not isinstance(patch_ops_raw, list) or not patch_ops_raw:
            return None
        patch_ops: list[dict[str, Any]] = []
        for item in patch_ops_raw:
            if isinstance(item, dict):
                patch_ops.append(dict(item))
        if not patch_ops:
            return None

        normalized: dict[str, Any] = {
            "strategy_id": strategy_id,
            "patch_ops": patch_ops,
        }

        source_trade_raw = payload.get("source_trade")
        if isinstance(source_trade_raw, dict):
            source_trade: dict[str, Any] = {}
            job_id = source_trade_raw.get("job_id")
            if isinstance(job_id, str) and job_id.strip():
                source_trade["job_id"] = job_id.strip()
            trade_index = source_trade_raw.get("trade_index")
            if isinstance(trade_index, int) and trade_index >= 0:
                source_trade["trade_index"] = trade_index
            elif isinstance(trade_index, str) and trade_index.strip().isdigit():
                source_trade["trade_index"] = int(trade_index.strip())
            trade_uid = source_trade_raw.get("trade_uid")
            if isinstance(trade_uid, str) and trade_uid.strip():
                source_trade["trade_uid"] = trade_uid.strip()
            if source_trade:
                normalized["source_trade"] = source_trade

        backtest_request_raw = payload.get("backtest_request")
        if isinstance(backtest_request_raw, dict):
            backtest_request: dict[str, Any] = {}
            for key in (
                "start_date",
                "end_date",
                "initial_capital",
                "commission_rate",
                "slippage_bps",
            ):
                value = backtest_request_raw.get(key)
                if value is None:
                    continue
                if isinstance(value, str):
                    stripped = value.strip()
                    if stripped:
                        backtest_request[key] = stripped
                    continue
                if isinstance(value, int | float):
                    backtest_request[key] = value
            if backtest_request:
                normalized["backtest_request"] = backtest_request

        return normalized

    async def _prepare_deployment_artifacts(
        self,
        *,
        artifacts: dict[str, Any],
        strategy_profile: dict[str, Any],
        confirmed_at: str,
        db: AsyncSession,
        user_id: UUID,
    ) -> None:
        strategy_id = str(strategy_profile.get("strategy_id", "")).strip()
        if not strategy_id:
            return
        preference_view = await TradingPreferenceService(db).get_view(user_id=user_id)
        deploy_defaults = (
            dict(preference_view.deploy_defaults)
            if isinstance(preference_view.deploy_defaults, dict)
            else {}
        )

        scope_keys = (
            "strategy_name",
            "strategy_market",
            "strategy_tickers",
            "strategy_tickers_csv",
            "strategy_primary_symbol",
            "strategy_timeframe",
        )
        scope_updates: dict[str, Any] = {
            key: strategy_profile[key]
            for key in scope_keys
            if key in strategy_profile and strategy_profile[key] is not None
        }

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
        stress_profile["strategy_id"] = strategy_id
        stress_profile.update(scope_updates)
        stress_profile.pop("backtest_job_id", None)
        stress_profile.pop("backtest_status", None)
        stress_profile.pop("backtest_error_code", None)
        stress_block["profile"] = stress_profile
        stress_block["missing_fields"] = ["backtest_job_id", "backtest_status"]

        deployment_block = artifacts.setdefault(
            Phase.DEPLOYMENT.value,
            {"profile": {}, "missing_fields": ["deployment_status"], "runtime": {}},
        )
        deployment_profile_raw = deployment_block.get("profile")
        deployment_profile = (
            dict(deployment_profile_raw)
            if isinstance(deployment_profile_raw, dict)
            else {}
        )
        deployment_profile["strategy_id"] = strategy_id
        deployment_profile.update(scope_updates)
        deployment_profile["deployment_status"] = "blocked"
        deployment_profile["broker_readiness_status"] = "unknown"
        deployment_profile["deployment_confirmation_status"] = "pending"
        deployment_profile.pop("selected_broker_account_id", None)
        deployment_profile.pop("selected_broker_label", None)
        deployment_profile.pop("selected_broker_source", None)
        deployment_profile["deployment_prepared_at"] = confirmed_at
        deployment_runtime_raw = deployment_block.get("runtime")
        deployment_runtime = (
            dict(deployment_runtime_raw)
            if isinstance(deployment_runtime_raw, dict)
            else {}
        )
        deployment_runtime_defaults = {
            "strategy_id": strategy_id,
            "strategy_name": str(scope_updates.get("strategy_name", "")).strip()
            or None,
            "deployment_status": "blocked",
            "broker_readiness_status": "unknown",
            "selected_broker_account_id": None,
            "selected_broker_label": None,
            "selected_broker_source": None,
            "deployment_confirmation_status": "pending",
            "deployment_summary_snapshot": {},
            "auto_execute_pending": False,
            "prepared_at": confirmed_at,
        }
        deployment_runtime.update(deployment_runtime_defaults)
        deployment_profile, deployment_runtime = hydrate_deployment_profile_defaults(
            profile=deployment_profile,
            runtime_state=deployment_runtime,
            deploy_defaults=deploy_defaults,
        )
        deployment_block["profile"] = deployment_profile
        deployment_block["runtime"] = deployment_runtime
        deployment_block["missing_fields"] = [
            "selected_broker_account_id",
            "deployment_confirmation_status",
        ]
        deployment_runtime.update(
            {
                "planned_capital_allocated": deployment_profile.get(
                    "planned_capital_allocated",
                    "10000",
                ),
                "planned_auto_start": bool(
                    deployment_profile.get("planned_auto_start", True)
                ),
                "planned_risk_limits": (
                    dict(deployment_profile.get("planned_risk_limits"))
                    if isinstance(deployment_profile.get("planned_risk_limits"), dict)
                    else {}
                ),
                "prepared_at": confirmed_at,
            }
        )


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    return None


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
                "get_indicator_catalog",
            ],
            "require_approval": "never",
        }
    ]
