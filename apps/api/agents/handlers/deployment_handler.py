"""Deployment phase handler."""

from __future__ import annotations

import json
from decimal import Decimal, InvalidOperation
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.agents.handler_protocol import (
    PhaseContext,
    PostProcessResult,
    PromptPieces,
)
from apps.api.agents.phases import Phase
from apps.api.agents.skills.deployment_skills import (
    REQUIRED_FIELDS,
    VALID_BROKER_READINESS_VALUES,
    VALID_CONFIRMATION_VALUES,
    VALID_STATUS_VALUES,
    build_deployment_dynamic_state,
    build_deployment_static_instructions,
)
from packages.domain.trading.broker_capability_policy import capability_supports_market
from packages.infra.trading.broker_account_store import ensure_builtin_sandbox_account
from packages.shared_settings.schema.settings import settings


class DeploymentHandler:
    """Implements deployment readiness, broker selection, and confirmation."""

    @property
    def phase_name(self) -> str:
        return Phase.DEPLOYMENT.value

    @property
    def required_fields(self) -> list[str]:
        return list(REQUIRED_FIELDS)

    @property
    def valid_values(self) -> dict[str, set[str]]:
        return {
            "deployment_status": set(VALID_STATUS_VALUES),
            "broker_readiness_status": set(VALID_BROKER_READINESS_VALUES),
            "deployment_confirmation_status": set(VALID_CONFIRMATION_VALUES),
        }

    def build_prompt(
        self,
        ctx: PhaseContext,
        user_message: str,
    ) -> PromptPieces:
        phase_data = ctx.session_artifacts.get(Phase.DEPLOYMENT.value, {})
        profile = dict(phase_data.get("profile", {}))
        runtime_state = dict(phase_data.get("runtime", {}))
        missing = self._compute_missing(profile)

        instructions = build_deployment_static_instructions(language=ctx.language)
        state_block = build_deployment_dynamic_state(
            missing_fields=missing,
            collected_fields=profile,
            deployment_state=runtime_state,
            phase_stage=ctx.runtime_policy.phase_stage,
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
        artifacts = ctx.session_artifacts
        phase_data = artifacts.setdefault(
            Phase.DEPLOYMENT.value,
            self.init_artifacts(),
        )
        profile = self._ensure_profile_defaults(dict(phase_data.get("profile", {})))
        runtime_state = self._ensure_runtime_defaults(dict(phase_data.get("runtime", {})))
        previous_confirmation = self._normalize_confirmation_status(
            profile.get("deployment_confirmation_status")
        )

        for patch in raw_patches:
            validated = self._validate_patch(patch)
            if not validated:
                continue
            selected_changed = "selected_broker_account_id" in validated
            profile.update(validated)
            if selected_changed:
                profile["selected_broker_source"] = "user_choice"
                profile["deployment_confirmation_status"] = "pending"

        self._apply_tool_call_updates(
            profile=profile,
            runtime_state=runtime_state,
            tool_calls=ctx.turn_context.get("mcp_tool_calls"),
        )
        await self._apply_choice_selection(
            profile=profile,
            runtime_state=runtime_state,
            choice_selection=ctx.turn_context.get("choice_selection"),
            db=db,
            user_id=ctx.user_id,
        )
        self._sync_selected_broker_details(profile=profile, runtime_state=runtime_state)

        deployment_status = self._resolve_deployment_status(
            profile=profile,
            runtime_state=runtime_state,
        )
        profile["deployment_status"] = deployment_status

        summary_snapshot = self._build_deployment_summary_snapshot(
            profile=profile,
            runtime_state=runtime_state,
        )
        runtime_state["deployment_summary_snapshot"] = summary_snapshot

        confirmation = self._normalize_confirmation_status(
            profile.get("deployment_confirmation_status")
        )
        auto_execute_pending = False
        if deployment_status == "deployed":
            confirmation = "confirmed"
        elif confirmation == "confirmed" and previous_confirmation != "confirmed":
            auto_execute_pending = self._can_execute(profile)
        profile["deployment_confirmation_status"] = confirmation
        runtime_state["auto_execute_pending"] = auto_execute_pending

        runtime_state["deployment_status"] = deployment_status
        runtime_state["broker_readiness_status"] = self._normalize_broker_readiness_status(
            profile.get("broker_readiness_status")
        )
        runtime_state["selected_broker_account_id"] = profile.get("selected_broker_account_id")
        runtime_state["selected_broker_label"] = profile.get("selected_broker_label")
        runtime_state["selected_broker_source"] = profile.get("selected_broker_source")
        runtime_state["deployment_confirmation_status"] = confirmation
        runtime_state["planned_capital_allocated"] = profile.get("planned_capital_allocated")
        runtime_state["planned_auto_start"] = bool(profile.get("planned_auto_start", True))
        runtime_state["planned_risk_limits"] = (
            dict(profile.get("planned_risk_limits"))
            if isinstance(profile.get("planned_risk_limits"), dict)
            else {}
        )

        missing = self._compute_missing(profile)

        phase_data["profile"] = profile
        phase_data["runtime"] = runtime_state
        phase_data["missing_fields"] = missing

        return PostProcessResult(
            artifacts=artifacts,
            missing_fields=missing,
            completed=False,
            phase_status={"deployment_status": deployment_status},
        )

    def filter_genui(
        self,
        payload: dict[str, Any],
        ctx: PhaseContext,
    ) -> dict[str, Any] | None:
        if str(payload.get("type", "")).strip().lower() != "choice_prompt":
            return payload

        phase_data = ctx.session_artifacts.get(Phase.DEPLOYMENT.value, {})
        profile = self._ensure_profile_defaults(dict(phase_data.get("profile", {})))
        missing = self._compute_missing(profile)
        if not missing:
            return payload

        normalized = dict(payload)
        target_field = missing[0]
        if target_field in REQUIRED_FIELDS:
            normalized["choice_id"] = target_field
        return normalized

    def build_fallback_choice_prompt(
        self,
        *,
        missing_fields: list[str],
        ctx: PhaseContext,
    ) -> dict[str, Any] | None:
        target_field = next(
            (field for field in missing_fields if field in REQUIRED_FIELDS),
            None,
        )
        if target_field is None:
            return None

        phase_data = ctx.session_artifacts.get(Phase.DEPLOYMENT.value, {})
        profile = self._ensure_profile_defaults(dict(phase_data.get("profile", {})))
        runtime_state = self._ensure_runtime_defaults(dict(phase_data.get("runtime", {})))

        if target_field == "selected_broker_account_id":
            return self._build_broker_choice_prompt(
                profile=profile,
                runtime_state=runtime_state,
            )

        if target_field == "deployment_confirmation_status":
            return {
                "type": "choice_prompt",
                "choice_id": "deployment_confirmation_status",
                "question": "Confirm this deployment summary before execution.",
                "subtitle": (
                    "Review broker, market, symbols, timeframe, capital, and risk "
                    "settings. Confirming will let the agent execute deployment tools."
                ),
                "options": [
                    {
                        "id": "confirmed",
                        "label": "Confirm deployment",
                        "subtitle": "Proceed with broker and deployment creation.",
                    },
                    {
                        "id": "needs_changes",
                        "label": "Make changes",
                        "subtitle": "Adjust broker or deployment settings first.",
                    },
                ],
            }

        return None

    def init_artifacts(self) -> dict[str, Any]:
        profile = self._ensure_profile_defaults({})
        runtime = self._ensure_runtime_defaults({})
        return {
            "profile": profile,
            "missing_fields": list(REQUIRED_FIELDS),
            "runtime": runtime,
        }

    def build_phase_entry_guidance(self, ctx: PhaseContext) -> str | None:
        if ctx.language == "zh":
            return (
                "进入部署阶段：先检查可用 broker 与策略市场是否匹配，"
                "再让用户确认 deployment summary，确认后才执行部署工具。"
            )
        return (
            "Entering deployment phase: check broker readiness first, then confirm a "
            "deployment summary before executing deployment tools."
        )

    def _compute_missing(self, profile: dict[str, Any]) -> list[str]:
        deployment_status = self._normalize_deployment_status(profile.get("deployment_status"))
        if deployment_status == "deployed":
            return []

        broker_ready = self._normalize_broker_readiness_status(
            profile.get("broker_readiness_status")
        )
        selected_broker_id = self._normalize_uuid_text(profile.get("selected_broker_account_id"))
        if broker_ready != "ready" or selected_broker_id is None:
            return ["selected_broker_account_id"]

        confirmation = self._normalize_confirmation_status(
            profile.get("deployment_confirmation_status")
        )
        if confirmation != "confirmed":
            return ["deployment_confirmation_status"]
        return []

    def _validate_patch(self, patch: dict[str, Any]) -> dict[str, Any]:
        output: dict[str, Any] = {}

        raw_status = patch.get("deployment_status")
        if isinstance(raw_status, str):
            status_value = self._normalize_deployment_status(raw_status)
            if status_value in VALID_STATUS_VALUES:
                output["deployment_status"] = status_value

        raw_broker_status = patch.get("broker_readiness_status")
        if isinstance(raw_broker_status, str):
            broker_status = self._normalize_broker_readiness_status(raw_broker_status)
            if broker_status in VALID_BROKER_READINESS_VALUES:
                output["broker_readiness_status"] = broker_status

        raw_selected_broker = patch.get("selected_broker_account_id")
        selected_broker_id = self._normalize_uuid_text(raw_selected_broker)
        if selected_broker_id is not None:
            output["selected_broker_account_id"] = selected_broker_id

        raw_confirmation = patch.get("deployment_confirmation_status")
        if isinstance(raw_confirmation, str):
            confirmation = self._normalize_confirmation_status(raw_confirmation)
            if confirmation in VALID_CONFIRMATION_VALUES:
                output["deployment_confirmation_status"] = confirmation

        normalized_capital = self._normalize_capital_value(
            patch.get("planned_capital_allocated")
        )
        if normalized_capital is not None:
            output["planned_capital_allocated"] = normalized_capital

        normalized_auto_start = _coerce_bool(patch.get("planned_auto_start"))
        if normalized_auto_start is not None:
            output["planned_auto_start"] = normalized_auto_start

        risk_limits = patch.get("planned_risk_limits")
        if isinstance(risk_limits, dict):
            output["planned_risk_limits"] = dict(risk_limits)

        return output

    def _apply_tool_call_updates(
        self,
        *,
        profile: dict[str, Any],
        runtime_state: dict[str, Any],
        tool_calls: Any,
    ) -> None:
        if not isinstance(tool_calls, list):
            return

        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            name = str(call.get("name") or call.get("tool_name") or "").strip()
            if not name:
                continue

            payload = _coerce_json_object(call.get("output"))
            if not isinstance(payload, dict):
                continue
            if payload.get("ok") is False:
                continue

            data = _extract_tool_data_payload(payload)
            if not isinstance(data, dict):
                continue

            if name == "trading_list_broker_accounts":
                accounts = data.get("accounts")
                if isinstance(accounts, list):
                    runtime_state["broker_accounts"] = list(accounts)
                default_id = self._normalize_uuid_text(data.get("default_broker_account_id"))
                if default_id is not None:
                    runtime_state["default_broker_account_id"] = default_id
                continue

            if name == "trading_check_deployment_readiness":
                self._apply_readiness_payload(
                    profile=profile,
                    runtime_state=runtime_state,
                    data=data,
                )
                continue

            if name == "trading_create_builtin_sandbox_broker_account":
                self._apply_builtin_sandbox_payload(
                    profile=profile,
                    runtime_state=runtime_state,
                    data=data,
                )
                continue

            if name == "trading_list_deployments":
                deployments = data.get("deployments")
                if isinstance(deployments, list):
                    runtime_state["deployment_inventory"] = list(deployments)
                    active = [
                        item
                        for item in deployments
                        if isinstance(item, dict)
                        and str(item.get("status", "")).strip().lower() == "active"
                    ]
                    runtime_state["active_deployments"] = active
                    if deployments:
                        runtime_state["latest_deployment"] = deployments[0]
                continue

            if name in {
                "trading_create_paper_deployment",
                "trading_start_deployment",
                "trading_pause_deployment",
                "trading_stop_deployment",
            }:
                deployment = data.get("deployment")
                if isinstance(deployment, dict):
                    runtime_state["latest_deployment"] = dict(deployment)
                    deployment_id = self._normalize_uuid_text(
                        deployment.get("deployment_id") or deployment.get("id")
                    )
                    if deployment_id is not None:
                        runtime_state["latest_deployment_id"] = deployment_id
                        profile["latest_deployment_id"] = deployment_id
                    status = str(deployment.get("status", "")).strip().lower()
                    if status == "active":
                        runtime_state["active_deployments"] = [dict(deployment)]
                    elif status:
                        runtime_state.setdefault("active_deployments", [])
                resolved_broker = self._normalize_uuid_text(
                    data.get("resolved_broker_account_id")
                )
                if resolved_broker is not None:
                    profile["selected_broker_account_id"] = resolved_broker
                continue

    def _apply_readiness_payload(
        self,
        *,
        profile: dict[str, Any],
        runtime_state: dict[str, Any],
        data: dict[str, Any],
    ) -> None:
        accounts = data.get("accounts")
        if isinstance(accounts, list):
            runtime_state["broker_accounts"] = list(accounts)

        matched_accounts_raw = data.get("matched_accounts")
        matched_accounts = (
            [item for item in matched_accounts_raw if isinstance(item, dict)]
            if isinstance(matched_accounts_raw, list)
            else []
        )
        runtime_state["broker_readiness"] = dict(data)
        runtime_state["available_broker_choices"] = list(matched_accounts)

        default_id = self._normalize_uuid_text(data.get("default_broker_account_id"))
        if default_id is not None:
            runtime_state["default_broker_account_id"] = default_id

        broker_status = self._normalize_broker_readiness_status(data.get("status"))
        current_selected = self._normalize_uuid_text(profile.get("selected_broker_account_id"))

        matched_ids: list[str] = []
        for item in matched_accounts:
            account_id = self._normalize_uuid_text(item.get("broker_account_id"))
            if account_id is not None:
                matched_ids.append(account_id)

        if current_selected is not None and current_selected in matched_ids:
            profile["broker_readiness_status"] = "ready"
            return

        if broker_status == "ready":
            preferred_id = self._normalize_uuid_text(data.get("preferred_broker_account_id"))
            selected_broker_id = preferred_id
            if selected_broker_id is None and len(matched_ids) == 1:
                selected_broker_id = matched_ids[0]
            if selected_broker_id is not None:
                profile["selected_broker_account_id"] = selected_broker_id
                profile.setdefault("selected_broker_source", "default")
            profile["broker_readiness_status"] = "ready"
            return

        if broker_status in {"needs_choice", "no_broker", "blocked"}:
            if current_selected is not None and current_selected not in matched_ids:
                profile.pop("selected_broker_account_id", None)
                profile.pop("selected_broker_label", None)
                profile.pop("selected_broker_source", None)
                profile["deployment_confirmation_status"] = "pending"
            if broker_status == "blocked" and matched_ids:
                if len(matched_ids) == 1:
                    profile["selected_broker_account_id"] = matched_ids[0]
                    profile.setdefault("selected_broker_source", "default")
                    profile["broker_readiness_status"] = "ready"
                    return
                broker_status = "needs_choice"
            profile["broker_readiness_status"] = broker_status

    async def _apply_choice_selection(
        self,
        *,
        profile: dict[str, Any],
        runtime_state: dict[str, Any],
        choice_selection: Any,
        db: AsyncSession,
        user_id: UUID,
    ) -> None:
        if not isinstance(choice_selection, dict):
            return

        choice_id = str(choice_selection.get("choice_id") or "").strip()
        option_id = str(choice_selection.get("selected_option_id") or "").strip()
        if not choice_id or not option_id:
            return

        if choice_id == "deployment_confirmation_status":
            if self._can_accept_confirmation(profile):
                confirmation = self._normalize_confirmation_status(option_id)
                if confirmation in VALID_CONFIRMATION_VALUES:
                    profile["deployment_confirmation_status"] = confirmation
            return

        if choice_id != "selected_broker_account_id":
            return

        selected_broker_id = self._normalize_uuid_text(option_id)
        if selected_broker_id is not None:
            profile["selected_broker_account_id"] = selected_broker_id
            profile["selected_broker_source"] = "user_choice"
            profile["deployment_confirmation_status"] = "pending"
            return

        if option_id == "create_builtin_sandbox":
            account = await ensure_builtin_sandbox_account(db, user_id=user_id)
            self._apply_builtin_sandbox_model(
                profile=profile,
                runtime_state=runtime_state,
                account=account,
            )

    def _apply_builtin_sandbox_payload(
        self,
        *,
        profile: dict[str, Any],
        runtime_state: dict[str, Any],
        data: dict[str, Any],
    ) -> None:
        account = data.get("broker_account")
        if not isinstance(account, dict):
            return

        account_id = self._normalize_uuid_text(
            account.get("broker_account_id") or account.get("id")
        )
        if account_id is None:
            return

        profile["selected_broker_account_id"] = account_id
        profile["selected_broker_label"] = "Built-in Sandbox"
        profile["selected_broker_source"] = "builtin_sandbox"
        profile["deployment_confirmation_status"] = "pending"

        capabilities = account.get("capabilities")
        strategy_market = str(profile.get("strategy_market") or "").strip()
        if capability_supports_market(
            capabilities=capabilities if isinstance(capabilities, dict) else {},
            market=strategy_market or None,
        ):
            profile["broker_readiness_status"] = "ready"
        else:
            profile["broker_readiness_status"] = "blocked"

        runtime_state["builtin_sandbox_account"] = dict(account)

    def _apply_builtin_sandbox_model(
        self,
        *,
        profile: dict[str, Any],
        runtime_state: dict[str, Any],
        account: Any,
    ) -> None:
        capabilities = (
            dict(account.capabilities)
            if isinstance(getattr(account, "capabilities", None), dict)
            else {}
        )
        summary = {
            "broker_account_id": str(account.id),
            "id": str(account.id),
            "label": "Built-in Sandbox",
            "provider": "sandbox",
            "exchange_id": "sandbox",
            "status": str(getattr(account, "status", "active")),
            "is_default": bool(getattr(account, "is_default", False)),
            "is_sandbox": True,
            "capabilities": capabilities,
        }
        runtime_state["builtin_sandbox_account"] = summary
        self._upsert_runtime_broker_account(runtime_state=runtime_state, account=summary)
        self._apply_selected_broker_choice(
            profile=profile,
            runtime_state=runtime_state,
            account=summary,
        )

    def _apply_selected_broker_choice(
        self,
        *,
        profile: dict[str, Any],
        runtime_state: dict[str, Any],
        account: dict[str, Any],
    ) -> None:
        selected_broker_id = self._normalize_uuid_text(
            account.get("broker_account_id") or account.get("id")
        )
        if selected_broker_id is None:
            return

        label = str(account.get("label") or "").strip() or None
        profile["selected_broker_account_id"] = selected_broker_id
        profile["selected_broker_label"] = label
        profile["selected_broker_source"] = "user_choice"
        profile["deployment_confirmation_status"] = "pending"
        runtime_state["selected_broker_account_id"] = selected_broker_id
        if label is not None:
            runtime_state["selected_broker_label"] = label

        strategy_market = str(profile.get("strategy_market") or "").strip()
        capabilities = account.get("capabilities")
        if capability_supports_market(
            capabilities=capabilities if isinstance(capabilities, dict) else {},
            market=strategy_market or None,
        ):
            profile["broker_readiness_status"] = "ready"
        else:
            profile["broker_readiness_status"] = "blocked"

    def _upsert_runtime_broker_account(
        self,
        *,
        runtime_state: dict[str, Any],
        account: dict[str, Any],
    ) -> None:
        account_id = self._normalize_uuid_text(
            account.get("broker_account_id") or account.get("id")
        )
        if account_id is None:
            return
        current = runtime_state.get("broker_accounts")
        items = (
            [item for item in current if isinstance(item, dict)]
            if isinstance(current, list)
            else []
        )
        filtered = [
            item
            for item in items
            if self._normalize_uuid_text(item.get("broker_account_id") or item.get("id"))
            != account_id
        ]
        filtered.append(dict(account))
        runtime_state["broker_accounts"] = filtered

    def _sync_selected_broker_details(
        self,
        *,
        profile: dict[str, Any],
        runtime_state: dict[str, Any],
    ) -> None:
        selected_broker_id = self._normalize_uuid_text(profile.get("selected_broker_account_id"))
        if selected_broker_id is None:
            profile.pop("selected_broker_label", None)
            return

        candidates: list[dict[str, Any]] = []
        for key in ("available_broker_choices", "broker_accounts"):
            raw = runtime_state.get(key)
            if isinstance(raw, list):
                candidates.extend(item for item in raw if isinstance(item, dict))

        builtin_account = runtime_state.get("builtin_sandbox_account")
        if isinstance(builtin_account, dict):
            candidates.append(builtin_account)

        for account in candidates:
            account_id = self._normalize_uuid_text(
                account.get("broker_account_id") or account.get("id")
            )
            if account_id != selected_broker_id:
                continue
            label = str(account.get("label") or "").strip()
            if not label and str(account.get("provider", "")).strip().lower() == "sandbox":
                label = "Built-in Sandbox"
            if label:
                profile["selected_broker_label"] = label
            provider = str(account.get("provider") or "").strip().lower()
            if profile.get("selected_broker_source") is None:
                profile["selected_broker_source"] = "default" if bool(
                    account.get("is_default")
                ) else provider or "user_choice"
            return

    def _resolve_deployment_status(
        self,
        *,
        profile: dict[str, Any],
        runtime_state: dict[str, Any],
    ) -> str:
        active_deployments = runtime_state.get("active_deployments")
        if isinstance(active_deployments, list) and active_deployments:
            return "deployed"

        latest_deployment = runtime_state.get("latest_deployment")
        if isinstance(latest_deployment, dict):
            latest_status = str(latest_deployment.get("status", "")).strip().lower()
            if latest_status == "active":
                return "deployed"

        if self._normalize_broker_readiness_status(profile.get("broker_readiness_status")) == "ready":
            if self._normalize_uuid_text(profile.get("selected_broker_account_id")) is not None:
                return "ready"
        return "blocked"

    def _build_deployment_summary_snapshot(
        self,
        *,
        profile: dict[str, Any],
        runtime_state: dict[str, Any],
    ) -> dict[str, Any]:
        selected_account = self._find_selected_broker_account(
            profile=profile,
            runtime_state=runtime_state,
        )
        latest_deployment = runtime_state.get("latest_deployment")
        deployment = dict(latest_deployment) if isinstance(latest_deployment, dict) else {}
        strategy_symbols = _normalize_symbols(
            profile.get("strategy_tickers"),
            profile.get("strategy_tickers_csv"),
            profile.get("strategy_primary_symbol"),
        )
        summary = {
            "strategy_name": _clean_text(profile.get("strategy_name")),
            "market": _clean_text(profile.get("strategy_market")),
            "symbols": strategy_symbols,
            "timeframe": _clean_text(profile.get("strategy_timeframe")),
            "selected_broker": _clean_text(profile.get("selected_broker_label")),
            "selected_broker_account_id": self._normalize_uuid_text(
                profile.get("selected_broker_account_id")
            ),
            "broker_provider": _clean_text(selected_account.get("provider")),
            "broker_exchange_id": _clean_text(selected_account.get("exchange_id")),
            "deployment_mode": "paper",
            "capital_allocated": _clean_text(
                profile.get("planned_capital_allocated")
            )
            or "10000",
            "risk_limits": (
                dict(profile.get("planned_risk_limits"))
                if isinstance(profile.get("planned_risk_limits"), dict)
                else {}
            ),
            "will_auto_start": bool(profile.get("planned_auto_start", True)),
            "broker_readiness_status": self._normalize_broker_readiness_status(
                profile.get("broker_readiness_status")
            ),
            "deployment_confirmation_status": self._normalize_confirmation_status(
                profile.get("deployment_confirmation_status")
            ),
            "deployment_status": self._normalize_deployment_status(
                profile.get("deployment_status")
            ),
        }
        if deployment:
            summary["latest_deployment"] = deployment
        blockers = []
        broker_readiness = runtime_state.get("broker_readiness")
        if isinstance(broker_readiness, dict):
            raw_blockers = broker_readiness.get("blockers")
            if isinstance(raw_blockers, list):
                blockers = [str(item).strip() for item in raw_blockers if str(item).strip()]
        if blockers:
            summary["blockers"] = blockers
        return summary

    def _build_broker_choice_prompt(
        self,
        *,
        profile: dict[str, Any],
        runtime_state: dict[str, Any],
    ) -> dict[str, Any] | None:
        broker_status = self._normalize_broker_readiness_status(
            profile.get("broker_readiness_status")
        )
        strategy_market = _clean_text(profile.get("strategy_market"))

        if broker_status == "needs_choice":
            raw_accounts = runtime_state.get("available_broker_choices")
            option_source = (
                [item for item in raw_accounts if isinstance(item, dict)]
                if isinstance(raw_accounts, list)
                else []
            )
            options = []
            for account in option_source:
                account_id = self._normalize_uuid_text(account.get("broker_account_id"))
                label = str(account.get("label") or "").strip()
                if account_id is None or not label:
                    continue
                subtitle_parts = [
                    part
                    for part in (
                        str(account.get("provider") or "").strip(),
                        str(account.get("exchange_id") or "").strip(),
                    )
                    if part
                ]
                options.append(
                    {
                        "id": account_id,
                        "label": label,
                        "subtitle": " / ".join(subtitle_parts) or account_id,
                    }
                )
            if len(options) < 2:
                return None
            return {
                "type": "choice_prompt",
                "choice_id": "selected_broker_account_id",
                "question": (
                    "Choose which connected broker should be used for this deployment."
                ),
                "subtitle": (
                    "Multiple connected brokers support the current strategy market. "
                    "Select the one you want before deployment can continue."
                ),
                "options": options,
            }

        if broker_status in {"no_broker", "blocked"}:
            subtitle = (
                "Open Settings > Broker Connectors, choose a broker, and follow the "
                "credential prompts. You can also create the built-in sandbox account "
                "from chat if this strategy targets us_stocks or crypto."
            )
            options = []
            if strategy_market in {"us_stocks", "crypto"}:
                options.append(
                    {
                        "id": "create_builtin_sandbox",
                        "label": "Use built-in sandbox",
                        "subtitle": "Create or reactivate the platform sandbox broker.",
                    }
                )
            options.append(
                {
                    "id": "open_broker_connectors",
                    "label": "Open Broker Connectors",
                    "subtitle": "Go to Settings > Broker Connectors to bind a broker.",
                }
            )
            options.append(
                {
                    "id": "modify_strategy_scope",
                    "label": "Modify strategy scope",
                    "subtitle": "Change the market or instrument if needed.",
                }
            )
            return {
                "type": "choice_prompt",
                "choice_id": "selected_broker_account_id",
                "question": (
                    "No compatible broker is ready for this deployment. How would you like to proceed?"
                ),
                "subtitle": subtitle,
                "options": options,
            }

        return None

    def _can_execute(self, profile: dict[str, Any]) -> bool:
        return (
            self._normalize_broker_readiness_status(profile.get("broker_readiness_status"))
            == "ready"
            and self._normalize_uuid_text(profile.get("selected_broker_account_id"))
            is not None
            and self._normalize_confirmation_status(
                profile.get("deployment_confirmation_status")
            )
            == "confirmed"
        )

    def _can_accept_confirmation(self, profile: dict[str, Any]) -> bool:
        return (
            self._normalize_broker_readiness_status(profile.get("broker_readiness_status"))
            == "ready"
            and self._normalize_uuid_text(profile.get("selected_broker_account_id"))
            is not None
        )

    def _find_selected_broker_account(
        self,
        *,
        profile: dict[str, Any],
        runtime_state: dict[str, Any],
    ) -> dict[str, Any]:
        selected_broker_id = self._normalize_uuid_text(profile.get("selected_broker_account_id"))
        if selected_broker_id is None:
            return {}
        for key in ("available_broker_choices", "broker_accounts"):
            raw = runtime_state.get(key)
            if not isinstance(raw, list):
                continue
            for item in raw:
                if not isinstance(item, dict):
                    continue
                account_id = self._normalize_uuid_text(
                    item.get("broker_account_id") or item.get("id")
                )
                if account_id == selected_broker_id:
                    return dict(item)
        builtin_account = runtime_state.get("builtin_sandbox_account")
        if isinstance(builtin_account, dict):
            account_id = self._normalize_uuid_text(
                builtin_account.get("broker_account_id") or builtin_account.get("id")
            )
            if account_id == selected_broker_id:
                return dict(builtin_account)
        return {}

    @staticmethod
    def _normalize_uuid_text(value: Any) -> str | None:
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
    def _normalize_deployment_status(value: Any) -> str:
        text = str(value or "").strip().lower()
        if text in VALID_STATUS_VALUES:
            return text
        return "blocked"

    @staticmethod
    def _normalize_broker_readiness_status(value: Any) -> str:
        text = str(value or "").strip().lower()
        if text in VALID_BROKER_READINESS_VALUES:
            return text
        return "unknown"

    @staticmethod
    def _normalize_confirmation_status(value: Any) -> str:
        text = str(value or "").strip().lower()
        if text in VALID_CONFIRMATION_VALUES:
            return text
        return "pending"

    @staticmethod
    def _normalize_capital_value(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
        else:
            text = str(value).strip()
        try:
            amount = Decimal(text)
        except (InvalidOperation, ValueError):
            return None
        if amount <= 0:
            return None
        return format(amount.normalize(), "f")

    def _ensure_profile_defaults(self, profile: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(profile)
        normalized["deployment_status"] = self._normalize_deployment_status(
            normalized.get("deployment_status")
        )
        normalized["broker_readiness_status"] = self._normalize_broker_readiness_status(
            normalized.get("broker_readiness_status")
        )
        normalized["deployment_confirmation_status"] = self._normalize_confirmation_status(
            normalized.get("deployment_confirmation_status")
        )
        normalized.setdefault("planned_capital_allocated", "10000")
        normalized.setdefault("planned_auto_start", True)
        if not isinstance(normalized.get("planned_risk_limits"), dict):
            normalized["planned_risk_limits"] = {}
        return normalized

    def _ensure_runtime_defaults(self, runtime_state: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(runtime_state)
        normalized.setdefault("deployment_status", "blocked")
        normalized.setdefault("broker_readiness_status", "unknown")
        normalized.setdefault("deployment_confirmation_status", "pending")
        normalized.setdefault("planned_capital_allocated", "10000")
        normalized.setdefault("planned_auto_start", True)
        normalized.setdefault("planned_risk_limits", {})
        normalized.setdefault("deployment_summary_snapshot", {})
        normalized.setdefault("auto_execute_pending", False)
        return normalized


def _coerce_json_object(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return dict(value)
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _extract_tool_data_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    nested = payload.get("data")
    if isinstance(nested, dict):
        return dict(nested)

    flattened = {
        key: value
        for key, value in payload.items()
        if key not in {"category", "tool", "ok", "timestamp_utc", "error", "data"}
    }
    if flattened:
        return flattened
    return None


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


def _clean_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _normalize_symbols(
    tickers: Any,
    tickers_csv: Any,
    primary_symbol: Any,
) -> list[str]:
    if isinstance(tickers, list):
        output = [str(item).strip() for item in tickers if str(item).strip()]
        if output:
            return output
    if isinstance(tickers_csv, str) and tickers_csv.strip():
        output = [part.strip() for part in tickers_csv.split(",") if part.strip()]
        if output:
            return output
    symbol = _clean_text(primary_symbol)
    if symbol is not None:
        return [symbol]
    return []
