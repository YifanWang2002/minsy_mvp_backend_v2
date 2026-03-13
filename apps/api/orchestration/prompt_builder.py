"""Orchestrator mixin extracted from legacy implementation."""

from __future__ import annotations

from .shared import *  # noqa: F403


class PromptBuilderMixin:
    async def _prepare_turn_context(
        self,
        *,
        session: Session,
        user: User,
        payload: ChatSendRequest,
        language: str,
        phase_before: str,
        user_runtime_policy: HandlerRuntimePolicy,
        prompt_user_message: str,
        phase_turn_count: int,
        handler: Any,
        turn_request_id: str,
        turn_id: str,
        user_message_id: UUID,
    ) -> _TurnPreparation:
        artifacts = copy.deepcopy(session.artifacts or {})
        artifacts = self._ensure_phase_keyed(artifacts)
        await self._hydrate_strategy_context(
            session=session,
            user_id=user.id,
            phase=phase_before,
            artifacts=artifacts,
            user_message=payload.message,
        )
        await self._hydrate_deployment_defaults(
            artifacts=artifacts,
            phase=phase_before,
            user_id=user.id,
        )
        pre_strategy_instrument_before: str | None = None
        if phase_before == Phase.PRE_STRATEGY.value:
            pre_data = artifacts.get(Phase.PRE_STRATEGY.value)
            if isinstance(pre_data, dict):
                pre_profile = pre_data.get("profile")
                if isinstance(pre_profile, dict):
                    value = pre_profile.get("target_instrument")
                    if isinstance(value, str) and value.strip():
                        pre_strategy_instrument_before = value.strip()

        effective_prompt_user_message = await self._maybe_reset_response_chain(
            session=session,
            phase=phase_before,
            phase_turn_count=phase_turn_count,
            user_message_id=user_message_id,
            prompt_user_message=prompt_user_message,
        )

        runtime_policy = self._resolve_runtime_policy(
            phase=phase_before,
            artifacts=artifacts,
            user_runtime_policy=user_runtime_policy,
        )
        choice_selection = self._extract_choice_selection_from_message(payload.message)
        ctx = PhaseContext(
            user_id=user.id,
            session_artifacts=artifacts,
            session_id=session.id,
            language=language,
            runtime_policy=runtime_policy,
            turn_context={
                "choice_selection": choice_selection,
            },
        )
        prompt = handler.build_prompt(ctx, effective_prompt_user_message)
        prompt = self._apply_execution_policy(
            session=session,
            phase=phase_before,
            payload=payload,
            prompt=prompt,
        )
        prompt = self._enforce_prompt_budget(prompt)
        tools = self._merge_tools(
            base_tools=prompt.tools,
            runtime_policy=runtime_policy,
        )
        tools = self._attach_mcp_context_headers(
            tools=tools,
            user_id=user.id,
            session_id=session.id,
            phase=phase_before,
            request_id=turn_request_id,
        )
        return _TurnPreparation(
            turn_id=turn_id,
            user_message_id=user_message_id,
            phase_before=phase_before,
            phase_turn_count=phase_turn_count,
            prompt_user_message=effective_prompt_user_message,
            handler=handler,
            artifacts=artifacts,
            pre_strategy_instrument_before=pre_strategy_instrument_before,
            ctx=ctx,
            prompt=prompt,
            tools=tools,
        )

    async def _hydrate_deployment_defaults(
        self,
        *,
        artifacts: dict[str, Any],
        phase: str,
        user_id: UUID,
    ) -> None:
        if phase != Phase.DEPLOYMENT.value:
            return
        from apps.api.agents.deployment_defaults import (
            hydrate_deployment_profile_defaults,
        )
        from packages.domain.trading.services.trading_preference_service import (
            TradingPreferenceService,
        )

        deployment_block = artifacts.get(Phase.DEPLOYMENT.value)
        if not isinstance(deployment_block, dict):
            return
        profile = (
            dict(deployment_block.get("profile"))
            if isinstance(deployment_block.get("profile"), dict)
            else {}
        )
        runtime_state = (
            dict(deployment_block.get("runtime"))
            if isinstance(deployment_block.get("runtime"), dict)
            else {}
        )

        preference_view = await TradingPreferenceService(self.db).get_view(user_id=user_id)
        deploy_defaults = (
            dict(preference_view.deploy_defaults)
            if isinstance(preference_view.deploy_defaults, dict)
            else {}
        )
        profile, runtime_state = hydrate_deployment_profile_defaults(
            profile=profile,
            runtime_state=runtime_state,
            deploy_defaults=deploy_defaults,
        )
        deployment_block["profile"] = profile
        deployment_block["runtime"] = runtime_state

    def _resolve_runtime_policy(
        self,
        *,
        phase: str,
        artifacts: dict[str, Any],
        user_runtime_policy: HandlerRuntimePolicy,
    ) -> HandlerRuntimePolicy:
        phase_policy = self._build_phase_runtime_policy(
            phase=phase, artifacts=artifacts
        )
        return self._merge_runtime_policies(
            phase_policy=phase_policy,
            user_policy=user_runtime_policy,
        )

    def _build_phase_runtime_policy(
        self,
        *,
        phase: str,
        artifacts: dict[str, Any],
    ) -> HandlerRuntimePolicy:
        if phase == Phase.STRATEGY.value:
            return self._build_strategy_runtime_policy(artifacts=artifacts)
        if phase == Phase.STRESS_TEST.value:
            return self._build_stress_test_runtime_policy(artifacts=artifacts)
        if phase == Phase.DEPLOYMENT.value:
            return self._build_deployment_runtime_policy(artifacts=artifacts)
        return HandlerRuntimePolicy()

    def _build_strategy_runtime_policy(
        self,
        *,
        artifacts: dict[str, Any],
    ) -> HandlerRuntimePolicy:
        profile = self._extract_phase_profile(
            artifacts=artifacts, phase=Phase.STRATEGY.value
        )
        strategy_id = self._coerce_uuid_text(profile.get("strategy_id"))
        if strategy_id is None:
            stress_profile = self._extract_phase_profile(
                artifacts=artifacts,
                phase=Phase.STRESS_TEST.value,
            )
            strategy_id = self._coerce_uuid_text(stress_profile.get("strategy_id"))

        if strategy_id is not None:
            return HandlerRuntimePolicy(
                phase_stage="artifact_ops",
                tool_mode="replace",
                allowed_tools=self._build_strategy_artifact_ops_allowed_tools(),
            )

        return HandlerRuntimePolicy(
            phase_stage="schema_only",
            tool_mode="replace",
            allowed_tools=[
                self._build_strategy_tool_def(
                    allowed_tools=list(_STRATEGY_SCHEMA_ONLY_TOOL_NAMES),
                ),
                self._build_market_data_tool_def(
                    allowed_tools=list(_STRATEGY_MARKET_DATA_TOOL_NAMES),
                ),
            ],
        )

    def _build_stress_test_runtime_policy(
        self,
        *,
        artifacts: dict[str, Any],
    ) -> HandlerRuntimePolicy:
        profile = self._extract_phase_profile(
            artifacts=artifacts, phase=Phase.STRESS_TEST.value
        )
        raw_status = profile.get("backtest_status")
        status = raw_status.strip().lower() if isinstance(raw_status, str) else ""

        if status == "done":
            return HandlerRuntimePolicy(
                phase_stage="feedback",
                tool_mode="replace",
                allowed_tools=self._build_stress_feedback_allowed_tools(),
            )

        return HandlerRuntimePolicy(
            phase_stage="bootstrap",
            tool_mode="replace",
            allowed_tools=[
                self._build_market_data_tool_def(
                    allowed_tools=list(_MARKET_DATA_MINIMAL_TOOL_NAMES),
                ),
                self._build_backtest_tool_def(
                    allowed_tools=list(_BACKTEST_BOOTSTRAP_TOOL_NAMES),
                ),
            ],
        )

    def _build_strategy_artifact_ops_allowed_tools(self) -> list[dict[str, Any]]:
        return [
            self._build_strategy_tool_def(
                allowed_tools=list(_STRATEGY_ARTIFACT_OPS_TOOL_NAMES),
            ),
            self._build_market_data_tool_def(
                allowed_tools=list(_STRATEGY_MARKET_DATA_TOOL_NAMES),
            ),
            self._build_backtest_tool_def(
                allowed_tools=list(_BACKTEST_FEEDBACK_TOOL_NAMES),
            ),
        ]

    def _build_stress_feedback_allowed_tools(self) -> list[dict[str, Any]]:
        return [
            self._build_market_data_tool_def(
                allowed_tools=list(_MARKET_DATA_MINIMAL_TOOL_NAMES),
            ),
            self._build_backtest_tool_def(
                allowed_tools=list(_BACKTEST_FEEDBACK_TOOL_NAMES),
            ),
            self._build_strategy_tool_def(
                allowed_tools=list(_STRATEGY_ARTIFACT_OPS_TOOL_NAMES),
            ),
        ]

    def _build_deployment_runtime_policy(
        self,
        *,
        artifacts: dict[str, Any],
    ) -> HandlerRuntimePolicy:
        profile = self._extract_phase_profile(
            artifacts=artifacts,
            phase=Phase.DEPLOYMENT.value,
        )
        raw_status = profile.get("deployment_status")
        status = raw_status.strip().lower() if isinstance(raw_status, str) else "blocked"
        if status not in {"ready", "deployed", "blocked"}:
            status = "blocked"

        raw_broker_status = profile.get("broker_readiness_status")
        broker_status = (
            raw_broker_status.strip().lower()
            if isinstance(raw_broker_status, str)
            else "unknown"
        )
        if broker_status not in {"unknown", "no_broker", "needs_choice", "ready", "blocked"}:
            broker_status = "unknown"

        raw_confirmation = profile.get("deployment_confirmation_status")
        confirmation_status = (
            raw_confirmation.strip().lower()
            if isinstance(raw_confirmation, str)
            else "pending"
        )
        if confirmation_status not in {"pending", "confirmed", "needs_changes"}:
            confirmation_status = "pending"

        allowed_tools = list(_TRADING_DEPLOYMENT_TOOL_NAMES)

        if status == "deployed":
            phase_stage = "deployment_deployed"
        elif broker_status in {"no_broker", "blocked"}:
            phase_stage = "deployment_preflight_blocked"
        elif broker_status == "needs_choice":
            phase_stage = "deployment_needs_broker_choice"
        elif broker_status == "ready" and confirmation_status != "confirmed":
            phase_stage = "deployment_review_pending"
        elif broker_status == "ready" and confirmation_status == "confirmed":
            phase_stage = "deployment_execute_ready"
        else:
            phase_stage = "deployment_preflight"

        return HandlerRuntimePolicy(
            phase_stage=phase_stage,
            tool_mode="replace",
            allowed_tools=[
                self._build_trading_tool_def(
                    allowed_tools=allowed_tools,
                )
            ],
        )

    @staticmethod
    def _extract_phase_profile(
        *, artifacts: dict[str, Any], phase: str
    ) -> dict[str, Any]:
        phase_block = artifacts.get(phase)
        if not isinstance(phase_block, dict):
            return {}
        profile = phase_block.get("profile")
        if not isinstance(profile, dict):
            return {}
        return dict(profile)

    @staticmethod
    def _build_strategy_tool_def(*, allowed_tools: list[str]) -> dict[str, Any]:
        return {
            "type": "mcp",
            "server_label": "strategy",
            "server_url": settings.strategy_mcp_server_url,
            "allowed_tools": allowed_tools,
            "require_approval": "never",
        }

    @staticmethod
    def _build_backtest_tool_def(*, allowed_tools: list[str]) -> dict[str, Any]:
        return {
            "type": "mcp",
            "server_label": "backtest",
            "server_url": settings.backtest_mcp_server_url,
            "allowed_tools": allowed_tools,
            "require_approval": "never",
        }

    @staticmethod
    def _build_market_data_tool_def(*, allowed_tools: list[str]) -> dict[str, Any]:
        return {
            "type": "mcp",
            "server_label": "market_data",
            "server_url": settings.market_data_mcp_server_url,
            "allowed_tools": allowed_tools,
            "require_approval": "never",
        }

    @staticmethod
    def _build_trading_tool_def(*, allowed_tools: list[str]) -> dict[str, Any]:
        return {
            "type": "mcp",
            "server_label": "trading",
            "server_url": settings.trading_mcp_server_url,
            "allowed_tools": allowed_tools,
            "require_approval": "never",
        }

    def _merge_tools(
        self,
        *,
        base_tools: list[dict[str, Any]] | None,
        runtime_policy: HandlerRuntimePolicy,
    ) -> list[dict[str, Any]] | None:
        merged = list(base_tools or [])

        requested_tools = runtime_policy.allowed_tools or []
        if requested_tools:
            if runtime_policy.tool_mode == "replace":
                merged = list(requested_tools)
            else:
                merged.extend(requested_tools)

        return merged or None

    def _attach_mcp_context_headers(
        self,
        *,
        tools: list[dict[str, Any]] | None,
        user_id: UUID,
        session_id: UUID,
        phase: str,
        request_id: str | None = None,
    ) -> list[dict[str, Any]] | None:
        if not tools:
            return tools

        trace = get_chat_debug_trace()
        trace_id = trace.trace_id if trace and isinstance(trace.trace_id, str) else None
        token = create_mcp_context_token(
            user_id=user_id,
            session_id=session_id,
            ttl_seconds=settings.mcp_context_ttl_seconds,
            trace_id=trace_id,
            phase=phase,
            request_id=request_id,
        )

        decorated: list[dict[str, Any]] = []
        for item in tools:
            if not isinstance(item, dict):
                decorated.append(item)
                continue
            if str(item.get("type", "")).strip().lower() != "mcp":
                decorated.append(dict(item))
                continue
            server_label = str(item.get("server_label", "")).strip().lower()
            if server_label not in _MCP_CONTEXT_ENABLED_SERVER_LABELS:
                decorated.append(dict(item))
                continue

            next_tool = dict(item)
            raw_headers = item.get("headers")
            headers = dict(raw_headers) if isinstance(raw_headers, dict) else {}
            headers[MCP_CONTEXT_HEADER] = token
            next_tool["headers"] = headers
            decorated.append(next_tool)
        return decorated

    @staticmethod
    def _redact_stream_request_kwargs_for_trace(
        stream_request_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        payload = copy.deepcopy(stream_request_kwargs)
        tools = payload.get("tools")
        if not isinstance(tools, list):
            return payload

        redacted_tools: list[Any] = []
        for item in tools:
            if not isinstance(item, dict):
                redacted_tools.append(item)
                continue

            next_tool = dict(item)
            raw_headers = next_tool.get("headers")
            if isinstance(raw_headers, dict):
                sanitized_headers: dict[str, Any] = {}
                for key, value in raw_headers.items():
                    normalized_key = str(key).strip().lower()
                    if normalized_key in {"authorization", MCP_CONTEXT_HEADER} or (
                        "token" in normalized_key
                    ):
                        sanitized_headers[str(key)] = "***"
                        continue
                    sanitized_headers[str(key)] = value
                next_tool["headers"] = sanitized_headers
            redacted_tools.append(next_tool)
        payload["tools"] = redacted_tools
        return payload

    @staticmethod
    def _merge_runtime_policies(
        *,
        phase_policy: HandlerRuntimePolicy,
        user_policy: HandlerRuntimePolicy,
    ) -> HandlerRuntimePolicy:
        phase_stage = user_policy.phase_stage or phase_policy.phase_stage
        tool_mode = phase_policy.tool_mode
        allowed_tools = list(phase_policy.allowed_tools or [])

        user_tools = list(user_policy.allowed_tools or [])
        if user_tools:
            if user_policy.tool_mode == "replace":
                tool_mode = "replace"
                allowed_tools = user_tools
            else:
                if tool_mode != "replace":
                    tool_mode = "append"
                allowed_tools.extend(user_tools)
        elif not allowed_tools:
            tool_mode = user_policy.tool_mode

        return HandlerRuntimePolicy(
            phase_stage=phase_stage,
            tool_mode=tool_mode,
            allowed_tools=allowed_tools or None,
        )

    def _validate_execution_policy_request(self, payload: ChatSendRequest) -> None:
        if payload.execution_policy is None:
            return

        normalized = self._normalize_execution_policy_payload(
            payload.execution_policy.model_dump(mode="json", exclude_none=True),
        )
        model = normalized.get("model")
        if isinstance(model, str):
            self._assert_allowed_execution_model(model)

    def _apply_execution_policy(
        self,
        *,
        session: Session,
        phase: str,
        payload: ChatSendRequest,
        prompt: Any,
    ) -> Any:
        metadata = dict(session.metadata_ or {})
        phase_defaults = self._build_phase_execution_policy_defaults(phase=phase)
        global_defaults = self._build_global_execution_policy_defaults()
        session_defaults = self._normalize_execution_policy_payload(
            metadata.get(_EXECUTION_POLICY_META_KEY),
        )
        turn_overrides = (
            self._normalize_execution_policy_payload(
                payload.execution_policy.model_dump(mode="json", exclude_none=True),
            )
            if payload.execution_policy is not None
            else {}
        )

        resolved: dict[str, Any] = {}
        for key in (
            "model",
            "reasoning_effort",
            "max_output_tokens",
            "response_verbosity",
        ):
            candidate = turn_overrides.get(key)
            if candidate is None:
                candidate = session_defaults.get(key)
            if candidate is None:
                candidate = phase_defaults.get(key)
            if candidate is None:
                candidate = global_defaults.get(key)
            resolved[key] = candidate

        resolved_model = str(resolved.get("model") or "").strip()
        if not resolved_model:
            resolved_model = settings.openai_response_model.strip()
        resolved_model = self._coerce_allowed_execution_model(resolved_model)
        resolved["model"] = resolved_model

        resolved_effort = resolved.get("reasoning_effort")
        resolved_reasoning: dict[str, Any] | None = None
        base_reasoning = (
            dict(prompt.reasoning)
            if isinstance(getattr(prompt, "reasoning", None), dict)
            else {}
        )
        if isinstance(resolved_effort, str) and resolved_effort in _EXECUTION_POLICY_ALLOWED_EFFORTS:
            base_reasoning["effort"] = resolved_effort
        if base_reasoning:
            resolved_reasoning = base_reasoning

        resolved_max_output_tokens = self._coerce_positive_int(
            resolved.get("max_output_tokens")
        )
        resolved_verbosity = resolved.get("response_verbosity")
        if (
            not isinstance(resolved_verbosity, str)
            or resolved_verbosity not in _EXECUTION_POLICY_ALLOWED_VERBOSITY
        ):
            resolved_verbosity = None

        prompt.model = resolved_model
        prompt.reasoning = resolved_reasoning
        prompt.max_output_tokens = resolved_max_output_tokens
        prompt.response_verbosity = resolved_verbosity

        if payload.execution_policy is not None:
            persisted_defaults = dict(session_defaults)
            persisted_defaults.update(turn_overrides)
            if "model" not in persisted_defaults:
                persisted_defaults["model"] = resolved_model
            if "reasoning_effort" not in persisted_defaults and isinstance(
                resolved_effort, str
            ):
                persisted_defaults["reasoning_effort"] = resolved_effort
            if (
                "max_output_tokens" not in persisted_defaults
                and resolved_max_output_tokens is not None
            ):
                persisted_defaults["max_output_tokens"] = resolved_max_output_tokens
            if "response_verbosity" not in persisted_defaults and isinstance(
                resolved_verbosity, str
            ):
                persisted_defaults["response_verbosity"] = resolved_verbosity
            metadata[_EXECUTION_POLICY_META_KEY] = persisted_defaults

        metadata[_EXECUTION_POLICY_LAST_RESOLVED_META_KEY] = {
            "model": resolved_model,
            "reasoning_effort": (
                resolved_reasoning.get("effort")
                if isinstance(resolved_reasoning, dict)
                else None
            ),
            "max_output_tokens": resolved_max_output_tokens,
            "response_verbosity": resolved_verbosity,
            "phase": phase,
            "at": datetime.now(UTC).isoformat(),
        }
        session.metadata_ = metadata
        return prompt

    @staticmethod
    def _coerce_positive_int(value: Any) -> int | None:
        try:
            if value is None:
                return None
            normalized = int(value)
        except (TypeError, ValueError):
            return None
        if normalized <= 0:
            return None
        return normalized

    @staticmethod
    def _normalize_execution_policy_payload(raw: Any) -> dict[str, Any]:
        if not isinstance(raw, dict):
            return {}
        output: dict[str, Any] = {}

        raw_model = raw.get("model")
        if isinstance(raw_model, str):
            model = raw_model.strip().lower()
            if model:
                output["model"] = model

        raw_effort = raw.get("reasoning_effort")
        if isinstance(raw_effort, str):
            effort = raw_effort.strip().lower()
            if effort in _EXECUTION_POLICY_ALLOWED_EFFORTS:
                output["reasoning_effort"] = effort

        raw_max_tokens = raw.get("max_output_tokens")
        try:
            max_tokens = int(raw_max_tokens) if raw_max_tokens is not None else None
        except (TypeError, ValueError):
            max_tokens = None
        if isinstance(max_tokens, int) and max_tokens > 0:
            output["max_output_tokens"] = max_tokens

        raw_verbosity = raw.get("response_verbosity")
        if isinstance(raw_verbosity, str):
            verbosity = raw_verbosity.strip().lower()
            if verbosity in _EXECUTION_POLICY_ALLOWED_VERBOSITY:
                output["response_verbosity"] = verbosity

        return output

    @staticmethod
    def _build_phase_execution_policy_defaults(*, phase: str) -> dict[str, Any]:
        defaults: dict[str, dict[str, Any]] = {
            Phase.KYC.value: {"reasoning_effort": "none"},
            Phase.PRE_STRATEGY.value: {"reasoning_effort": "none"},
            Phase.STRATEGY.value: {"reasoning_effort": "low"},
            Phase.STRESS_TEST.value: {"reasoning_effort": "low"},
            Phase.DEPLOYMENT.value: {"reasoning_effort": "none"},
        }
        return dict(defaults.get(phase, {}))

    @staticmethod
    def _build_global_execution_policy_defaults() -> dict[str, Any]:
        model = settings.openai_response_model.strip()
        return {
            "model": model if model else "gpt-5.2",
            "reasoning_effort": "low",
        }

    def _assert_allowed_execution_model(self, model: str) -> str:
        normalized = model.strip().lower()
        allowed = {item.strip().lower() for item in settings.openai_allowed_models if item.strip()}
        if normalized and normalized in allowed:
            return normalized
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "EXECUTION_POLICY_MODEL_NOT_ALLOWED",
                "message": (
                    f"Model '{model}' is not in OPENAI_ALLOWED_MODELS_JSON allowlist."
                ),
                "allowed_models": sorted(allowed),
            },
        )

    def _coerce_allowed_execution_model(self, model: str) -> str:
        normalized = model.strip().lower()
        allowed = {item.strip().lower() for item in settings.openai_allowed_models if item.strip()}
        if normalized and normalized in allowed:
            return normalized
        fallback = settings.openai_response_model.strip().lower()
        if fallback and fallback in allowed:
            return fallback
        if allowed:
            return sorted(allowed)[0]
        return "gpt-5.2"

    def _enforce_prompt_budget(self, prompt: Any) -> Any:
        input_limit = max(int(settings.openai_prompt_input_max_chars), 1)

        enriched_input = str(getattr(prompt, "enriched_input", "") or "")

        prompt.enriched_input = self._trim_enriched_input_by_budget(
            enriched_input=enriched_input,
            limit=input_limit,
        )
        return prompt

    def _trim_enriched_input_by_budget(self, *, enriched_input: str, limit: int) -> str:
        output = enriched_input
        if len(output) <= limit:
            return output

        output = self._remove_tagged_section(output, tag="MCP TOOL SNAPSHOTS")
        if len(output) <= limit:
            return output

        output = self._remove_lines_by_prefixes(
            output,
            prefixes=(
                "- available_markets:",
                "- allowed_instruments_for_target_market:",
                "- deployment_runtime_state:",
            ),
        )
        if len(output) <= limit:
            return output

        return f"{output[:limit].rstrip()}\n\n[TRUNCATED]"

    @staticmethod
    def _remove_tagged_section(text: str, *, tag: str) -> str:
        pattern = re.compile(
            rf"\[\s*{re.escape(tag)}\s*\][\s\S]*?(?=\n\[[A-Z][^\n]*\]|\Z)",
            flags=re.IGNORECASE,
        )
        return pattern.sub("", text).strip()

    @staticmethod
    def _remove_lines_by_prefixes(text: str, *, prefixes: tuple[str, ...]) -> str:
        normalized_prefixes = tuple(prefix.strip().lower() for prefix in prefixes)
        output_lines: list[str] = []
        for line in text.splitlines():
            stripped = line.strip().lower()
            if any(stripped.startswith(prefix) for prefix in normalized_prefixes):
                continue
            output_lines.append(line)
        return "\n".join(output_lines)

    async def _maybe_reset_response_chain(
        self,
        *,
        session: Session,
        phase: str,
        phase_turn_count: int,
        user_message_id: UUID,
        prompt_user_message: str,
    ) -> str:
        threshold = max(int(settings.openai_chain_reset_turns_per_phase), 1)
        if (
            session.previous_response_id is None
            or phase_turn_count <= threshold
            or (phase_turn_count - 1) % threshold != 0
        ):
            return prompt_user_message

        digest = await self._build_phase_chain_digest(
            session_id=session.id,
            phase=phase,
            exclude_user_message_id=user_message_id,
        )
        session.previous_response_id = None

        metadata = dict(session.metadata_ or {})
        metadata["chain_reset"] = {
            "phase": phase,
            "phase_turn_count": phase_turn_count,
            "threshold": threshold,
            "reset_at": datetime.now(UTC).isoformat(),
            "reason": "periodic_phase_chain_reset",
        }
        session.metadata_ = metadata

        if not digest:
            return prompt_user_message
        return f"{digest}{prompt_user_message}"

    async def _build_phase_chain_digest(
        self,
        *,
        session_id: UUID,
        phase: str,
        exclude_user_message_id: UUID,
    ) -> str:
        stmt = (
            select(Message.role, Message.content)
            .where(
                Message.session_id == session_id,
                Message.phase == phase,
                Message.role.in_(("user", "assistant")),
                Message.id != exclude_user_message_id,
            )
            .order_by(Message.created_at.desc(), Message.id.desc())
            .limit(8)
        )
        rows = (await self.db.execute(stmt)).all()
        if not rows:
            return ""

        entries: list[tuple[str, str]] = []
        for role, content in reversed(rows):
            normalized = self._normalize_carryover_utterance(content)
            if not normalized:
                continue
            entries.append((str(role), normalized))

        if not entries:
            return ""

        lines = [
            "[PHASE MEMORY]",
            "- note: compressed recap after periodic context-chain reset",
            f"- phase: {phase}",
        ]
        for role, content in entries[-6:]:
            lines.append(f"- {role}: {content}")
        lines.append("[/PHASE MEMORY]")
        return "\n".join(lines) + "\n\n"

    @staticmethod
    def _build_runtime_policy(payload: ChatSendRequest) -> HandlerRuntimePolicy:
        runtime = payload.runtime_policy
        if runtime is None:
            return HandlerRuntimePolicy()
        return HandlerRuntimePolicy(
            phase_stage=runtime.phase_stage,
            tool_mode=runtime.tool_mode,
            allowed_tools=list(runtime.allowed_tools or []),
        )

    @staticmethod
    def _extract_choice_selection_from_message(text: str) -> dict[str, Any] | None:
        if not isinstance(text, str) or not text.strip():
            return None
        match = re.search(
            r"<\s*CHOICE_SELECTION\s*>([\s\S]*?)<\s*/\s*CHOICE_SELECTION\s*>",
            text,
            flags=re.IGNORECASE,
        )
        if match is None:
            return None
        raw_json = match.group(1)
        if not isinstance(raw_json, str) or not raw_json.strip():
            return None
        try:
            payload = json.loads(raw_json.strip())
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        choice_id = payload.get("choice_id")
        option_id = payload.get("selected_option_id")
        if not isinstance(choice_id, str) or not choice_id.strip():
            return None
        if not isinstance(option_id, str) or not option_id.strip():
            return None
        normalized: dict[str, Any] = {
            "choice_id": choice_id.strip(),
            "selected_option_id": option_id.strip(),
        }
        label = payload.get("selected_option_label")
        if isinstance(label, str) and label.strip():
            normalized["selected_option_label"] = label.strip()
        return normalized
