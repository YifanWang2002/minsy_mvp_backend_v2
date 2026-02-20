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
        pre_strategy_instrument_before: str | None = None
        if phase_before == Phase.PRE_STRATEGY.value:
            pre_data = artifacts.get(Phase.PRE_STRATEGY.value)
            if isinstance(pre_data, dict):
                pre_profile = pre_data.get("profile")
                if isinstance(pre_profile, dict):
                    value = pre_profile.get("target_instrument")
                    if isinstance(value, str) and value.strip():
                        pre_strategy_instrument_before = value.strip()

        runtime_policy = self._resolve_runtime_policy(
            phase=phase_before,
            artifacts=artifacts,
            user_runtime_policy=user_runtime_policy,
        )
        ctx = PhaseContext(
            user_id=user.id,
            session_artifacts=artifacts,
            session_id=session.id,
            language=language,
            runtime_policy=runtime_policy,
        )
        prompt = handler.build_prompt(ctx, prompt_user_message)
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
            phase_before=phase_before,
            phase_turn_count=phase_turn_count,
            prompt_user_message=prompt_user_message,
            handler=handler,
            artifacts=artifacts,
            pre_strategy_instrument_before=pre_strategy_instrument_before,
            ctx=ctx,
            prompt=prompt,
            tools=tools,
        )

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
                )
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
                allowed_tools=list(_MARKET_DATA_MINIMAL_TOOL_NAMES),
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
