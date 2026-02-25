#!/usr/bin/env python3
"""Codex backend full-chain audit (frontend-like, real dependencies).

Goals:
1) strategy -> deployment automatic chain via chat/orchestrator/MCP.
2) post-deployment paper-trading runtime chain (market-data -> order -> portfolio updates).
3) telegram notification chain validation from business triggers (not ad-hoc send).

This script intentionally does NOT rely on project test suites.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

import httpx
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.orchestrator.constants import _TRADING_DEPLOYMENT_TOOL_NAMES
from src.engine.execution.runtime_state_store import runtime_state_store
from src.mcp.trading.tools import TOOL_NAMES as TRADING_MCP_TOOL_NAMES
from src.models import database as db_module


def _now() -> datetime:
    return datetime.now(UTC)


def _safe_json_loads(value: Any) -> Any:
    if isinstance(value, dict | list):
        return value
    if isinstance(value, str):
        text_value = value.strip()
        if not text_value:
            return value
        try:
            return json.loads(text_value)
        except json.JSONDecodeError:
            return value
    return value


def _as_iso(value: Any) -> Any:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC).isoformat()
    return value


def _json_default(value: Any) -> Any:
    converted = _as_iso(value)
    if converted is not value:
        return converted
    if isinstance(value, UUID):
        return str(value)
    return str(value)


def _extract_tool_names_from_skill(skill_path: Path) -> list[str]:
    content = skill_path.read_text(encoding="utf-8")
    return sorted(set(re.findall(r"`(trading_[a-z0-9_]+)`", content)))


def _pick_latest_by_created(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return sorted(
        rows,
        key=lambda item: str(item.get("created_at") or ""),
        reverse=True,
    )[0]


@dataclass(slots=True)
class TurnResult:
    text: str
    done_payload: dict[str, Any]
    phase_after: str
    missing_fields: list[str]
    raw_events: int


class ApiClient:
    def __init__(
        self,
        *,
        base_url: str,
        email: str,
        password: str,
        language: str,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.email = email
        self.password = password
        self.language = language
        self.access_token: str | None = None
        self.user_id: str | None = None
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "ApiClient":
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(connect=20.0, read=360.0, write=20.0, pool=20.0),
            trust_env=False,
        )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._client is not None:
            await self._client.aclose()

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("ApiClient must be used as async context manager.")
        return self._client

    def _headers(self) -> dict[str, str]:
        headers = {"content-type": "application/json"}
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
        return headers

    async def login(self) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, 6):
            try:
                response = await self.client.post(
                    "/auth/login",
                    json={"email": self.email, "password": self.password},
                    headers={"content-type": "application/json"},
                )
                if response.status_code >= 500:
                    response.raise_for_status()
                payload = response.json()
                self.access_token = str(payload["access_token"])
                self.user_id = str(payload["user_id"])
                return payload
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                await asyncio.sleep(min(1.5 * attempt, 6.0))
        raise RuntimeError(f"login failed after retries: {last_error}")

    async def get_me(self) -> dict[str, Any]:
        response = await self.client.get("/auth/me", headers=self._headers())
        response.raise_for_status()
        return response.json()

    async def create_thread(self) -> dict[str, Any]:
        response = await self.client.post(
            "/chat/new-thread",
            json={},
            headers=self._headers(),
        )
        response.raise_for_status()
        return response.json()

    async def get_session(self, session_id: str) -> dict[str, Any]:
        response = await self.client.get(
            f"/sessions/{session_id}",
            headers=self._headers(),
        )
        response.raise_for_status()
        return response.json()

    async def list_deployments(self) -> list[dict[str, Any]]:
        response = await self.client.get("/deployments", headers=self._headers())
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, list) else []

    async def get_deployment(self, deployment_id: str) -> dict[str, Any]:
        response = await self.client.get(
            f"/deployments/{deployment_id}",
            headers=self._headers(),
        )
        response.raise_for_status()
        return response.json()

    async def get_orders(self, deployment_id: str) -> list[dict[str, Any]]:
        response = await self.client.get(
            f"/deployments/{deployment_id}/orders",
            headers=self._headers(),
        )
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, list) else []

    async def get_positions(self, deployment_id: str) -> list[dict[str, Any]]:
        response = await self.client.get(
            f"/deployments/{deployment_id}/positions",
            headers=self._headers(),
        )
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, list) else []

    async def get_pnl(self, deployment_id: str) -> list[dict[str, Any]]:
        response = await self.client.get(
            f"/deployments/{deployment_id}/pnl",
            headers=self._headers(),
        )
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, list) else []

    async def get_signals(self, deployment_id: str) -> list[dict[str, Any]]:
        response = await self.client.get(
            f"/deployments/{deployment_id}/signals",
            headers=self._headers(),
        )
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, list) else []

    async def get_portfolio(self, deployment_id: str) -> dict[str, Any]:
        response = await self.client.get(
            f"/deployments/{deployment_id}/portfolio",
            headers=self._headers(),
        )
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, dict) else {}

    async def get_fills(self, deployment_id: str) -> list[dict[str, Any]]:
        response = await self.client.get(
            f"/deployments/{deployment_id}/fills",
            headers=self._headers(),
        )
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, list) else []

    async def process_now(self, deployment_id: str) -> dict[str, Any]:
        response = await self.client.post(
            f"/deployments/{deployment_id}/process-now",
            headers=self._headers(),
        )
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, dict) else {}

    async def stream_chat_turn(self, *, session_id: str, message: str) -> TurnResult:
        done_payload: dict[str, Any] = {}
        text_chunks: list[str] = []
        current_event = "message"
        data_lines: list[str] = []
        event_count = 0
        done_seen = False

        def _flush_event() -> tuple[str, dict[str, Any] | None] | None:
            nonlocal current_event, data_lines
            if not data_lines:
                current_event = "message"
                return None
            raw = "\n".join(data_lines)
            data_lines = []
            parsed: dict[str, Any] | None
            try:
                payload = json.loads(raw)
                parsed = payload if isinstance(payload, dict) else None
            except json.JSONDecodeError:
                parsed = None
            evt = current_event
            current_event = "message"
            return evt, parsed

        async with self.client.stream(
            "POST",
            "/chat/send-openai-stream",
            params={"language": self.language},
            headers=self._headers(),
            json={"session_id": session_id, "message": message},
        ) as response:
            response.raise_for_status()
            async for raw_line in response.aiter_lines():
                if raw_line is None:
                    continue
                line = str(raw_line).rstrip("\r")
                if line == "":
                    flushed = _flush_event()
                    if flushed is None:
                        continue
                    event_count += 1
                    evt, payload = flushed
                    if not isinstance(payload, dict):
                        continue
                    if evt == "stream" and payload.get("type") == "text_delta":
                        delta = payload.get("delta")
                        if isinstance(delta, str):
                            text_chunks.append(delta)
                    if evt == "stream" and payload.get("type") == "done":
                        done_payload = dict(payload)
                        done_seen = True
                        break
                    continue

                if line.startswith(":"):
                    continue
                if line.startswith("event:"):
                    current_event = line.split(":", 1)[1].strip() or "message"
                    continue
                if line.startswith("data:"):
                    data_lines.append(line.split(":", 1)[1].lstrip())
            if done_seen:
                # Do not wait for server-side socket close once terminal payload is received.
                return TurnResult(
                    text="".join(text_chunks),
                    done_payload=done_payload,
                    phase_after=str(done_payload.get("phase") or "").strip().lower(),
                    missing_fields=(
                        [str(item) for item in done_payload.get("missing_fields")]
                        if isinstance(done_payload.get("missing_fields"), list)
                        else []
                    ),
                    raw_events=event_count,
                )

        tail = _flush_event()
        if tail is not None:
            event_count += 1
            evt, payload = tail
            if evt == "stream" and isinstance(payload, dict) and payload.get("type") == "done":
                done_payload = dict(payload)

        if not done_payload:
            raise RuntimeError("SSE stream ended without `done` payload.")

        phase_after = str(done_payload.get("phase") or "").strip().lower()
        if not phase_after:
            raise RuntimeError("`done` payload missing phase.")
        missing = done_payload.get("missing_fields")
        missing_fields = [str(item) for item in missing] if isinstance(missing, list) else []
        return TurnResult(
            text="".join(text_chunks),
            done_payload=done_payload,
            phase_after=phase_after,
            missing_fields=missing_fields,
            raw_events=event_count,
        )


class DatabaseProbe:
    async def __aenter__(self) -> "DatabaseProbe":
        await db_module.init_postgres(ensure_schema=False)
        if db_module.AsyncSessionLocal is None:
            raise RuntimeError("AsyncSessionLocal is not initialized.")
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await db_module.close_postgres()

    async def fetch_all(self, query: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        assert db_module.AsyncSessionLocal is not None
        async with db_module.AsyncSessionLocal() as session:
            result = await session.execute(text(query), params or {})
            rows = []
            for row in result:
                mapped = dict(row._mapping)
                rows.append({key: _as_iso(value) for key, value in mapped.items()})
            return rows

    async def fetch_one(self, query: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
        rows = await self.fetch_all(query, params=params)
        if not rows:
            return None
        return rows[0]


def _extract_mcp_calls(session_detail: dict[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    messages = session_detail.get("messages")
    if not isinstance(messages, list):
        return output
    for message in messages:
        if not isinstance(message, dict):
            continue
        if str(message.get("role")) != "assistant":
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for tool in tool_calls:
            if not isinstance(tool, dict):
                continue
            if str(tool.get("type")) != "mcp_call":
                continue
            output.append(
                {
                    "message_id": str(message.get("id")),
                    "phase": str(message.get("phase")),
                    "created_at": message.get("created_at"),
                    "name": str(tool.get("name") or ""),
                    "status": str(tool.get("status") or ""),
                    "arguments_raw": tool.get("arguments"),
                    "arguments": _safe_json_loads(tool.get("arguments")),
                    "output_raw": tool.get("output"),
                    "output": _safe_json_loads(tool.get("output")),
                    "error": tool.get("error"),
                }
            )
    return output


async def run_step_1_strategy_to_deployment(
    *,
    api: ApiClient,
    db: DatabaseProbe,
    audit_start: datetime,
) -> dict[str, Any]:
    step: dict[str, Any] = {
        "ok": False,
        "session_id": None,
        "phase_start": None,
        "phase_end": None,
        "strategy_id": None,
        "deployment_id": None,
        "deployment_status": None,
        "transition": {},
        "prompt_handler_sync": {},
        "turns": [],
        "mcp_calls": [],
        "findings": [],
    }

    pre_deployments = await api.list_deployments()
    pre_deployment_ids = {str(item.get("deployment_id")) for item in pre_deployments if item.get("deployment_id")}

    thread = await api.create_thread()
    session_id = str(thread.get("session_id"))
    phase = str(thread.get("phase", "")).strip().lower()
    step["session_id"] = session_id
    step["phase_start"] = phase

    async def run_turn(label: str, message: str) -> tuple[TurnResult, dict[str, Any]]:
        nonlocal phase
        print(f"[step1] turn_start label={label} phase_before={phase}", flush=True)
        before_session = await api.get_session(session_id)
        before_msg_count = (
            len(before_session.get("messages"))
            if isinstance(before_session.get("messages"), list)
            else 0
        )
        try:
            turn = await asyncio.wait_for(
                api.stream_chat_turn(session_id=session_id, message=message),
                timeout=120.0,
            )
            session_detail = await api.get_session(session_id)
        except TimeoutError:
            session_detail = await api.get_session(session_id)
            messages = (
                session_detail.get("messages")
                if isinstance(session_detail.get("messages"), list)
                else []
            )
            after_msg_count = len(messages)
            if after_msg_count <= before_msg_count:
                raise
            assistant_text = ""
            for msg in reversed(messages):
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    assistant_text = str(msg.get("content") or "")
                    break
            synthetic_phase = str(session_detail.get("current_phase") or phase).strip().lower()
            turn = TurnResult(
                text=assistant_text,
                done_payload={
                    "type": "done_timeout_fallback",
                    "phase": synthetic_phase,
                    "missing_fields": [],
                },
                phase_after=synthetic_phase,
                missing_fields=[],
                raw_events=0,
            )
        phase = str(session_detail.get("current_phase") or turn.phase_after).strip().lower()
        print(
            f"[step1] turn_done label={label} phase_after={phase} missing={turn.missing_fields}",
            flush=True,
        )
        step["turns"].append(
            {
                "label": label,
                "input": message,
                "phase_after": phase,
                "missing_fields": turn.missing_fields,
                "done": turn.done_payload,
                "text_preview": turn.text[:500],
                "raw_event_count": turn.raw_events,
            }
        )
        return turn, session_detail

    if phase == "pre_strategy":
        await run_turn(
            "pre_strategy_scope",
            (
                "我们直接进入策略开发。请一次性写入以下 pre_strategy 字段并自动推进到 strategy：\n"
                "target_market=crypto\n"
                "target_instrument=BTCUSD\n"
                "opportunity_frequency_bucket=multiple_per_day\n"
                "holding_period_bucket=intraday_scalp\n"
                "不要让我再点按钮或手动提交。"
            ),
        )

    if phase == "pre_strategy":
        await run_turn(
            "pre_strategy_retry",
            "如果还有缺字段，请本回合全部补齐并立即推进到 strategy 阶段。",
        )

    if phase == "strategy":
        await run_turn(
            "strategy_generate_upsert",
            (
                "在 strategy 阶段执行：\n"
                "1) 生成并保存一个可部署的 crypto BTCUSD 1m DSL；\n"
                "2) 入场条件可简化为恒真（1>0），出场条件恒假（0>1），用于打通交易链路；\n"
                "3) 调用 strategy_validate_dsl 和 strategy_upsert_dsl；\n"
                "4) 本回合先不要确认部署，只把 strategy_id 写入 AGENT_STATE_PATCH。"
            ),
        )

    if phase == "strategy":
        await run_turn(
            "strategy_backtest_then_confirm",
            (
                "请基于当前 strategy_id 触发一次回测（backtest_create_job + backtest_get_job），"
                "然后在同一轮把 strategy_confirmed=true 写入 AGENT_STATE_PATCH，自动推进到 deployment。"
            ),
        )

    if phase == "strategy":
        await run_turn(
            "strategy_force_confirm",
            (
                "不要阻塞等待。现在直接输出包含 strategy_id + strategy_confirmed=true 的 AGENT_STATE_PATCH，"
                "并推进到 deployment。"
            ),
        )

    if phase != "deployment":
        step["phase_end"] = phase
        step["findings"].append(
            f"Session failed to reach deployment phase automatically. phase_end={phase}"
        )
        return step

    await run_turn(
        "deployment_create_start",
        (
            "现在在 deployment 阶段直接完成部署：\n"
            "- 调用 trading_create_paper_deployment，auto_start=true；\n"
            '- risk_limits 至少包含 {"order_qty":0.001,"max_position_notional":200}；\n'
            "- 不要向我要 strategy_id 或 broker_account_id，优先从上下文自动解析；\n"
            "- 然后调用 trading_list_deployments 验证 active，并输出 deployment_status=deployed。"
        ),
    )

    session_detail = await api.get_session(session_id)
    mcp_calls = _extract_mcp_calls(session_detail)
    step["mcp_calls"] = mcp_calls
    step["phase_end"] = str(session_detail.get("current_phase") or phase)

    artifacts = session_detail.get("artifacts") if isinstance(session_detail.get("artifacts"), dict) else {}
    strategy_profile = (
        artifacts.get("strategy", {}).get("profile")
        if isinstance(artifacts.get("strategy"), dict)
        else {}
    )
    deployment_profile = (
        artifacts.get("deployment", {}).get("profile")
        if isinstance(artifacts.get("deployment"), dict)
        else {}
    )
    strategy_id = str(strategy_profile.get("strategy_id") or "").strip() or None
    deployment_id_from_artifacts = str(deployment_profile.get("deployment_id") or "").strip() or None
    deployment_status_profile = str(deployment_profile.get("deployment_status") or "").strip() or None
    step["strategy_id"] = strategy_id

    post_deployments = await api.list_deployments()
    created_candidates = [
        item
        for item in post_deployments
        if str(item.get("deployment_id")) not in pre_deployment_ids
    ]
    deployment_pick = _pick_latest_by_created(created_candidates)
    if deployment_pick is None and deployment_id_from_artifacts:
        deployment_pick = next(
            (
                item
                for item in post_deployments
                if str(item.get("deployment_id")) == deployment_id_from_artifacts
            ),
            None,
        )
    if deployment_pick is None and strategy_id:
        deployment_pick = _pick_latest_by_created(
            [
                item
                for item in post_deployments
                if str(item.get("strategy_id")) == strategy_id
            ]
        )

    deployment_id = (
        str(deployment_pick.get("deployment_id"))
        if isinstance(deployment_pick, dict) and deployment_pick.get("deployment_id")
        else deployment_id_from_artifacts
    )
    step["deployment_id"] = deployment_id

    deployment_status = None
    if deployment_id:
        deployment_payload = await api.get_deployment(deployment_id)
        deployment_status = str(deployment_payload.get("status") or "")
    step["deployment_status"] = deployment_status or deployment_status_profile

    transition = await db.fetch_one(
        """
        select
            id::text as phase_transition_id,
            from_phase,
            to_phase,
            trigger,
            metadata,
            created_at
        from phase_transitions
        where session_id = cast(:session_id as uuid)
          and to_phase = 'deployment'
        order by created_at desc
        limit 1
        """,
        {"session_id": session_id},
    )
    step["transition"] = transition or {}

    skill_tools = _extract_tool_names_from_skill(ROOT / "src/agents/skills/deployment/skills.md")
    policy_tools = sorted(set(_TRADING_DEPLOYMENT_TOOL_NAMES))
    registered_tools = sorted(
        {
            name
            for name in TRADING_MCP_TOOL_NAMES
            if name.startswith("trading_")
            and name not in {"trading_ping", "trading_capabilities"}
        }
    )
    step["prompt_handler_sync"] = {
        "deployment_skill_tools": skill_tools,
        "orchestrator_policy_tools": policy_tools,
        "trading_registered_tools": registered_tools,
        "skill_minus_policy": sorted(set(skill_tools) - set(policy_tools)),
        "policy_minus_skill": sorted(set(policy_tools) - set(skill_tools)),
        "policy_minus_registered": sorted(set(policy_tools) - set(registered_tools)),
    }

    used_deployment_tools = {
        call["name"]
        for call in mcp_calls
        if str(call.get("phase")) == "deployment"
    }
    step["used_deployment_tools"] = sorted(used_deployment_tools)

    if not transition:
        step["findings"].append("No recorded phase_transition row to deployment for this session.")
    elif str(transition.get("trigger")) != "ai_output":
        step["findings"].append(
            f"Phase transition to deployment is not ai_output (trigger={transition.get('trigger')})."
        )

    if not deployment_id:
        step["findings"].append("No deployment_id resolved from artifacts or deployments list.")
    if deployment_status != "active":
        step["findings"].append(f"Deployment is not active after deployment turn (status={deployment_status}).")

    required_tools = {"trading_create_paper_deployment", "trading_list_deployments"}
    if not required_tools.issubset(used_deployment_tools):
        step["findings"].append(
            f"Deployment-phase tool usage missing expected tools: {sorted(required_tools - used_deployment_tools)}"
        )

    if step["prompt_handler_sync"]["skill_minus_policy"]:
        step["findings"].append("Deployment skill tool list is out-of-sync with orchestrator policy.")
    if step["prompt_handler_sync"]["policy_minus_registered"]:
        step["findings"].append("Orchestrator deployment policy references unregistered trading tools.")

    step["ok"] = not step["findings"]
    return step


async def run_step_2_paper_trading_runtime(
    *,
    api: ApiClient,
    deployment_id: str | None,
) -> dict[str, Any]:
    step: dict[str, Any] = {
        "ok": False,
        "deployment_id": deployment_id,
        "loops": [],
        "scheduler_evidence": {},
        "market_data_evidence": {},
        "orders_evidence": {},
        "portfolio_evidence": {},
        "findings": [],
    }
    if not deployment_id:
        step["findings"].append("Step 2 skipped: missing deployment_id from Step 1.")
        return step

    baseline_orders = await api.get_orders(deployment_id)
    baseline_signals = await api.get_signals(deployment_id)
    baseline_order_count = len(baseline_orders)
    baseline_signal_count = len(baseline_signals)
    celery_autonomous_activity = False

    # First observe background runtime without manual trigger.
    for passive_index in range(1, 13):
        print(f"[step2] passive_observe_loop={passive_index}", flush=True)
        await asyncio.sleep(2.0)
        deployment = await api.get_deployment(deployment_id)
        orders = await api.get_orders(deployment_id)
        signals = await api.get_signals(deployment_id)
        run_payload = deployment.get("run") if isinstance(deployment.get("run"), dict) else {}
        runtime_state = (
            run_payload.get("runtime_state")
            if isinstance(run_payload.get("runtime_state"), dict)
            else {}
        )
        scheduler_state = (
            runtime_state.get("scheduler")
            if isinstance(runtime_state.get("scheduler"), dict)
            else {}
        )
        observed = (
            len(orders) > baseline_order_count
            or len(signals) > baseline_signal_count
            or bool(scheduler_state.get("last_enqueued_at"))
            or bool(scheduler_state.get("fallback_last_enqueued_at"))
        )
        step["loops"].append(
            {
                "mode": "passive",
                "loop": passive_index,
                "deployment_status": deployment.get("status"),
                "orders_total": len(orders),
                "signals_total": len(signals),
                "run": run_payload,
            }
        )
        if observed:
            celery_autonomous_activity = True
            print("[step2] detected autonomous runtime activity", flush=True)
            break

    best_snapshot: dict[str, Any] = {}
    for index in range(1, 25):
        print(f"[step2] runtime_loop={index}", flush=True)
        process_now_payload: dict[str, Any] = {}
        try:
            process_now_payload = await api.process_now(deployment_id)
        except Exception as exc:  # noqa: BLE001
            process_now_payload = {"error": f"{type(exc).__name__}: {exc}"}

        await asyncio.sleep(2.0)
        deployment = await api.get_deployment(deployment_id)
        orders = await api.get_orders(deployment_id)
        positions = await api.get_positions(deployment_id)
        pnl_rows = await api.get_pnl(deployment_id)
        signals = await api.get_signals(deployment_id)
        fills = await api.get_fills(deployment_id)
        portfolio = await api.get_portfolio(deployment_id)
        redis_runtime_state = await runtime_state_store.get(UUID(deployment_id))

        real_orders = [
            order
            for order in orders
            if isinstance(order.get("provider_order_id"), str)
            and not str(order.get("provider_order_id")).startswith("paper-")
        ]
        snapshot = {
            "loop": index,
            "process_now": process_now_payload,
            "deployment_status": deployment.get("status"),
            "run": deployment.get("run"),
            "orders_total": len(orders),
            "real_orders_total": len(real_orders),
            "fills_total": len(fills),
            "positions_total": len(positions),
            "signals_total": len(signals),
            "pnl_snapshots_total": len(pnl_rows),
            "portfolio": portfolio,
            "redis_runtime_state": redis_runtime_state,
        }
        step["loops"].append(snapshot)
        best_snapshot = snapshot
        print(
            (
                "[step2] snapshot "
                f"signals={snapshot['signals_total']} orders={snapshot['orders_total']} "
                f"real_orders={snapshot['real_orders_total']} pnl={snapshot['pnl_snapshots_total']}"
            ),
            flush=True,
        )

        has_signal_activity = len(signals) > 0
        has_real_order = len(real_orders) > 0
        has_pnl = len(pnl_rows) > 0
        if has_signal_activity and has_real_order and has_pnl:
            break

    if not best_snapshot:
        step["findings"].append("No runtime loop snapshot collected.")
        return step

    run_payload = best_snapshot.get("run") if isinstance(best_snapshot.get("run"), dict) else {}
    runtime_state = run_payload.get("runtime_state") if isinstance(run_payload.get("runtime_state"), dict) else {}
    scheduler_state = runtime_state.get("scheduler") if isinstance(runtime_state.get("scheduler"), dict) else {}
    execution_state = runtime_state.get("execution") if isinstance(runtime_state.get("execution"), dict) else {}

    step["scheduler_evidence"] = {
        "celery_autonomous_activity": celery_autonomous_activity,
        "timeframe_seconds": run_payload.get("timeframe_seconds"),
        "last_trigger_bucket": run_payload.get("last_trigger_bucket"),
        "last_enqueued_at": run_payload.get("last_enqueued_at"),
        "scheduler_state": scheduler_state,
        "execution_state": execution_state,
        "redis_runtime_state": best_snapshot.get("redis_runtime_state"),
    }
    step["market_data_evidence"] = {
        "latest_signal": best_snapshot.get("process_now"),
        "signals_total": best_snapshot.get("signals_total"),
    }
    step["orders_evidence"] = {
        "orders_total": best_snapshot.get("orders_total"),
        "real_orders_total": best_snapshot.get("real_orders_total"),
        "fills_total": best_snapshot.get("fills_total"),
    }
    step["portfolio_evidence"] = {
        "pnl_snapshots_total": best_snapshot.get("pnl_snapshots_total"),
        "portfolio": best_snapshot.get("portfolio"),
        "positions_total": best_snapshot.get("positions_total"),
    }

    if best_snapshot.get("deployment_status") != "active":
        step["findings"].append(
            f"Deployment status is not active during runtime loop (status={best_snapshot.get('deployment_status')})."
        )
    if int(best_snapshot.get("signals_total") or 0) <= 0:
        step["findings"].append("No signal events were persisted for deployment runtime.")
    if int(best_snapshot.get("real_orders_total") or 0) <= 0:
        step["findings"].append(
            "No real Alpaca-backed order found (provider_order_id missing or local `paper-*` only)."
        )
    if int(best_snapshot.get("pnl_snapshots_total") or 0) <= 0:
        step["findings"].append("No PnL snapshots generated during runtime processing.")
    if not celery_autonomous_activity:
        step["findings"].append(
            "No autonomous Celery runtime tick observed within passive window; "
            "signal/order evidence came from manual process-now trigger."
        )
    redis_runtime_state = best_snapshot.get("redis_runtime_state")
    if not isinstance(redis_runtime_state, dict):
        step["findings"].append("No runtime_state found in Redis-backed RuntimeStateStore.")

    step["ok"] = not step["findings"]
    return step


async def run_step_3_notifications(
    *,
    db: DatabaseProbe,
    user_id: str | None,
    deployment_id: str | None,
    strategy_id: str | None,
    audit_start: datetime,
) -> dict[str, Any]:
    step: dict[str, Any] = {
        "ok": False,
        "outbox_rows": [],
        "delivery_attempts": [],
        "event_summary": {},
        "findings": [],
    }
    if not user_id:
        step["findings"].append("Missing user_id.")
        return step

    poll_until = _now() + timedelta(seconds=75)
    rows: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    while _now() < poll_until:
        print("[step3] polling notification outbox...", flush=True)
        rows = await db.fetch_all(
            """
            select
                o.id::text as outbox_id,
                o.user_id::text as user_id,
                o.channel,
                o.event_type,
                o.event_key,
                o.payload,
                o.status,
                o.retry_count,
                o.max_retries,
                o.scheduled_at,
                o.sent_at,
                o.next_retry_at,
                o.last_error,
                o.created_at
            from notification_outbox o
            where o.user_id = cast(:user_id as uuid)
              and o.created_at >= cast(:start_ts as timestamptz)
            order by o.created_at desc
            limit 300
            """,
            {
                "user_id": user_id,
                "start_ts": audit_start,
            },
        )
        if rows:
            attempts = await db.fetch_all(
                """
                select
                    a.id::text as attempt_id,
                    a.outbox_id::text as outbox_id,
                    a.provider,
                    a.provider_message_id,
                    a.success,
                    a.error_code,
                    a.error_message,
                    a.request_payload,
                    a.response_payload,
                    a.attempted_at,
                    a.created_at
                from notification_delivery_attempts a
                join notification_outbox o
                  on o.id = a.outbox_id
                where o.user_id = cast(:user_id as uuid)
                  and o.created_at >= cast(:start_ts as timestamptz)
                order by a.attempted_at desc
                limit 500
                """,
                {
                    "user_id": user_id,
                    "start_ts": audit_start,
                },
            )

        deployment_started_ok = any(
            row.get("event_type") == "DEPLOYMENT_STARTED"
            and (
                not deployment_id
                or str((row.get("payload") or {}).get("deployment_id")) == deployment_id
            )
            for row in rows
        )
        if deployment_started_ok:
            print("[step3] deployment_started event observed", flush=True)
            break
        await asyncio.sleep(3.0)

    step["outbox_rows"] = rows
    step["delivery_attempts"] = attempts

    def _match_rows(event_type: str) -> list[dict[str, Any]]:
        matched = [row for row in rows if row.get("event_type") == event_type]
        if deployment_id:
            matched = [
                row
                for row in matched
                if str((row.get("payload") or {}).get("deployment_id") or "") in {"", deployment_id}
            ]
        if strategy_id and event_type == "BACKTEST_COMPLETED":
            matched = [
                row
                for row in matched
                if str((row.get("payload") or {}).get("strategy_id") or "") in {"", strategy_id}
            ]
        return matched

    event_types = [
        "BACKTEST_COMPLETED",
        "DEPLOYMENT_STARTED",
        "POSITION_OPENED",
        "POSITION_CLOSED",
        "RISK_TRIGGERED",
        "EXECUTION_ANOMALY",
        "TRADE_APPROVAL_REQUESTED",
        "TRADE_APPROVAL_APPROVED",
        "TRADE_APPROVAL_REJECTED",
        "TRADE_APPROVAL_EXPIRED",
    ]
    summary: dict[str, Any] = {}
    for event_type in event_types:
        event_rows = _match_rows(event_type)
        outbox_ids = {row["outbox_id"] for row in event_rows}
        event_attempts = [item for item in attempts if item.get("outbox_id") in outbox_ids]
        summary[event_type] = {
            "count": len(event_rows),
            "sent": sum(1 for row in event_rows if row.get("status") == "sent"),
            "failed_or_dead": sum(
                1 for row in event_rows if row.get("status") in {"failed", "dead"}
            ),
            "telegram_success_attempts": sum(
                1
                for item in event_attempts
                if item.get("provider") == "telegram" and bool(item.get("success"))
            ),
        }
    step["event_summary"] = summary

    deployment_summary = summary.get("DEPLOYMENT_STARTED", {})
    if int(deployment_summary.get("count") or 0) <= 0:
        step["findings"].append("No DEPLOYMENT_STARTED notification row generated in outbox.")
    elif int(deployment_summary.get("telegram_success_attempts") or 0) <= 0:
        step["findings"].append("DEPLOYMENT_STARTED exists but no successful Telegram delivery attempt.")

    for event_type in ("POSITION_OPENED", "POSITION_CLOSED"):
        if int(summary.get(event_type, {}).get("count") or 0) > 0 and int(
            summary.get(event_type, {}).get("telegram_success_attempts") or 0
        ) <= 0:
            step["findings"].append(f"{event_type} generated but Telegram delivery did not succeed.")

    if int(summary.get("BACKTEST_COMPLETED", {}).get("count") or 0) <= 0:
        step["findings"].append(
            "No BACKTEST_COMPLETED notification observed during this audit window."
        )

    if int(summary.get("RISK_TRIGGERED", {}).get("count") or 0) == 0:
        step["findings"].append(
            "No RISK_TRIGGERED notification observed; runtime currently does not enqueue this event path."
        )
    if int(summary.get("EXECUTION_ANOMALY", {}).get("count") or 0) == 0:
        step["findings"].append(
            "No EXECUTION_ANOMALY notification observed; runtime currently does not enqueue this event path."
        )

    step["ok"] = not step["findings"]
    return step


async def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    report: dict[str, Any] = {
        "started_at": _now().isoformat(),
        "base_url": args.base_url,
        "user_email": args.email,
        "language": args.language,
        "step_1": {},
        "step_2": {},
        "step_3": {},
    }
    audit_start = _now()

    async with ApiClient(
        base_url=args.base_url,
        email=args.email,
        password=args.password,
        language=args.language,
    ) as api, DatabaseProbe() as db:
        print("[audit] login...", flush=True)
        auth_payload = await api.login()
        print("[audit] login_ok", flush=True)
        me_payload = await api.get_me()
        report["auth"] = {
            "user_id": auth_payload.get("user_id"),
            "kyc_status": me_payload.get("kyc_status"),
            "login_ok": True,
        }

        print("[audit] step1 begin", flush=True)
        step_1 = await run_step_1_strategy_to_deployment(
            api=api,
            db=db,
            audit_start=audit_start,
        )
        print("[audit] step1 done", flush=True)
        report["step_1"] = step_1

        print("[audit] step2 begin", flush=True)
        step_2 = await run_step_2_paper_trading_runtime(
            api=api,
            deployment_id=step_1.get("deployment_id"),
        )
        print("[audit] step2 done", flush=True)
        report["step_2"] = step_2

        print("[audit] step3 begin", flush=True)
        step_3 = await run_step_3_notifications(
            db=db,
            user_id=api.user_id,
            deployment_id=step_1.get("deployment_id"),
            strategy_id=step_1.get("strategy_id"),
            audit_start=audit_start,
        )
        print("[audit] step3 done", flush=True)
        report["step_3"] = step_3

    report["finished_at"] = _now().isoformat()
    report["overall_ok"] = bool(
        report["step_1"].get("ok")
        and report["step_2"].get("ok")
        and report["step_3"].get("ok")
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Codex backend full-chain audit.")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000/api/v1",
        help="Backend API base URL.",
    )
    parser.add_argument(
        "--email",
        default="2@test.com",
        help="Audit account email.",
    )
    parser.add_argument(
        "--password",
        default="123456",
        help="Audit account password.",
    )
    parser.add_argument(
        "--language",
        default="zh",
        help="Chat language query parameter.",
    )
    parser.add_argument(
        "--report",
        default=str(ROOT / "artifacts" / "codex_backend_full_chain_audit_20260225.json"),
        help="Path to write JSON report.",
    )
    return parser.parse_args()


def print_brief(report: dict[str, Any]) -> None:
    step_1 = report.get("step_1", {})
    step_2 = report.get("step_2", {})
    step_3 = report.get("step_3", {})
    print("== Codex Audit Brief ==")
    print(f"overall_ok: {report.get('overall_ok')}")
    print(
        "step1:",
        {
            "ok": step_1.get("ok"),
            "session_id": step_1.get("session_id"),
            "phase_start": step_1.get("phase_start"),
            "phase_end": step_1.get("phase_end"),
            "strategy_id": step_1.get("strategy_id"),
            "deployment_id": step_1.get("deployment_id"),
            "deployment_status": step_1.get("deployment_status"),
            "findings": step_1.get("findings"),
        },
    )
    print(
        "step2:",
        {
            "ok": step_2.get("ok"),
            "deployment_id": step_2.get("deployment_id"),
            "orders": step_2.get("orders_evidence"),
            "findings": step_2.get("findings"),
        },
    )
    print(
        "step3:",
        {
            "ok": step_3.get("ok"),
            "event_summary": step_3.get("event_summary"),
            "findings": step_3.get("findings"),
        },
    )


def main() -> int:
    args = parse_args()
    report_path = Path(args.report).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = asyncio.run(run_audit(args))
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    print_brief(report)
    print(f"report_path: {report_path}")
    return 0 if report.get("overall_ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
