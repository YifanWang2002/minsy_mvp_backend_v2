#!/usr/bin/env python3
"""Frontend-like end-to-end audit for strategy->deployment->paper trading->notification chain.

This script intentionally does NOT rely on project tests.
It exercises real API endpoints and real OpenAI Responses streaming behavior.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
import inspect
import json
from pathlib import Path
from typing import Any
from uuid import UUID

import httpx
from sqlalchemy import select

from src.agents.orchestrator.constants import _TRADING_DEPLOYMENT_TOOL_NAMES
from src.mcp.trading.tools import TOOL_NAMES as TRADING_MCP_TOOL_NAMES
from src.mcp.trading.tools import trading_create_paper_deployment
from src.models import database as db_module
from src.models.backtest import BacktestJob
from src.models.notification_delivery_attempt import NotificationDeliveryAttempt
from src.models.notification_outbox import NotificationOutbox
from src.models.phase_transition import PhaseTransition


@dataclass(slots=True)
class TurnAudit:
    turn_index: int
    user_message: str
    done_payload: dict[str, Any] | None
    done_phase: str | None
    text_preview: str
    mcp_calls: list[dict[str, Any]] = field(default_factory=list)
    mcp_call_names: list[str] = field(default_factory=list)
    sse_event_counter: dict[str, int] = field(default_factory=dict)
    stream_error: str | None = None


@dataclass(slots=True)
class Step1Result:
    ok: bool
    session_id: str
    session_final_phase: str
    strategy_id: str | None
    deployment_id: str | None
    auto_transition_strategy_to_deployment: bool
    transition_records: list[dict[str, Any]]
    trading_mcp_tools_used: list[str]
    prompt_tool_sync: dict[str, Any]
    issues: list[str]


@dataclass(slots=True)
class Step2Result:
    ok: bool
    deployment_id: str | None
    deployment_status: str | None
    runtime_scheduler_state: dict[str, Any]
    signals_count: int
    orders_count: int
    positions_count: int
    pnl_snapshots_count: int
    alpaca_real_order_detected: bool
    runtime_poll_samples: list[dict[str, Any]]
    process_now_calls: list[dict[str, Any]]
    issues: list[str]


@dataclass(slots=True)
class Step3Result:
    ok: bool
    outbox_summary: dict[str, Any]
    attempt_summary: dict[str, Any]
    backtest_jobs: list[dict[str, Any]]
    key_events_observed: list[str]
    issues: list[str]


@dataclass(slots=True)
class AuditReport:
    started_at: str
    finished_at: str
    duration_seconds: float
    base_url: str
    user_email: str
    user_id: str | None
    session_id: str | None
    strategy_id: str | None
    deployment_id: str | None
    turns: list[TurnAudit]
    step1: Step1Result
    step2: Step2Result
    step3: Step3Result


class FrontendLikeClient:
    def __init__(self, base_url: str, *, timeout_seconds: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds))
        self._auth_header: dict[str, str] = {}

    async def aclose(self) -> None:
        await self._client.aclose()

    async def login(self, *, email: str, password: str) -> dict[str, Any]:
        payload = {"email": email, "password": password}
        response = await self._client.post(
            f"{self.base_url}/auth/login",
            json=payload,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        data = response.json()
        access_token = str(data.get("access_token", "")).strip()
        if not access_token:
            raise RuntimeError("Login succeeded but access_token is empty.")
        self._auth_header = {"Authorization": f"Bearer {access_token}"}
        return data

    async def get_json(self, path: str, **kwargs: Any) -> Any:
        response = await self._client.get(
            f"{self.base_url}{path}",
            headers=self._auth_header,
            **kwargs,
        )
        response.raise_for_status()
        return response.json()

    async def post_json(self, path: str, *, json_body: dict[str, Any], **kwargs: Any) -> Any:
        response = await self._client.post(
            f"{self.base_url}{path}",
            headers={**self._auth_header, "Content-Type": "application/json"},
            json=json_body,
            **kwargs,
        )
        response.raise_for_status()
        return response.json() if response.content else {}

    async def post_sse(
        self,
        path: str,
        *,
        json_body: dict[str, Any],
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        async with self._client.stream(
            "POST",
            f"{self.base_url}{path}",
            headers={**self._auth_header, "Content-Type": "application/json"},
            params=params,
            json=json_body,
        ) as response:
            response.raise_for_status()
            event_name: str | None = None
            data_lines: list[str] = []

            async for raw_line in response.aiter_lines():
                line = raw_line.rstrip("\n")
                if not line:
                    if not data_lines:
                        event_name = None
                        continue
                    payload_raw = "\n".join(data_lines).strip()
                    payload: dict[str, Any]
                    try:
                        parsed = json.loads(payload_raw)
                        payload = parsed if isinstance(parsed, dict) else {"_raw": payload_raw}
                    except json.JSONDecodeError:
                        payload = {"_raw": payload_raw}
                    events.append(
                        {
                            "event": event_name or "message",
                            "payload": payload,
                        }
                    )
                    event_name = None
                    data_lines.clear()
                    continue

                if line.startswith(":"):
                    continue
                if line.startswith("event:"):
                    event_name = line.removeprefix("event:").strip()
                    continue
                if line.startswith("data:"):
                    data_lines.append(line.removeprefix("data:").lstrip())
                    continue

            if data_lines:
                payload_raw = "\n".join(data_lines).strip()
                try:
                    parsed = json.loads(payload_raw)
                    payload = parsed if isinstance(parsed, dict) else {"_raw": payload_raw}
                except json.JSONDecodeError:
                    payload = {"_raw": payload_raw}
                events.append({"event": event_name or "message", "payload": payload})
        return events


def _now_utc() -> datetime:
    return datetime.now(UTC)


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat()


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return _iso(value)
    if isinstance(value, UUID):
        return str(value)
    if hasattr(value, "name"):
        return str(value.name)
    return str(value)


def _parse_dt_maybe(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    normalized = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _extract_last_assistant_message(session_detail: dict[str, Any]) -> dict[str, Any] | None:
    messages = session_detail.get("messages")
    if not isinstance(messages, list):
        return None
    for item in reversed(messages):
        if isinstance(item, dict) and str(item.get("role", "")).strip() == "assistant":
            return item
    return None


def _extract_mcp_calls(message: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(message, dict):
        return []
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        return []
    output: list[dict[str, Any]] = []
    for item in tool_calls:
        if not isinstance(item, dict):
            continue
        if str(item.get("type", "")).strip().lower() != "mcp_call":
            continue
        output.append(dict(item))
    return output


def _event_counter(events: list[dict[str, Any]]) -> dict[str, int]:
    counter: dict[str, int] = {}
    for item in events:
        name = str(item.get("event", "message"))
        counter[name] = int(counter.get(name, 0)) + 1
    return counter


async def _run_turn(
    *,
    api: FrontendLikeClient,
    session_id: str,
    message: str,
    turn_index: int,
    language: str = "zh",
) -> TurnAudit:
    sse_events = await api.post_sse(
        "/chat/send-openai-stream",
        params={"language": language},
        json_body={"session_id": session_id, "message": message},
    )
    done_payload: dict[str, Any] | None = None
    stream_error: str | None = None
    text_parts: list[str] = []
    for item in sse_events:
        payload = item.get("payload")
        if not isinstance(payload, dict):
            continue
        if payload.get("type") == "text_delta":
            delta = payload.get("delta")
            if isinstance(delta, str) and delta:
                text_parts.append(delta)
        if payload.get("type") == "done":
            done_payload = payload
            raw_error = payload.get("stream_error")
            if isinstance(raw_error, str) and raw_error.strip():
                stream_error = raw_error.strip()

    session_detail = await api.get_json(f"/sessions/{session_id}")
    last_assistant = _extract_last_assistant_message(session_detail)
    mcp_calls = _extract_mcp_calls(last_assistant)
    mcp_call_names = [
        str(item.get("name", "")).strip()
        for item in mcp_calls
        if isinstance(item.get("name"), str) and str(item.get("name")).strip()
    ]
    text_preview = "".join(text_parts).strip()
    if not text_preview and isinstance(last_assistant, dict):
        content = last_assistant.get("content")
        if isinstance(content, str):
            text_preview = content.strip()
    if len(text_preview) > 500:
        text_preview = f"{text_preview[:500]}..."

    return TurnAudit(
        turn_index=turn_index,
        user_message=message,
        done_payload=done_payload,
        done_phase=str(done_payload.get("phase")) if isinstance(done_payload, dict) else None,
        text_preview=text_preview,
        mcp_calls=mcp_calls,
        mcp_call_names=mcp_call_names,
        sse_event_counter=_event_counter(sse_events),
        stream_error=stream_error,
    )


async def _load_transition_rows(session_id: UUID) -> list[dict[str, Any]]:
    await db_module.init_postgres(ensure_schema=False)
    assert db_module.AsyncSessionLocal is not None
    async with db_module.AsyncSessionLocal() as db:
        rows = (
            await db.scalars(
                select(PhaseTransition)
                .where(PhaseTransition.session_id == session_id)
                .order_by(PhaseTransition.created_at.asc()),
            )
        ).all()
        return [
            {
                "id": str(item.id),
                "from_phase": item.from_phase,
                "to_phase": item.to_phase,
                "trigger": item.trigger,
                "metadata": item.metadata_ if isinstance(item.metadata_, dict) else {},
                "created_at": _iso(item.created_at),
            }
            for item in rows
        ]


def _prompt_tool_sync_check() -> dict[str, Any]:
    deployment_allowed = set(_TRADING_DEPLOYMENT_TOOL_NAMES)
    mcp_tool_names = {name for name in TRADING_MCP_TOOL_NAMES if name.startswith("trading_")}
    core_runtime_tools = {name for name in mcp_tool_names if name not in {"trading_ping", "trading_capabilities"}}

    deployment_skill_path = (
        Path(__file__).resolve().parent.parent / "src" / "agents" / "skills" / "deployment" / "skills.md"
    )
    skill_listed_tools: set[str] = set()
    if deployment_skill_path.exists():
        lines = deployment_skill_path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            stripped = line.strip()
            if not stripped.startswith("-"):
                continue
            if "`trading_" not in stripped:
                continue
            tool_name = stripped.split("`", 2)[1].strip()
            if tool_name:
                skill_listed_tools.add(tool_name)

    create_sig = inspect.signature(trading_create_paper_deployment)
    create_params = [name for name in create_sig.parameters.keys()]
    return {
        "orchestrator_allowed_tools": sorted(deployment_allowed),
        "trading_mcp_registered_tools": sorted(core_runtime_tools),
        "deployment_skill_listed_tools": sorted(skill_listed_tools),
        "missing_in_orchestrator": sorted(core_runtime_tools - deployment_allowed),
        "missing_in_mcp": sorted(deployment_allowed - core_runtime_tools),
        "missing_in_deployment_skill": sorted(deployment_allowed - skill_listed_tools),
        "extra_in_deployment_skill": sorted(skill_listed_tools - deployment_allowed),
        "trading_create_paper_deployment_params": create_params,
    }


def _pick_target_deployment(
    *,
    deployments_after: list[dict[str, Any]],
    baseline_ids: set[str],
    strategy_id: str | None,
    deployment_id_from_artifacts: str | None,
) -> dict[str, Any] | None:
    by_new = [
        item
        for item in deployments_after
        if str(item.get("deployment_id")) not in baseline_ids
    ]
    if deployment_id_from_artifacts:
        for item in deployments_after:
            if str(item.get("deployment_id")) == deployment_id_from_artifacts:
                return item
    if strategy_id:
        by_strategy = [
            item for item in deployments_after if str(item.get("strategy_id")) == strategy_id
        ]
        if by_strategy:
            return sorted(
                by_strategy,
                key=lambda item: _parse_dt_maybe(item.get("created_at")) or datetime.min.replace(tzinfo=UTC),
                reverse=True,
            )[0]
    if by_new:
        return sorted(
            by_new,
            key=lambda item: _parse_dt_maybe(item.get("created_at")) or datetime.min.replace(tzinfo=UTC),
            reverse=True,
        )[0]
    return None


async def _collect_notifications_since(
    *,
    user_id: UUID,
    since: datetime,
) -> tuple[list[NotificationOutbox], list[NotificationDeliveryAttempt], list[BacktestJob]]:
    await db_module.init_postgres(ensure_schema=False)
    assert db_module.AsyncSessionLocal is not None
    async with db_module.AsyncSessionLocal() as db:
        outbox_rows = (
            await db.scalars(
                select(NotificationOutbox)
                .where(
                    NotificationOutbox.user_id == user_id,
                    NotificationOutbox.created_at >= since,
                )
                .order_by(NotificationOutbox.created_at.asc()),
            )
        ).all()
        outbox_list = list(outbox_rows)
        outbox_ids = [item.id for item in outbox_list]
        attempts: list[NotificationDeliveryAttempt] = []
        if outbox_ids:
            attempts = list(
                (
                    await db.scalars(
                        select(NotificationDeliveryAttempt)
                        .where(NotificationDeliveryAttempt.outbox_id.in_(outbox_ids))
                        .order_by(NotificationDeliveryAttempt.attempted_at.asc()),
                    )
                ).all()
            )
        backtest_jobs = list(
            (
                await db.scalars(
                    select(BacktestJob)
                    .where(
                        BacktestJob.user_id == user_id,
                        BacktestJob.created_at >= since - timedelta(minutes=1),
                    )
                    .order_by(BacktestJob.created_at.desc()),
                )
            ).all()
        )
        return outbox_list, attempts, backtest_jobs


async def run_audit() -> AuditReport:
    started_at = _now_utc()
    base_url = "http://127.0.0.1:8000/api/v1"
    login_email = "2@test.com"
    login_password = "123456"

    api = FrontendLikeClient(base_url, timeout_seconds=300.0)
    turns: list[TurnAudit] = []
    user_id: str | None = None
    session_id: str | None = None
    strategy_id: str | None = None
    deployment_id: str | None = None
    try:
        await api.login(email=login_email, password=login_password)
        me = await api.get_json("/auth/me")
        user_id = str(me.get("user_id"))
        baseline_deployments = await api.get_json("/deployments")
        baseline_ids = {str(item.get("deployment_id")) for item in baseline_deployments if isinstance(item, dict)}

        thread = await api.post_json("/chat/new-thread", json_body={"metadata": {"audit": "frontend_like_chain"}})
        session_id = str(thread["session_id"])
        current_phase = str(thread.get("phase", ""))

        turn_index = 1
        if current_phase == "pre_strategy":
            pre_message = (
                "我想做一个加密货币策略：市场 crypto，标的是 BTCUSD，机会频率选择 multiple_per_day，"
                "持有周期 intraday。请直接进入下一阶段。"
            )
            turn = await _run_turn(
                api=api,
                session_id=session_id,
                message=pre_message,
                turn_index=turn_index,
            )
            turns.append(turn)
            turn_index += 1
            current_phase = turn.done_phase or current_phase

        strategy_prompts = [
            (
                "请直接生成一个可运行的 BTCUSD 1m 策略 DSL，并且不要等我点确认按钮："
                "你自己完成 strategy_upsert 保存，然后创建 backtest 任务并追踪到完成；"
                "回测完成后把策略标记为 strategy_confirmed=true 并进入 deployment 阶段。"
                "不要让我提供任何 strategy_id 或 session_id。"
            ),
            "继续完成：回测完成后请直接确认策略并进入 deployment，不要让我手工确认。",
            "继续，不要提问，直接完成 strategy->deployment 自动推进。",
        ]
        while current_phase == "strategy" and turn_index <= 10:
            prompt = strategy_prompts[min(turn_index - 1, len(strategy_prompts) - 1)]
            turn = await _run_turn(
                api=api,
                session_id=session_id,
                message=prompt,
                turn_index=turn_index,
            )
            turns.append(turn)
            turn_index += 1
            current_phase = turn.done_phase or current_phase
            if current_phase == "deployment":
                break

        deployment_prompts = [
            (
                "现在请直接使用我已绑定的默认 Alpaca paper 账号创建部署并自动启动，"
                "不要让我提供任何 strategy_id/deployment_id。完成后告诉我 deployed 状态。"
            ),
            "继续把 deployment 启动到 active/deployed，并汇报 deployment_id。",
            "请继续执行，直到 deployment 处于 active/deployed。",
        ]
        while current_phase == "deployment" and turn_index <= 15:
            prompt = deployment_prompts[min(turn_index - 1, len(deployment_prompts) - 1)]
            turn = await _run_turn(
                api=api,
                session_id=session_id,
                message=prompt,
                turn_index=turn_index,
            )
            turns.append(turn)
            turn_index += 1
            current_phase = turn.done_phase or current_phase
            if any(name in {"trading_start_deployment", "trading_create_paper_deployment"} for name in turn.mcp_call_names):
                # give runtime a short window to settle before deciding whether another prompt is needed
                await asyncio.sleep(2.0)
            detail = await api.get_json(f"/sessions/{session_id}")
            artifacts = detail.get("artifacts") if isinstance(detail, dict) else {}
            deployment_block = artifacts.get("deployment") if isinstance(artifacts, dict) else {}
            profile = deployment_block.get("profile") if isinstance(deployment_block, dict) else {}
            runtime = deployment_block.get("runtime") if isinstance(deployment_block, dict) else {}
            status_hint = str(profile.get("deployment_status") or runtime.get("phase_status") or "").lower().strip()
            if status_hint == "deployed":
                break

        session_detail = await api.get_json(f"/sessions/{session_id}")
        final_phase = str(session_detail.get("current_phase", ""))
        artifacts = session_detail.get("artifacts") if isinstance(session_detail, dict) else {}
        strategy_profile = (
            artifacts.get("strategy", {}).get("profile", {})
            if isinstance(artifacts, dict)
            else {}
        )
        deployment_profile = (
            artifacts.get("deployment", {}).get("profile", {})
            if isinstance(artifacts, dict)
            else {}
        )
        strategy_id = str(deployment_profile.get("strategy_id") or strategy_profile.get("strategy_id") or "").strip() or None
        deployment_id = str(deployment_profile.get("deployment_id") or "").strip() or None

        transitions = await _load_transition_rows(UUID(session_id))
        trading_tools_used = sorted(
            {
                name
                for turn in turns
                for name in turn.mcp_call_names
                if name.startswith("trading_")
            }
        )
        auto_transition = any(
            row.get("from_phase") == "strategy"
            and row.get("to_phase") == "deployment"
            and row.get("trigger") == "ai_output"
            for row in transitions
        )
        prompt_tool_sync = _prompt_tool_sync_check()
        step1_issues: list[str] = []
        if final_phase != "deployment":
            step1_issues.append(f"session final phase is {final_phase}, not deployment")
        if not auto_transition:
            step1_issues.append("missing strategy->deployment transition with trigger=ai_output")
        if "trading_create_paper_deployment" not in trading_tools_used:
            step1_issues.append("trading_create_paper_deployment was not observed in assistant MCP calls")
        if "trading_start_deployment" not in trading_tools_used:
            step1_issues.append("trading_start_deployment was not observed in assistant MCP calls")
        if prompt_tool_sync["missing_in_orchestrator"]:
            step1_issues.append(f"orchestrator missing trading MCP tools: {prompt_tool_sync['missing_in_orchestrator']}")
        if prompt_tool_sync["missing_in_mcp"]:
            step1_issues.append(f"orchestrator references tools absent in MCP trading server: {prompt_tool_sync['missing_in_mcp']}")
        if prompt_tool_sync["missing_in_deployment_skill"]:
            step1_issues.append(f"deployment skill markdown missing tools: {prompt_tool_sync['missing_in_deployment_skill']}")

        deployments_after = await api.get_json("/deployments")
        target_deployment = _pick_target_deployment(
            deployments_after=deployments_after,
            baseline_ids=baseline_ids,
            strategy_id=strategy_id,
            deployment_id_from_artifacts=deployment_id,
        )
        if target_deployment is not None:
            deployment_id = str(target_deployment.get("deployment_id"))

        step1 = Step1Result(
            ok=not step1_issues,
            session_id=session_id,
            session_final_phase=final_phase,
            strategy_id=strategy_id,
            deployment_id=deployment_id,
            auto_transition_strategy_to_deployment=auto_transition,
            transition_records=transitions,
            trading_mcp_tools_used=trading_tools_used,
            prompt_tool_sync=prompt_tool_sync,
            issues=step1_issues,
        )

        step2_issues: list[str] = []
        runtime_samples: list[dict[str, Any]] = []
        process_now_calls: list[dict[str, Any]] = []
        signals_count = 0
        orders_count = 0
        positions_count = 0
        pnl_count = 0
        scheduler_state: dict[str, Any] = {}
        deployment_status: str | None = None
        alpaca_real_order_detected = False
        if deployment_id is None:
            step2_issues.append("step1 did not produce a target deployment_id")
        else:
            poll_deadline = _now_utc() + timedelta(seconds=120)
            while _now_utc() < poll_deadline:
                dep = await api.get_json(f"/deployments/{deployment_id}")
                orders = await api.get_json(f"/deployments/{deployment_id}/orders")
                positions = await api.get_json(f"/deployments/{deployment_id}/positions")
                pnl = await api.get_json(f"/deployments/{deployment_id}/pnl")
                signals = await api.get_json(f"/deployments/{deployment_id}/signals", params={"limit": 100})
                runtime_state = dep.get("run", {}).get("runtime_state", {}) if isinstance(dep, dict) else {}
                scheduler_state = runtime_state.get("scheduler", {}) if isinstance(runtime_state, dict) else {}
                deployment_status = str(dep.get("status", ""))
                signals_count = len(signals) if isinstance(signals, list) else 0
                orders_count = len(orders) if isinstance(orders, list) else 0
                positions_count = len(positions) if isinstance(positions, list) else 0
                pnl_count = len(pnl) if isinstance(pnl, list) else 0
                if isinstance(orders, list):
                    alpaca_real_order_detected = any(
                        isinstance(item, dict)
                        and isinstance(item.get("provider_order_id"), str)
                        and item.get("provider_order_id")
                        and not str(item.get("provider_order_id")).startswith("paper-")
                        for item in orders
                    )

                runtime_samples.append(
                    {
                        "polled_at": _iso(_now_utc()),
                        "deployment_status": deployment_status,
                        "signals_count": signals_count,
                        "orders_count": orders_count,
                        "positions_count": positions_count,
                        "pnl_snapshots_count": pnl_count,
                        "run_status": dep.get("run", {}).get("status") if isinstance(dep, dict) else None,
                        "last_bar_time": dep.get("run", {}).get("last_bar_time") if isinstance(dep, dict) else None,
                        "scheduler": scheduler_state,
                    }
                )
                if signals_count >= 2 and orders_count >= 1 and pnl_count >= 1:
                    break
                await asyncio.sleep(5.0)

            if orders_count < 1:
                for _ in range(8):
                    response = await api.post_json(
                        f"/deployments/{deployment_id}/process-now",
                        json_body={},
                    )
                    process_now_calls.append(response if isinstance(response, dict) else {"raw": response})
                    await asyncio.sleep(1.0)
                orders = await api.get_json(f"/deployments/{deployment_id}/orders")
                signals = await api.get_json(f"/deployments/{deployment_id}/signals", params={"limit": 100})
                pnl = await api.get_json(f"/deployments/{deployment_id}/pnl")
                positions = await api.get_json(f"/deployments/{deployment_id}/positions")
                signals_count = len(signals) if isinstance(signals, list) else 0
                orders_count = len(orders) if isinstance(orders, list) else 0
                positions_count = len(positions) if isinstance(positions, list) else 0
                pnl_count = len(pnl) if isinstance(pnl, list) else 0
                if isinstance(orders, list):
                    alpaca_real_order_detected = any(
                        isinstance(item, dict)
                        and isinstance(item.get("provider_order_id"), str)
                        and item.get("provider_order_id")
                        and not str(item.get("provider_order_id")).startswith("paper-")
                        for item in orders
                    )

            if deployment_status != "active":
                step2_issues.append(f"deployment status is {deployment_status}, expected active")
            if signals_count < 1:
                step2_issues.append("no signal events were observed for deployment")
            if orders_count < 1:
                step2_issues.append("no orders were observed for deployment")
            if pnl_count < 1:
                step2_issues.append("no pnl snapshots were observed")
            if not alpaca_real_order_detected:
                step2_issues.append("no provider_order_id from Alpaca detected (possible local-simulated-only path)")
            if not scheduler_state:
                step2_issues.append("deployment run scheduler state is empty")

        step2 = Step2Result(
            ok=not step2_issues,
            deployment_id=deployment_id,
            deployment_status=deployment_status,
            runtime_scheduler_state=scheduler_state,
            signals_count=signals_count,
            orders_count=orders_count,
            positions_count=positions_count,
            pnl_snapshots_count=pnl_count,
            alpaca_real_order_detected=alpaca_real_order_detected,
            runtime_poll_samples=runtime_samples,
            process_now_calls=process_now_calls,
            issues=step2_issues,
        )

        step3_issues: list[str] = []
        observed_events: list[str] = []
        outbox_summary: dict[str, Any] = {}
        attempt_summary: dict[str, Any] = {}
        backtest_job_views: list[dict[str, Any]] = []
        if user_id is None:
            step3_issues.append("missing user_id, cannot query notification outbox")
        else:
            user_uuid = UUID(user_id)
            # allow beat/worker some time to dispatch events
            await asyncio.sleep(12.0)
            outbox_rows, attempts, backtest_jobs = await _collect_notifications_since(
                user_id=user_uuid,
                since=started_at - timedelta(seconds=30),
            )
            outbox_summary = {
                "total": len(outbox_rows),
                "by_event_type": {},
                "by_status": {},
            }
            for row in outbox_rows:
                event_type = str(row.event_type)
                status_key = str(row.status)
                outbox_summary["by_event_type"][event_type] = int(outbox_summary["by_event_type"].get(event_type, 0)) + 1
                outbox_summary["by_status"][status_key] = int(outbox_summary["by_status"].get(status_key, 0)) + 1
            attempt_summary = {
                "total": len(attempts),
                "success": sum(1 for item in attempts if bool(item.success)),
                "failed": sum(1 for item in attempts if not bool(item.success)),
                "telegram_message_ids": [
                    str(item.provider_message_id)
                    for item in attempts
                    if item.provider == "telegram" and item.provider_message_id
                ],
            }
            backtest_job_views = [
                {
                    "job_id": str(item.id),
                    "strategy_id": str(item.strategy_id),
                    "status": item.status,
                    "created_at": _iso(item.created_at),
                    "completed_at": _iso(item.completed_at),
                }
                for item in backtest_jobs[:20]
            ]
            key_events = {"BACKTEST_COMPLETED", "DEPLOYMENT_STARTED", "POSITION_OPENED", "POSITION_CLOSED"}
            observed_events = sorted(
                event for event in key_events if int(outbox_summary.get("by_event_type", {}).get(event, 0)) > 0
            )
            missing = sorted(key_events - set(observed_events))
            if missing:
                step3_issues.append(f"missing notification events in outbox: {missing}")
            if int(outbox_summary.get("by_status", {}).get("sent", 0)) < 1:
                step3_issues.append("no outbox row reached sent status")
            if not attempt_summary.get("telegram_message_ids"):
                step3_issues.append("no telegram delivery attempt with provider_message_id observed")
            # functional gaps worth surfacing even when run passes:
            if int(outbox_summary.get("by_event_type", {}).get("RISK_TRIGGERED", 0)) == 0:
                step3_issues.append("RISK_TRIGGERED event not observed; runtime currently may not enqueue this path")
            if int(outbox_summary.get("by_event_type", {}).get("EXECUTION_ANOMALY", 0)) == 0:
                step3_issues.append("EXECUTION_ANOMALY event not observed; runtime currently may not enqueue this path")

        step3 = Step3Result(
            ok=not step3_issues,
            outbox_summary=outbox_summary,
            attempt_summary=attempt_summary,
            backtest_jobs=backtest_job_views,
            key_events_observed=observed_events,
            issues=step3_issues,
        )

        finished_at = _now_utc()
        return AuditReport(
            started_at=_iso(started_at) or "",
            finished_at=_iso(finished_at) or "",
            duration_seconds=(finished_at - started_at).total_seconds(),
            base_url=base_url,
            user_email=login_email,
            user_id=user_id,
            session_id=session_id,
            strategy_id=strategy_id,
            deployment_id=deployment_id,
            turns=turns,
            step1=step1,
            step2=step2,
            step3=step3,
        )
    finally:
        await api.aclose()


async def _async_main() -> int:
    report = await run_audit()
    root = Path(__file__).resolve().parent.parent
    out_dir = root / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"audit_frontend_like_full_chain_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
    payload = {
        **asdict(report),
        "turns": [asdict(item) for item in report.turns],
        "step1": asdict(report.step1),
        "step2": asdict(report.step2),
        "step3": asdict(report.step3),
    }
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    summary = {
        "step1_ok": report.step1.ok,
        "step2_ok": report.step2.ok,
        "step3_ok": report.step3.ok,
        "session_id": report.session_id,
        "strategy_id": report.strategy_id,
        "deployment_id": report.deployment_id,
        "report_path": str(out_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    return asyncio.run(_async_main())


if __name__ == "__main__":
    raise SystemExit(main())
