#!/usr/bin/env python3
"""Frontend-like live audit for strategy->deployment->paper-trading->notifications chain.

This script intentionally does NOT reuse project test cases. It drives real API endpoints,
uses OpenAI streaming turns, and inspects DB/runtime artifacts for evidence.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

import requests
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.orchestrator.constants import _TRADING_DEPLOYMENT_TOOL_NAMES
from src.config import settings
from src.engine import DataLoader
from src.engine.backtest.service import (
    BacktestJobNotFoundError,
    create_backtest_job,
    get_backtest_job_view,
    schedule_backtest_job,
)
from src.engine.execution.credentials import CredentialCipher
from src.mcp.backtest.tools import TOOL_NAMES as BACKTEST_MCP_TOOL_NAMES
from src.mcp.market_data.tools import TOOL_NAMES as MARKET_DATA_MCP_TOOL_NAMES
from src.mcp.strategy.tools import TOOL_NAMES as STRATEGY_MCP_TOOL_NAMES
from src.mcp.trading.tools import TOOL_NAMES as TRADING_MCP_TOOL_NAMES
from src.models import database as db_module

UUID_PATTERN = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")


@dataclass(slots=True)
class TurnResult:
    message: str
    phase_after: str | None
    missing_fields: list[str]
    done_payload: dict[str, Any]
    text: str
    mcp_tools: list[str]


@dataclass(slots=True)
class StepOneResult:
    ok: bool
    session_id: str
    phase_start: str
    phase_end: str
    strategy_to_deployment_transition: bool
    transition_trigger: str | None
    deployment_id: str | None
    deployment_status: str | None
    used_mcp_tools: list[str] = field(default_factory=list)
    turn_logs: list[dict[str, Any]] = field(default_factory=list)
    prompt_tool_sync: dict[str, Any] = field(default_factory=dict)
    findings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class StepTwoResult:
    ok: bool
    deployment_id: str
    deployment_status_final: str | None
    celery_scheduler_evidence: dict[str, Any]
    market_data_evidence: dict[str, Any]
    order_evidence: dict[str, Any]
    account_update_evidence: dict[str, Any]
    findings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class StepThreeResult:
    ok: bool
    backtest_job: dict[str, Any]
    notification_events: dict[str, Any]
    telegram_delivery: dict[str, Any]
    findings: list[str] = field(default_factory=list)


class ApiClient:
    def __init__(self, *, base_url: str, email: str, password: str, language: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.email = email
        self.password = password
        self.language = language
        self.session = requests.Session()
        self.access_token: str | None = None

    def _url(self, path: str) -> str:
        if path.startswith("http://") or path.startswith("https://"):
            return path
        return f"{self.base_url}{path}"

    def _headers(self) -> dict[str, str]:
        if not self.access_token:
            return {"content-type": "application/json"}
        return {
            "content-type": "application/json",
            "Authorization": f"Bearer {self.access_token}",
        }

    def login(self) -> dict[str, Any]:
        resp = self.session.post(
            self._url("/auth/login"),
            json={"email": self.email, "password": self.password},
            timeout=30,
        )
        resp.raise_for_status()
        payload = resp.json()
        self.access_token = str(payload["access_token"])
        return payload

    def create_thread(self) -> dict[str, Any]:
        resp = self.session.post(
            self._url("/chat/new-thread"),
            headers=self._headers(),
            json={},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def get_session(self, session_id: str) -> dict[str, Any]:
        resp = self.session.get(
            self._url(f"/sessions/{session_id}"),
            headers=self._headers(),
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def list_deployments(self) -> list[dict[str, Any]]:
        resp = self.session.get(
            self._url("/deployments"),
            headers=self._headers(),
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else []

    def get_deployment(self, deployment_id: str) -> dict[str, Any]:
        resp = self.session.get(
            self._url(f"/deployments/{deployment_id}"),
            headers=self._headers(),
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def get_orders(self, deployment_id: str) -> list[dict[str, Any]]:
        resp = self.session.get(
            self._url(f"/deployments/{deployment_id}/orders"),
            headers=self._headers(),
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else []

    def get_positions(self, deployment_id: str) -> list[dict[str, Any]]:
        resp = self.session.get(
            self._url(f"/deployments/{deployment_id}/positions"),
            headers=self._headers(),
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else []

    def get_pnl(self, deployment_id: str) -> list[dict[str, Any]]:
        resp = self.session.get(
            self._url(f"/deployments/{deployment_id}/pnl"),
            headers=self._headers(),
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else []

    def get_signals(self, deployment_id: str) -> list[dict[str, Any]]:
        resp = self.session.get(
            self._url(f"/deployments/{deployment_id}/signals"),
            headers=self._headers(),
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else []

    def stream_chat_turn(self, *, session_id: str, message: str, timeout_seconds: int) -> TurnResult:
        url = self._url(f"/chat/send-openai-stream?language={self.language}")
        resp = self.session.post(
            url,
            headers=self._headers(),
            json={"session_id": session_id, "message": message},
            timeout=timeout_seconds,
            stream=True,
        )
        resp.raise_for_status()

        text_parts: list[str] = []
        done_payload: dict[str, Any] = {}
        event_name = "message"
        data_lines: list[str] = []

        def _flush_event() -> tuple[str, dict[str, Any] | None] | None:
            nonlocal event_name, data_lines
            if not data_lines:
                event_name = "message"
                return None
            raw = "\n".join(data_lines)
            data_lines = []
            parsed: dict[str, Any] | None
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = None
            output = (event_name, parsed)
            event_name = "message"
            return output

        for raw_line in resp.iter_lines(decode_unicode=True):
            if raw_line is None:
                continue
            line = str(raw_line)
            if line == "":
                flushed = _flush_event()
                if flushed is None:
                    continue
                evt, payload = flushed
                if not isinstance(payload, dict):
                    continue
                if evt == "stream" and payload.get("type") == "text_delta":
                    delta = payload.get("delta")
                    if isinstance(delta, str):
                        text_parts.append(delta)
                if evt == "stream" and payload.get("type") == "done":
                    done_payload = dict(payload)
                continue
            if line.startswith(":"):
                continue
            if line.startswith("event:"):
                event_name = line.split(":", 1)[1].strip() or "message"
                continue
            if line.startswith("data:"):
                data_lines.append(line.split(":", 1)[1].lstrip())

        flushed = _flush_event()
        if flushed is not None:
            evt, payload = flushed
            if isinstance(payload, dict) and evt == "stream" and payload.get("type") == "done":
                done_payload = dict(payload)

        if not done_payload:
            raise RuntimeError("stream ended without done payload")

        phase_after = done_payload.get("phase")
        if not isinstance(phase_after, str):
            phase_after = None

        missing_fields = done_payload.get("missing_fields")
        if not isinstance(missing_fields, list):
            missing_fields = []

        session_detail = self.get_session(session_id)
        messages = session_detail.get("messages") if isinstance(session_detail.get("messages"), list) else []
        assistant_message = None
        for item in reversed(messages):
            if isinstance(item, dict) and item.get("role") == "assistant":
                assistant_message = item
                break

        tool_calls = assistant_message.get("tool_calls") if isinstance(assistant_message, dict) else []
        mcp_tools: list[str] = []
        if isinstance(tool_calls, list):
            for item in tool_calls:
                if not isinstance(item, dict):
                    continue
                if item.get("type") != "mcp_call":
                    continue
                name = item.get("name")
                if isinstance(name, str) and name.strip():
                    mcp_tools.append(name.strip())

        return TurnResult(
            message=message,
            phase_after=phase_after,
            missing_fields=[str(v) for v in missing_fields if isinstance(v, str)],
            done_payload=done_payload,
            text="".join(text_parts),
            mcp_tools=mcp_tools,
        )


def _parse_skill_tool_list(
    md_path: Path,
    *,
    anchor_line: str,
) -> list[str]:
    content = md_path.read_text(encoding="utf-8")
    tools: list[str] = []
    in_section = False
    for line in content.splitlines():
        stripped = line.strip()
        if in_section and stripped.startswith("## "):
            break
        if stripped == anchor_line:
            in_section = True
            continue
        if not in_section:
            continue
        if not stripped.startswith("- `"):
            continue
        if "`" not in stripped[3:]:
            continue
        name = stripped.split("`", 2)[1].strip()
        if name:
            tools.append(name)
    return tools


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


async def _load_phase_transitions(session_id: UUID) -> list[dict[str, Any]]:
    await db_module.init_postgres(ensure_schema=False)
    assert db_module.AsyncSessionLocal is not None
    async with db_module.AsyncSessionLocal() as db:
        rows = (
            await db.execute(
                text(
                    """
                    select from_phase, to_phase, trigger, metadata, created_at
                    from phase_transitions
                    where session_id = :session_id
                    order by created_at asc
                    """
                ),
                {"session_id": session_id},
            )
        ).mappings().all()
    await db_module.close_postgres()
    return [dict(row) for row in rows]


async def _load_broker_credentials(broker_account_id: UUID) -> dict[str, str] | None:
    await db_module.init_postgres(ensure_schema=False)
    assert db_module.AsyncSessionLocal is not None
    async with db_module.AsyncSessionLocal() as db:
        row = (
            await db.execute(
                text(
                    """
                    select encrypted_credentials
                    from broker_accounts
                    where id = :id
                    limit 1
                    """
                ),
                {"id": broker_account_id},
            )
        ).mappings().first()
    await db_module.close_postgres()
    if row is None:
        return None
    encrypted = row.get("encrypted_credentials")
    if not isinstance(encrypted, str) or not encrypted.strip():
        return None
    try:
        payload = CredentialCipher().decrypt(encrypted)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return {str(k): str(v) for k, v in payload.items() if isinstance(v, str)}


async def _trigger_backtest_completion(
    *,
    strategy_id: UUID,
    user_id: UUID,
    market: str,
    symbol: str,
    timeframe: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    loader = DataLoader()
    metadata = loader.get_symbol_metadata(market, symbol)
    available = metadata.get("available_timerange") if isinstance(metadata.get("available_timerange"), dict) else {}
    start_raw = str(available.get("start") or "").strip()
    end_raw = str(available.get("end") or "").strip()
    if not start_raw or not end_raw:
        return {
            "ok": False,
            "reason": "missing_data_coverage",
            "market": market,
            "symbol": symbol,
            "timeframe": timeframe,
        }

    end_dt = datetime.fromisoformat(end_raw.replace("Z", "+00:00")).astimezone(UTC)
    start_dt = datetime.fromisoformat(start_raw.replace("Z", "+00:00")).astimezone(UTC)
    target_start = max(start_dt, end_dt - timedelta(days=2))

    await db_module.init_postgres(ensure_schema=False)
    assert db_module.AsyncSessionLocal is not None
    async with db_module.AsyncSessionLocal() as db:
        receipt = await create_backtest_job(
            db,
            strategy_id=strategy_id,
            start_date=target_start.isoformat(),
            end_date=end_dt.isoformat(),
            user_id=user_id,
            auto_commit=True,
        )
        await schedule_backtest_job(receipt.job_id)

    deadline = time.time() + timeout_seconds
    last_view: dict[str, Any] | None = None
    while time.time() < deadline:
        await asyncio.sleep(2.0)
        await db_module.init_postgres(ensure_schema=False)
        assert db_module.AsyncSessionLocal is not None
        async with db_module.AsyncSessionLocal() as db:
            try:
                view = await get_backtest_job_view(db, job_id=receipt.job_id, user_id=user_id)
            except BacktestJobNotFoundError:
                continue
            last_view = {
                "job_id": str(view.job_id),
                "status": view.status,
                "progress": view.progress,
                "current_step": view.current_step,
                "error": view.error,
            }
            if view.status in {"done", "failed"}:
                await db_module.close_postgres()
                return {
                    "ok": view.status == "done",
                    "job_id": str(view.job_id),
                    "status": view.status,
                    "progress": view.progress,
                    "current_step": view.current_step,
                    "error": view.error,
                    "timeframe": timeframe,
                    "market": market,
                    "symbol": symbol,
                    "start_date": target_start.isoformat(),
                    "end_date": end_dt.isoformat(),
                }

    await db_module.close_postgres()
    return {
        "ok": False,
        "reason": "timeout",
        "job": last_view,
        "timeframe": timeframe,
        "market": market,
        "symbol": symbol,
        "start_date": target_start.isoformat(),
        "end_date": end_dt.isoformat(),
    }


async def _query_notification_state(*, user_id: UUID, since_at: datetime) -> dict[str, Any]:
    await db_module.init_postgres(ensure_schema=False)
    assert db_module.AsyncSessionLocal is not None
    async with db_module.AsyncSessionLocal() as db:
        outbox_rows = (
            await db.execute(
                text(
                    """
                    select id, channel, event_type, event_key, payload, status,
                           retry_count, max_retries, sent_at, next_retry_at, last_error, created_at
                    from notification_outbox
                    where user_id = :user_id
                      and created_at >= :since_at
                    order by created_at asc
                    """
                ),
                {"user_id": user_id, "since_at": since_at},
            )
        ).mappings().all()
        attempts = (
            await db.execute(
                text(
                    """
                    select a.outbox_id, a.provider, a.success, a.error_code, a.error_message, a.attempted_at
                    from notification_delivery_attempts a
                    join notification_outbox o on o.id = a.outbox_id
                    where o.user_id = :user_id
                      and o.created_at >= :since_at
                    order by a.attempted_at asc
                    """
                ),
                {"user_id": user_id, "since_at": since_at},
            )
        ).mappings().all()
        binding = (
            await db.execute(
                text(
                    """
                    select provider, status, external_chat_id
                    from social_connector_bindings
                    where user_id = :user_id
                      and provider = 'telegram'
                    limit 1
                    """
                ),
                {"user_id": user_id},
            )
        ).mappings().first()
    await db_module.close_postgres()

    events = [dict(row) for row in outbox_rows]
    attempt_rows = [dict(row) for row in attempts]
    by_event: dict[str, int] = {}
    for row in events:
        key = str(row.get("event_type", "")).upper()
        by_event[key] = by_event.get(key, 0) + 1

    return {
        "events": events,
        "attempts": attempt_rows,
        "event_type_counts": by_event,
        "telegram_binding": dict(binding) if binding is not None else None,
    }


def _ensure_phase(client: ApiClient, *, session_id: str, target_phase: str, max_turns: int) -> tuple[str, list[TurnResult]]:
    turns: list[TurnResult] = []
    phase = client.get_session(session_id).get("current_phase")
    if phase == target_phase:
        return phase, turns

    filler_by_field = {
        "target_market": "crypto",
        "target_instrument": "BTCUSD",
        "opportunity_frequency_bucket": "multiple_per_day",
        "holding_period_bucket": "intraday",
    }

    for _ in range(max_turns):
        if phase == target_phase:
            break
        if phase == "pre_strategy":
            message = (
                "请记录以下策略范围并直接推进阶段：\n"
                "target_market=crypto\n"
                "target_instrument=BTCUSD\n"
                "opportunity_frequency_bucket=multiple_per_day\n"
                "holding_period_bucket=intraday\n"
                "不需要我点击任何按钮。"
            )
        elif phase == "strategy":
            message = (
                "请在本轮完成：\n"
                "1) 构建并校验完整 DSL（crypto, BTCUSD, timeframe=1m）\n"
                "2) 直接调用 strategy_upsert_dsl 保存\n"
                "3) 生成 AGENT_STATE_PATCH，把 strategy_confirmed 设为 true，自动推进 deployment\n"
                "约束：至少一个因子；long entry 用 cmp(1 gt 0) 恒真；long signal_exit 用 cmp(0 gt 1) 恒假。\n"
                "不要让我手动提供 strategy_id。"
            )
        elif phase == "deployment":
            message = (
                "请立即创建并启动一个新的 paper deployment，"
                "不要使用我手工输入的 strategy_id/deployment_id，"
                "并在完成后汇报 deployment_id 与状态。"
            )
        else:
            message = "继续执行到 deployment 阶段，不需要我额外操作。"

        result = client.stream_chat_turn(
            session_id=session_id,
            message=message,
            timeout_seconds=360,
        )
        turns.append(result)
        phase = result.phase_after or phase

        if phase == "pre_strategy" and result.missing_fields:
            missing_values = [
                f"{field}={filler_by_field[field]}"
                for field in result.missing_fields
                if field in filler_by_field
            ]
            if missing_values:
                patch_turn = client.stream_chat_turn(
                    session_id=session_id,
                    message="\n".join(missing_values),
                    timeout_seconds=360,
                )
                turns.append(patch_turn)
                phase = patch_turn.phase_after or phase

    return str(phase), turns


def _evaluate_prompt_tool_sync() -> dict[str, Any]:
    strategy_skills = _parse_skill_tool_list(
        Path("src/agents/skills/strategy/skills.md"),
        anchor_line="- In this phase, only use:",
    )
    deployment_skills = _parse_skill_tool_list(
        Path("src/agents/skills/deployment/skills.md"),
        anchor_line="- Available tools in this phase:",
    )

    strategy_runtime_set = set(STRATEGY_MCP_TOOL_NAMES) | set(BACKTEST_MCP_TOOL_NAMES) | set(MARKET_DATA_MCP_TOOL_NAMES)
    deployment_runtime_set = set(TRADING_MCP_TOOL_NAMES)

    strategy_missing = sorted(name for name in strategy_skills if name not in strategy_runtime_set)
    deployment_missing = sorted(name for name in deployment_skills if name not in deployment_runtime_set)

    orchestrator_expected = set(_TRADING_DEPLOYMENT_TOOL_NAMES)
    trading_tool_set = set(TRADING_MCP_TOOL_NAMES)

    return {
        "strategy_prompt_tools": strategy_skills,
        "deployment_prompt_tools": deployment_skills,
        "strategy_missing_in_runtime": strategy_missing,
        "deployment_missing_in_runtime": deployment_missing,
        "orchestrator_deployment_tools": sorted(orchestrator_expected),
        "trading_mcp_tools": sorted(trading_tool_set),
        "orchestrator_missing_in_trading_mcp": sorted(orchestrator_expected - trading_tool_set),
        "trading_mcp_not_exposed_by_orchestrator": sorted(trading_tool_set - orchestrator_expected),
    }


def run_step_one(client: ApiClient) -> StepOneResult:
    thread = client.create_thread()
    session_id = str(thread["session_id"])
    phase_start = str(thread.get("phase") or "")
    deployments_before = {str(item.get("deployment_id")) for item in client.list_deployments() if isinstance(item, dict)}

    phase_end, turns = _ensure_phase(
        client,
        session_id=session_id,
        target_phase="deployment",
        max_turns=8,
    )

    # Nudge deployment action when already in deployment phase.
    deploy_turn = client.stream_chat_turn(
        session_id=session_id,
        message=(
            "现在请自动执行部署动作：创建并启动新的 paper deployment，"
            "不要让我输入任何 ID。"
        ),
        timeout_seconds=360,
    )
    turns.append(deploy_turn)

    session = client.get_session(session_id)
    artifacts = session.get("artifacts") if isinstance(session.get("artifacts"), dict) else {}
    deployment_block = artifacts.get("deployment") if isinstance(artifacts.get("deployment"), dict) else {}
    deployment_profile = deployment_block.get("profile") if isinstance(deployment_block.get("profile"), dict) else {}

    deployment_id = deployment_profile.get("deployment_id") if isinstance(deployment_profile.get("deployment_id"), str) else None
    deployments_after = [item for item in client.list_deployments() if isinstance(item, dict)]
    new_ids = [
        str(item.get("deployment_id"))
        for item in deployments_after
        if isinstance(item.get("deployment_id"), str)
        and str(item.get("deployment_id")) not in deployments_before
    ]
    if deployment_id is None and new_ids:
        deployment_id = new_ids[0]

    deployment_status: str | None = None
    if deployment_id is not None:
        deployment = client.get_deployment(deployment_id)
        status = deployment.get("status")
        if isinstance(status, str):
            deployment_status = status

    transitions = asyncio.run(_load_phase_transitions(UUID(session_id)))
    strategy_to_deployment = [
        row
        for row in transitions
        if str(row.get("from_phase")) == "strategy" and str(row.get("to_phase")) == "deployment"
    ]
    transition_trigger = None
    if strategy_to_deployment:
        last = strategy_to_deployment[-1]
        trigger = last.get("trigger")
        if isinstance(trigger, str):
            transition_trigger = trigger

    all_tools: list[str] = []
    for turn in turns:
        all_tools.extend(turn.mcp_tools)

    sync_report = _evaluate_prompt_tool_sync()
    findings: list[str] = []
    if sync_report["strategy_missing_in_runtime"]:
        findings.append(f"strategy prompt has stale/missing tools: {sync_report['strategy_missing_in_runtime']}")
    if sync_report["deployment_missing_in_runtime"]:
        findings.append(f"deployment prompt has stale/missing tools: {sync_report['deployment_missing_in_runtime']}")
    if sync_report["orchestrator_missing_in_trading_mcp"]:
        findings.append(
            "orchestrator deployment tool policy references missing trading tools: "
            f"{sync_report['orchestrator_missing_in_trading_mcp']}"
        )

    tool_set = set(all_tools)
    if "trading_create_paper_deployment" not in tool_set:
        findings.append("deployment turn did not call trading_create_paper_deployment")
    if not ({"trading_start_deployment", "trading_create_paper_deployment"} & tool_set):
        findings.append("deployment turn did not call any start-capable trading tool")

    turn_logs = [
        {
            "message": turn.message,
            "phase_after": turn.phase_after,
            "missing_fields": turn.missing_fields,
            "mcp_tools": turn.mcp_tools,
            "done": turn.done_payload,
            "assistant_text_preview": turn.text[:500],
        }
        for turn in turns
    ]

    ok = (
        bool(strategy_to_deployment)
        and transition_trigger == "ai_output"
        and deployment_id is not None
        and deployment_status in {"active", "pending", "paused"}
        and not findings
    )

    return StepOneResult(
        ok=ok,
        session_id=session_id,
        phase_start=phase_start,
        phase_end=str(session.get("current_phase") or phase_end),
        strategy_to_deployment_transition=bool(strategy_to_deployment),
        transition_trigger=transition_trigger,
        deployment_id=deployment_id,
        deployment_status=deployment_status,
        used_mcp_tools=sorted(set(all_tools)),
        turn_logs=turn_logs,
        prompt_tool_sync=sync_report,
        findings=findings,
    )


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def run_step_two(client: ApiClient, *, step_one: StepOneResult, runtime_wait_seconds: int) -> StepTwoResult:
    if not step_one.deployment_id:
        return StepTwoResult(
            ok=False,
            deployment_id="",
            deployment_status_final=None,
            celery_scheduler_evidence={},
            market_data_evidence={},
            order_evidence={},
            account_update_evidence={},
            findings=["step1 did not produce deployment_id"],
        )

    deployment_id = step_one.deployment_id
    base_deployment = client.get_deployment(deployment_id)
    base_orders = client.get_orders(deployment_id)
    base_positions = client.get_positions(deployment_id)
    base_pnl = client.get_pnl(deployment_id)
    base_signals = client.get_signals(deployment_id)

    run_payload = base_deployment.get("run") if isinstance(base_deployment.get("run"), dict) else {}
    runtime_state = run_payload.get("runtime_state") if isinstance(run_payload.get("runtime_state"), dict) else {}
    scheduler = runtime_state.get("scheduler") if isinstance(runtime_state.get("scheduler"), dict) else {}
    baseline_last_enqueued_at = scheduler.get("last_enqueued_at")

    deadline = time.time() + runtime_wait_seconds
    observed: dict[str, Any] = {
        "orders": base_orders,
        "positions": base_positions,
        "pnl": base_pnl,
        "signals": base_signals,
        "deployment": base_deployment,
    }

    while time.time() < deadline:
        time.sleep(5)
        dep = client.get_deployment(deployment_id)
        orders = client.get_orders(deployment_id)
        positions = client.get_positions(deployment_id)
        pnl = client.get_pnl(deployment_id)
        signals = client.get_signals(deployment_id)

        observed.update(
            {
                "deployment": dep,
                "orders": orders,
                "positions": positions,
                "pnl": pnl,
                "signals": signals,
            }
        )

        new_order = len(orders) > len(base_orders)
        scheduler_next = (
            dep.get("run", {}).get("runtime_state", {}).get("scheduler", {}).get("last_enqueued_at")
            if isinstance(dep.get("run"), dict)
            else None
        )
        enqueued_changed = scheduler_next != baseline_last_enqueued_at and scheduler_next is not None
        signal_changed = len(signals) > len(base_signals)

        if new_order and enqueued_changed and signal_changed:
            break

    final_dep = observed["deployment"]
    final_orders = observed["orders"]
    final_positions = observed["positions"]
    final_pnl = observed["pnl"]
    final_signals = observed["signals"]

    final_status = final_dep.get("status") if isinstance(final_dep.get("status"), str) else None
    final_run = final_dep.get("run") if isinstance(final_dep.get("run"), dict) else {}
    final_runtime_state = final_run.get("runtime_state") if isinstance(final_run.get("runtime_state"), dict) else {}
    final_scheduler = final_runtime_state.get("scheduler") if isinstance(final_runtime_state.get("scheduler"), dict) else {}

    celery_evidence = {
        "baseline_last_enqueued_at": baseline_last_enqueued_at,
        "final_last_enqueued_at": final_scheduler.get("last_enqueued_at"),
        "changed": final_scheduler.get("last_enqueued_at") != baseline_last_enqueued_at,
        "run_status": final_run.get("status"),
        "runtime_last_signal": final_runtime_state.get("last_signal"),
        "runtime_last_signal_reason": final_runtime_state.get("last_signal_reason"),
    }

    market_data_evidence = {
        "signals_before": len(base_signals),
        "signals_after": len(final_signals),
        "signal_delta": len(final_signals) - len(base_signals),
        "runtime_reason": final_runtime_state.get("last_signal_reason"),
        "market_data_source": (
            final_runtime_state.get("execution", {}).get("market_data_source")
            if isinstance(final_runtime_state.get("execution"), dict)
            else None
        ),
    }

    order_evidence: dict[str, Any] = {
        "orders_before": len(base_orders),
        "orders_after": len(final_orders),
        "order_delta": len(final_orders) - len(base_orders),
        "provider_order_verified": False,
        "provider_order_id": None,
        "provider_lookup_status": None,
    }

    target_order = final_orders[0] if final_orders else None
    if isinstance(target_order, dict):
        provider_order_id = target_order.get("provider_order_id")
        if isinstance(provider_order_id, str) and provider_order_id.strip():
            provider_order_id = provider_order_id.strip()
            order_evidence["provider_order_id"] = provider_order_id
            if not provider_order_id.startswith("paper-"):
                broker_account_id = final_run.get("broker_account_id")
                if isinstance(broker_account_id, str):
                    creds = asyncio.run(_load_broker_credentials(UUID(broker_account_id)))
                else:
                    creds = None
                if creds:
                    api_key = creds.get("APCA-API-KEY-ID") or creds.get("api_key")
                    api_secret = creds.get("APCA-API-SECRET-KEY") or creds.get("api_secret")
                    base_url = creds.get("trading_base_url") or settings.alpaca_paper_trading_base_url
                    if api_key and api_secret:
                        try:
                            resp = requests.get(
                                f"{base_url.rstrip('/')}/v2/orders/{provider_order_id}",
                                headers={
                                    "APCA-API-KEY-ID": api_key,
                                    "APCA-API-SECRET-KEY": api_secret,
                                },
                                timeout=20,
                            )
                            order_evidence["provider_lookup_status"] = resp.status_code
                            if resp.status_code == 200:
                                payload = resp.json()
                                order_evidence["provider_lookup_payload"] = {
                                    "id": payload.get("id"),
                                    "status": payload.get("status"),
                                    "symbol": payload.get("symbol"),
                                    "side": payload.get("side"),
                                    "filled_qty": payload.get("filled_qty"),
                                }
                                order_evidence["provider_order_verified"] = str(payload.get("id")) == provider_order_id
                        except Exception as exc:  # noqa: BLE001
                            order_evidence["provider_lookup_error"] = f"{type(exc).__name__}: {exc}"

    latest_before = base_pnl[0] if base_pnl else {}
    latest_after = final_pnl[0] if final_pnl else {}
    account_update = {
        "positions_before": len(base_positions),
        "positions_after": len(final_positions),
        "position_delta": len(final_positions) - len(base_positions),
        "pnl_snapshots_before": len(base_pnl),
        "pnl_snapshots_after": len(final_pnl),
        "pnl_snapshot_delta": len(final_pnl) - len(base_pnl),
        "equity_before": _safe_float(latest_before.get("equity")) if isinstance(latest_before, dict) else None,
        "equity_after": _safe_float(latest_after.get("equity")) if isinstance(latest_after, dict) else None,
        "cash_before": _safe_float(latest_before.get("cash")) if isinstance(latest_before, dict) else None,
        "cash_after": _safe_float(latest_after.get("cash")) if isinstance(latest_after, dict) else None,
        "unrealized_before": _safe_float(latest_before.get("unrealized_pnl")) if isinstance(latest_before, dict) else None,
        "unrealized_after": _safe_float(latest_after.get("unrealized_pnl")) if isinstance(latest_after, dict) else None,
        "realized_before": _safe_float(latest_before.get("realized_pnl")) if isinstance(latest_before, dict) else None,
        "realized_after": _safe_float(latest_after.get("realized_pnl")) if isinstance(latest_after, dict) else None,
    }

    findings: list[str] = []
    if not celery_evidence["changed"]:
        findings.append("scheduler last_enqueued_at did not move during wait window")
    if market_data_evidence["signal_delta"] <= 0:
        findings.append("signal_events did not increase; market-data/runtime cycle evidence weak")
    if order_evidence["order_delta"] <= 0:
        findings.append("no new orders were created during runtime window")
    if not order_evidence["provider_order_verified"]:
        findings.append("could not prove order existence on Alpaca with provider_order_id lookup")
    if account_update["pnl_snapshot_delta"] <= 0:
        findings.append("pnl snapshot did not update during runtime window")

    ok = not findings
    return StepTwoResult(
        ok=ok,
        deployment_id=deployment_id,
        deployment_status_final=final_status,
        celery_scheduler_evidence=celery_evidence,
        market_data_evidence=market_data_evidence,
        order_evidence=order_evidence,
        account_update_evidence=account_update,
        findings=findings,
    )


def run_step_three(
    client: ApiClient,
    *,
    step_one: StepOneResult,
    notification_since_at: datetime,
    backtest_timeout_seconds: int,
) -> StepThreeResult:
    if not step_one.deployment_id:
        return StepThreeResult(
            ok=False,
            backtest_job={"ok": False, "reason": "missing_deployment"},
            notification_events={},
            telegram_delivery={},
            findings=["step1 did not produce deployment_id"],
        )

    deployment = client.get_deployment(step_one.deployment_id)
    strategy_id_raw = deployment.get("strategy_id")
    user_id_raw = deployment.get("user_id")
    market = deployment.get("market") if isinstance(deployment.get("market"), str) else "crypto"
    symbols = deployment.get("symbols") if isinstance(deployment.get("symbols"), list) else []
    symbol = str(symbols[0]) if symbols else "BTCUSD"
    timeframe = deployment.get("timeframe") if isinstance(deployment.get("timeframe"), str) else "1m"

    if not isinstance(strategy_id_raw, str) or not isinstance(user_id_raw, str):
        return StepThreeResult(
            ok=False,
            backtest_job={"ok": False, "reason": "missing_strategy_or_user"},
            notification_events={},
            telegram_delivery={},
            findings=["deployment payload missing strategy_id/user_id"],
        )

    backtest = asyncio.run(
        _trigger_backtest_completion(
            strategy_id=UUID(strategy_id_raw),
            user_id=UUID(user_id_raw),
            market=market,
            symbol=symbol,
            timeframe=timeframe,
            timeout_seconds=backtest_timeout_seconds,
        )
    )

    # Give notifications worker a small window to pick and dispatch.
    time.sleep(12)

    notif_state = asyncio.run(
        _query_notification_state(user_id=UUID(user_id_raw), since_at=notification_since_at)
    )

    events = notif_state.get("events") if isinstance(notif_state.get("events"), list) else []
    attempts = notif_state.get("attempts") if isinstance(notif_state.get("attempts"), list) else []

    target_deployment_id = step_one.deployment_id
    deployment_events = [
        row
        for row in events
        if isinstance(row, dict)
        and isinstance(row.get("payload"), dict)
        and str(row["payload"].get("deployment_id", "")) == target_deployment_id
    ]

    event_types = {str(row.get("event_type", "")).upper() for row in deployment_events}
    if isinstance(backtest.get("job_id"), str):
        for row in events:
            payload = row.get("payload") if isinstance(row, dict) else None
            if isinstance(payload, dict) and str(payload.get("job_id", "")) == str(backtest["job_id"]):
                event_types.add(str(row.get("event_type", "")).upper())

    successful_attempts = [row for row in attempts if isinstance(row, dict) and row.get("success") is True]
    failed_attempts = [row for row in attempts if isinstance(row, dict) and row.get("success") is False]

    telegram_delivery = {
        "binding": notif_state.get("telegram_binding"),
        "attempt_total": len(attempts),
        "attempt_success": len(successful_attempts),
        "attempt_failed": len(failed_attempts),
        "failed_error_codes": sorted(
            {
                str(row.get("error_code"))
                for row in failed_attempts
                if isinstance(row.get("error_code"), str)
            }
        ),
    }

    notification_events = {
        "event_type_counts": notif_state.get("event_type_counts"),
        "target_deployment_event_types": sorted(event_types),
        "target_deployment_event_count": len(deployment_events),
        "outbox_rows_total": len(events),
    }

    findings: list[str] = []
    required = {"DEPLOYMENT_STARTED", "POSITION_OPENED"}
    missing_required = sorted(required - event_types)
    if missing_required:
        findings.append(f"missing expected notification events for this chain: {missing_required}")
    if backtest.get("ok") and "BACKTEST_COMPLETED" not in event_types:
        findings.append("backtest completed but BACKTEST_COMPLETED event not found in outbox window")
    if telegram_delivery["attempt_success"] <= 0:
        findings.append("no successful Telegram delivery attempts in this window")

    ok = not findings
    return StepThreeResult(
        ok=ok,
        backtest_job=backtest,
        notification_events=notification_events,
        telegram_delivery=telegram_delivery,
        findings=findings,
    )


def build_report(
    *,
    started_at: str,
    login_payload: dict[str, Any],
    email: str,
    step1: StepOneResult,
    step2: StepTwoResult,
    step3: StepThreeResult,
) -> dict[str, Any]:
    return {
        "started_at": started_at,
        "finished_at": _now_iso(),
        "runtime": {
            "base_url": settings.api_v1_prefix,
            "openai_model": settings.openai_response_model,
            "strategy_mcp": settings.strategy_mcp_server_url,
            "backtest_mcp": settings.backtest_mcp_server_url,
            "market_mcp": settings.market_data_mcp_server_url,
            "trading_mcp": settings.trading_mcp_server_url,
            "paper_trading_enabled": settings.paper_trading_enabled,
            "paper_trading_execute_orders": settings.paper_trading_execute_orders,
            "notifications_enabled": settings.notifications_enabled,
            "telegram_enabled": settings.telegram_enabled,
            "trading_approval_enabled": settings.trading_approval_enabled,
        },
        "user": {
            "user_id": login_payload.get("user_id"),
            "email": email,
            "kyc_status": login_payload.get("user", {}).get("kyc_status"),
        },
        "step1_strategy_to_deployment": {
            "ok": step1.ok,
            "session_id": step1.session_id,
            "phase_start": step1.phase_start,
            "phase_end": step1.phase_end,
            "strategy_to_deployment_transition": step1.strategy_to_deployment_transition,
            "transition_trigger": step1.transition_trigger,
            "deployment_id": step1.deployment_id,
            "deployment_status": step1.deployment_status,
            "used_mcp_tools": step1.used_mcp_tools,
            "prompt_tool_sync": step1.prompt_tool_sync,
            "findings": step1.findings,
            "turn_logs": step1.turn_logs,
        },
        "step2_paper_trading": {
            "ok": step2.ok,
            "deployment_id": step2.deployment_id,
            "deployment_status_final": step2.deployment_status_final,
            "celery_scheduler_evidence": step2.celery_scheduler_evidence,
            "market_data_evidence": step2.market_data_evidence,
            "order_evidence": step2.order_evidence,
            "account_update_evidence": step2.account_update_evidence,
            "findings": step2.findings,
        },
        "step3_notifications": {
            "ok": step3.ok,
            "backtest_job": step3.backtest_job,
            "notification_events": step3.notification_events,
            "telegram_delivery": step3.telegram_delivery,
            "findings": step3.findings,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit backend live chain with frontend-like calls.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/api/v1")
    parser.add_argument("--email", default="2@test.com")
    parser.add_argument("--password", default="123456")
    parser.add_argument("--language", default="zh")
    parser.add_argument("--runtime-wait-seconds", type=int, default=120)
    parser.add_argument("--backtest-timeout-seconds", type=int, default=180)
    parser.add_argument(
        "--json-report",
        default="logs/audit_backend_live_chain_report.json",
        help="Path to write JSON report",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started_at = _now_iso()

    client = ApiClient(
        base_url=args.base_url,
        email=args.email,
        password=args.password,
        language=args.language,
    )

    login_payload = client.login()
    notification_since_at = datetime.now(UTC)

    step1 = run_step_one(client)
    step2 = run_step_two(
        client,
        step_one=step1,
        runtime_wait_seconds=max(30, int(args.runtime_wait_seconds)),
    )
    step3 = run_step_three(
        client,
        step_one=step1,
        notification_since_at=notification_since_at,
        backtest_timeout_seconds=max(60, int(args.backtest_timeout_seconds)),
    )

    report = build_report(
        started_at=started_at,
        login_payload=login_payload,
        email=args.email,
        step1=step1,
        step2=step2,
        step3=step3,
    )

    out_path = Path(args.json_report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))

    if not (step1.ok and step2.ok and step3.ok):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
