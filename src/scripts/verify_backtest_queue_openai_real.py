#!/usr/bin/env python3
"""Real-world verification for backtest queue + OpenAI polling behavior.

This script runs three checks against real infrastructure:
1) Queue pressure test: concurrent submit + polling to terminal status.
2) Real OpenAI tool-use test on backtest_create_job/backtest_get_job.
3) Poll-interval behavior test: can the model self-insert sleep without a sleep tool.

By default it starts local MCP and Celery worker processes, then executes tests.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import statistics
import subprocess
import sys
import time
from collections import Counter
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from uuid import uuid4

import httpx
from openai import APIError, OpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.engine.strategy import EXAMPLE_PATH, load_strategy_payload, upsert_strategy_dsl
from src.models import database as db_module
from src.models.session import Session as AgentSession
from src.models.user import User

DEFAULT_MODEL = settings.openai_response_model
DEFAULT_MCP_HOST = "127.0.0.1"
DEFAULT_MCP_PORT = 8111
DEFAULT_JSON_REPORT = "logs/backtest_queue_openai_real_report.json"


@dataclass
class QueueJobSubmission:
    strategy_id: str
    job_id: str
    submit_latency_ms: float
    submit_status: str
    submitted_at_monotonic: float


@dataclass
class QueueJobRuntime:
    job_id: str
    status: str
    progress: int
    current_step: str | None
    poll_count: int
    first_running_after_s: float | None
    terminal_after_s: float | None
    error: dict[str, Any] | None


@dataclass
class QueuePressureResult:
    ok: bool
    total_jobs: int
    completed_jobs: int
    failed_jobs: int
    timed_out_jobs: int
    submit_latency_ms_p50: float
    submit_latency_ms_p95: float
    first_running_after_s_p50: float | None
    first_running_after_s_p95: float | None
    terminal_after_s_p50: float | None
    terminal_after_s_p95: float | None
    jobs: list[QueueJobRuntime] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class OpenAICallRecord:
    tool: str
    status: str | None
    received_at_monotonic: float
    output_status: str | None
    output_error_code: str | None
    output_job_id: str | None


@dataclass
class OpenAICaseResult:
    name: str
    ok: bool
    reason: str
    response_id: str | None
    api_error: str | None
    event_counts: dict[str, int]
    called_tools: list[str]
    call_records: list[OpenAICallRecord]
    assistant_text: str
    get_job_intervals_s: list[float]
    asked_sleep_seconds: float | None
    observed_sleep_seconds_min: float | None

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["call_records"] = [asdict(item) for item in self.call_records]
        return payload


@dataclass
class Report:
    ok: bool
    timestamp_utc: str
    model: str
    server_url: str
    openai_server_url: str
    queue_pressure: dict[str, Any]
    openai_cases: list[dict[str, Any]]
    summary: dict[str, Any]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run real queue pressure + OpenAI MCP behavior verification for backtest flow.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--mcp-host", default=DEFAULT_MCP_HOST)
    parser.add_argument("--mcp-port", type=int, default=DEFAULT_MCP_PORT)
    parser.add_argument(
        "--mcp-server-url",
        default="",
        help="Optional MCP URL. If empty, use local host/port.",
    )
    parser.add_argument(
        "--openai-mcp-server-url",
        default="",
        help="Optional public MCP URL for OpenAI calls.",
    )
    parser.add_argument(
        "--start-local-mcp",
        action="store_true",
        help="Start local MCP server subprocess automatically.",
    )
    parser.add_argument(
        "--start-local-worker",
        action="store_true",
        help="Start local Celery worker subprocess automatically.",
    )
    parser.add_argument(
        "--start-ngrok",
        action="store_true",
        help="Start ngrok to expose local MCP for OpenAI calls.",
    )
    parser.add_argument("--ngrok-timeout-seconds", type=float, default=25.0)
    parser.add_argument("--worker-concurrency", type=int, default=1)
    parser.add_argument("--pressure-jobs", type=int, default=8)
    parser.add_argument("--pressure-submit-concurrency", type=int, default=4)
    parser.add_argument("--poll-interval-seconds", type=float, default=1.0)
    parser.add_argument("--pressure-timeout-seconds", type=float, default=240.0)
    parser.add_argument("--ai-blocker-jobs", type=int, default=3)
    parser.add_argument("--openai-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--startup-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--json-report", default=DEFAULT_JSON_REPORT)
    parser.add_argument("--always-zero", action="store_true")
    parser.add_argument(
        "--skip-openai",
        action="store_true",
        help="Skip real OpenAI cases (run queue pressure only).",
    )
    return parser.parse_args()


def _dump_model(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json", exclude_none=True)
    if isinstance(obj, dict):
        return obj
    return {"value": str(obj)}


def _jsonrpc_messages(raw_text: str) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for raw_line in raw_text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped.startswith("data:"):
            stripped = stripped.split("data:", 1)[1].strip()
        if not stripped:
            continue
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            messages.append(parsed)
    return messages


async def _rpc_call(
    client: httpx.AsyncClient,
    *,
    server_url: str,
    method: str,
    params: dict[str, Any],
    session_id: str | None,
    timeout_seconds: float,
) -> tuple[dict[str, Any], str | None]:
    request_id = str(uuid4())
    headers = {
        "accept": "application/json, text/event-stream",
        "content-type": "application/json",
    }
    if session_id:
        headers["mcp-session-id"] = session_id

    payload = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": params,
    }
    response = await client.post(
        server_url,
        headers=headers,
        json=payload,
        timeout=max(5.0, timeout_seconds),
    )
    response.raise_for_status()
    session_id_resp = response.headers.get("mcp-session-id") or session_id

    for message in _jsonrpc_messages(response.text):
        if message.get("id") != request_id:
            continue
        if "result" in message and isinstance(message["result"], dict):
            return message["result"], session_id_resp
        if "error" in message:
            raise RuntimeError(f"RPC {method} error: {message['error']}")
    raise RuntimeError(f"RPC {method} missing result")


def _parse_tool_payload(result: dict[str, Any]) -> dict[str, Any]:
    content = result.get("content")
    if not isinstance(content, list):
        raise RuntimeError(f"Unexpected tool result shape: {result}")
    for item in content:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
    raise RuntimeError(f"Tool result has no parsable JSON payload: {result}")


async def _initialize_mcp_session(
    client: httpx.AsyncClient,
    *,
    server_url: str,
    timeout_seconds: float,
) -> str | None:
    init_result, session_id = await _rpc_call(
        client,
        server_url=server_url,
        method="initialize",
        params={
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "verify-backtest-queue-openai-real", "version": "0.1.0"},
        },
        session_id=None,
        timeout_seconds=timeout_seconds,
    )
    if "protocolVersion" not in init_result:
        raise RuntimeError(f"Invalid initialize result: {init_result}")
    return session_id


async def _call_mcp_tool(
    client: httpx.AsyncClient,
    *,
    server_url: str,
    session_id: str | None,
    name: str,
    arguments: dict[str, Any],
    timeout_seconds: float,
) -> tuple[dict[str, Any], str | None]:
    result, next_session_id = await _rpc_call(
        client,
        server_url=server_url,
        method="tools/call",
        params={
            "name": name,
            "arguments": arguments,
        },
        session_id=session_id,
        timeout_seconds=timeout_seconds,
    )
    payload = _parse_tool_payload(result)
    return payload, next_session_id or session_id


async def _create_backtest_strategy_ids(count: int) -> list[str]:
    if count <= 0:
        return []

    await db_module.close_postgres()
    await db_module.init_postgres()
    assert db_module.AsyncSessionLocal is not None

    strategy_ids: list[str] = []
    async with db_module.AsyncSessionLocal() as db:
        strategy_ids = await _insert_strategies(db, count=count)
        await db.commit()
    await db_module.close_postgres()
    return strategy_ids


async def _insert_strategies(db: AsyncSession, *, count: int) -> list[str]:
    email = f"queue_openai_real_{uuid4().hex[:10]}@example.com"
    user = User(email=email, password_hash="hash", name=email)
    db.add(user)
    await db.flush()

    session = AgentSession(
        user_id=user.id,
        current_phase="strategy",
        status="active",
        artifacts={},
        metadata_={},
    )
    db.add(session)
    await db.flush()

    payload_template = load_strategy_payload(EXAMPLE_PATH)
    payload_template["universe"]["market"] = "crypto"
    payload_template["universe"]["tickers"] = ["BTCUSD"]

    strategy_ids: list[str] = []
    for idx in range(count):
        payload = json.loads(json.dumps(payload_template))
        payload["strategy"]["name"] = f"queue_openai_real_{idx + 1}"
        created = await upsert_strategy_dsl(db, session_id=session.id, dsl_payload=payload)
        strategy_ids.append(str(created.strategy.id))
    return strategy_ids


async def _submit_one_job(
    *,
    strategy_id: str,
    server_url: str,
    timeout_seconds: float,
) -> QueueJobSubmission:
    async with httpx.AsyncClient(trust_env=False) as client:
        session_id = await _initialize_mcp_session(
            client,
            server_url=server_url,
            timeout_seconds=timeout_seconds,
        )
        started = time.perf_counter()
        payload, _ = await _call_mcp_tool(
            client,
            server_url=server_url,
            session_id=session_id,
            name="backtest_create_job",
            arguments={"strategy_id": strategy_id, "run_now": False},
            timeout_seconds=timeout_seconds,
        )
        ended = time.perf_counter()

    job_id = str(payload.get("job_id", ""))
    if not job_id:
        raise RuntimeError(f"Missing job_id in payload: {payload}")
    return QueueJobSubmission(
        strategy_id=strategy_id,
        job_id=job_id,
        submit_latency_ms=(ended - started) * 1000.0,
        submit_status=str(payload.get("status", "")),
        submitted_at_monotonic=ended,
    )


async def _submit_jobs_with_concurrency(
    *,
    strategy_ids: list[str],
    server_url: str,
    timeout_seconds: float,
    concurrency: int,
) -> list[QueueJobSubmission]:
    if not strategy_ids:
        return []

    # Warm up one call first so MCP-side DB initialization is finished before
    # concurrent submits (avoids racey schema-init deadlocks).
    first = await _submit_one_job(
        strategy_id=strategy_ids[0],
        server_url=server_url,
        timeout_seconds=timeout_seconds,
    )

    remaining = strategy_ids[1:]
    if not remaining:
        return [first]

    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def _guarded_submit(strategy_id: str) -> QueueJobSubmission:
        async with semaphore:
            return await _submit_one_job(
                strategy_id=strategy_id,
                server_url=server_url,
                timeout_seconds=timeout_seconds,
            )

    tasks = [_guarded_submit(strategy_id) for strategy_id in remaining]
    rest = await asyncio.gather(*tasks)
    return [first, *rest]


async def _poll_jobs_until_terminal(
    *,
    submissions: list[QueueJobSubmission],
    server_url: str,
    timeout_seconds: float,
    poll_interval_seconds: float,
) -> tuple[list[QueueJobRuntime], list[str]]:
    if not submissions:
        return ([], [])

    start_monotonic = time.perf_counter()
    poll_state: dict[str, dict[str, Any]] = {}
    for item in submissions:
        poll_state[item.job_id] = {
            "submitted_at": item.submitted_at_monotonic,
            "first_running_after_s": None,
            "terminal_after_s": None,
            "poll_count": 0,
            "status": item.submit_status or "pending",
            "progress": 0,
            "current_step": None,
            "error": None,
            "terminal": item.submit_status in {"done", "failed"},
        }

    async with httpx.AsyncClient(trust_env=False) as client:
        session_id = await _initialize_mcp_session(
            client,
            server_url=server_url,
            timeout_seconds=timeout_seconds,
        )

        while True:
            pending_job_ids = [
                job_id
                for job_id, state in poll_state.items()
                if not bool(state["terminal"])
            ]
            if not pending_job_ids:
                break

            elapsed = time.perf_counter() - start_monotonic
            if elapsed > timeout_seconds:
                break

            poll_tasks = [
                _call_mcp_tool(
                    client,
                    server_url=server_url,
                    session_id=session_id,
                    name="backtest_get_job",
                    arguments={"job_id": job_id},
                    timeout_seconds=timeout_seconds,
                )
                for job_id in pending_job_ids
            ]
            poll_results = await asyncio.gather(*poll_tasks, return_exceptions=True)

            for job_id, result in zip(pending_job_ids, poll_results, strict=True):
                state = poll_state[job_id]
                state["poll_count"] = int(state["poll_count"]) + 1
                if isinstance(result, Exception):
                    state["status"] = "failed"
                    state["terminal"] = True
                    state["error"] = {
                        "code": "POLL_EXCEPTION",
                        "message": f"{type(result).__name__}: {result}",
                    }
                    state["terminal_after_s"] = time.perf_counter() - float(state["submitted_at"])
                    continue

                payload, maybe_session_id = result
                session_id = maybe_session_id
                status = str(payload.get("status", "")).strip().lower() or "failed"
                state["status"] = status
                state["progress"] = int(payload.get("progress", 0) or 0)
                state["current_step"] = payload.get("current_step")
                state["error"] = payload.get("error")

                now = time.perf_counter()
                if status == "running" and state["first_running_after_s"] is None:
                    state["first_running_after_s"] = now - float(state["submitted_at"])

                if status in {"done", "failed"}:
                    state["terminal"] = True
                    state["terminal_after_s"] = now - float(state["submitted_at"])

            await asyncio.sleep(max(0.1, poll_interval_seconds))

    runtimes: list[QueueJobRuntime] = []
    errors: list[str] = []
    for job_id, state in poll_state.items():
        if not bool(state["terminal"]):
            errors.append(f"job {job_id} timed out before terminal state")
        runtimes.append(
            QueueJobRuntime(
                job_id=job_id,
                status=str(state["status"]),
                progress=int(state["progress"]),
                current_step=(
                    str(state["current_step"]) if isinstance(state["current_step"], str) else None
                ),
                poll_count=int(state["poll_count"]),
                first_running_after_s=(
                    float(state["first_running_after_s"])
                    if isinstance(state["first_running_after_s"], float)
                    else None
                ),
                terminal_after_s=(
                    float(state["terminal_after_s"])
                    if isinstance(state["terminal_after_s"], float)
                    else None
                ),
                error=state["error"] if isinstance(state["error"], dict) else None,
            )
        )
    return (runtimes, errors)


def _p50(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.median(values))


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.95))))
    return float(ordered[idx])


async def run_queue_pressure_test(
    *,
    server_url: str,
    total_jobs: int,
    submit_concurrency: int,
    poll_interval_seconds: float,
    timeout_seconds: float,
) -> QueuePressureResult:
    strategy_ids = await _create_backtest_strategy_ids(total_jobs)
    submissions = await _submit_jobs_with_concurrency(
        strategy_ids=strategy_ids,
        server_url=server_url,
        timeout_seconds=timeout_seconds,
        concurrency=submit_concurrency,
    )
    runtimes, poll_errors = await _poll_jobs_until_terminal(
        submissions=submissions,
        server_url=server_url,
        timeout_seconds=timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
    )

    completed = [item for item in runtimes if item.status == "done"]
    failed = [item for item in runtimes if item.status == "failed"]
    timed_out = [item for item in runtimes if item.status not in {"done", "failed"}]

    submit_latencies = [item.submit_latency_ms for item in submissions]
    running_latencies = [
        item.first_running_after_s
        for item in runtimes
        if isinstance(item.first_running_after_s, float)
    ]
    terminal_latencies = [
        item.terminal_after_s
        for item in runtimes
        if isinstance(item.terminal_after_s, float)
    ]
    errors: list[str] = list(poll_errors)
    for item in failed:
        if isinstance(item.error, dict):
            code = str(item.error.get("code", "UNKNOWN"))
            message = str(item.error.get("message", ""))
            errors.append(f"job {item.job_id} failed with {code}: {message}")

    ok = len(completed) == len(runtimes) and not errors
    return QueuePressureResult(
        ok=ok,
        total_jobs=len(runtimes),
        completed_jobs=len(completed),
        failed_jobs=len(failed),
        timed_out_jobs=len(timed_out),
        submit_latency_ms_p50=round(_p50(submit_latencies), 3),
        submit_latency_ms_p95=round(_p95(submit_latencies), 3),
        first_running_after_s_p50=(
            round(_p50(running_latencies), 3) if running_latencies else None
        ),
        first_running_after_s_p95=(
            round(_p95(running_latencies), 3) if running_latencies else None
        ),
        terminal_after_s_p50=(
            round(_p50(terminal_latencies), 3) if terminal_latencies else None
        ),
        terminal_after_s_p95=(
            round(_p95(terminal_latencies), 3) if terminal_latencies else None
        ),
        jobs=runtimes,
        errors=errors,
    )


def _extract_backtest_output_fields(raw_output: Any) -> tuple[str | None, str | None, str | None]:
    if not isinstance(raw_output, str):
        return (None, None, None)
    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError:
        return (None, None, None)
    if not isinstance(parsed, dict):
        return (None, None, None)
    output_status = parsed.get("status")
    output_job_id = parsed.get("job_id")
    error_code: str | None = None
    raw_error = parsed.get("error")
    if isinstance(raw_error, dict):
        maybe_code = raw_error.get("code")
        if isinstance(maybe_code, str):
            error_code = maybe_code
    return (
        str(output_status) if isinstance(output_status, str) else None,
        error_code,
        str(output_job_id) if isinstance(output_job_id, str) else None,
    )


def run_openai_case(
    client: OpenAI,
    *,
    model: str,
    server_url: str,
    strategy_id: str,
    timeout_seconds: float,
    name: str,
    prompt: str,
    asked_sleep_seconds: float | None,
) -> OpenAICaseResult:
    event_counter: Counter[str] = Counter()
    call_records: list[OpenAICallRecord] = []
    called_tools: list[str] = []
    assistant_text_parts: list[str] = []
    response_id: str | None = None
    api_error: str | None = None

    try:
        with client.responses.stream(
            model=model,
            input=prompt.format(strategy_id=strategy_id),
            tools=[
                {
                    "type": "mcp",
                    "server_label": "backtest",
                    "server_url": server_url,
                    "allowed_tools": ["backtest_create_job", "backtest_get_job"],
                    "require_approval": "never",
                }
            ],
            timeout=timeout_seconds,
        ) as stream:
            for event in stream:
                event_type = getattr(event, "type", "unknown")
                event_counter[event_type] += 1

                if event_type == "response.output_text.delta":
                    delta = getattr(event, "delta", "")
                    if isinstance(delta, str) and delta:
                        assistant_text_parts.append(delta)
                    continue

                if event_type != "response.output_item.added":
                    continue

                payload = _dump_model(event)
                item = payload.get("item")
                if not isinstance(item, dict):
                    continue
                if item.get("type") != "mcp_call":
                    continue

                tool_name = item.get("name")
                tool_status = item.get("status")
                output_status, error_code, output_job_id = _extract_backtest_output_fields(
                    item.get("output")
                )
                if isinstance(tool_name, str):
                    called_tools.append(tool_name)
                    call_records.append(
                        OpenAICallRecord(
                            tool=tool_name,
                            status=(str(tool_status) if isinstance(tool_status, str) else None),
                            received_at_monotonic=time.monotonic(),
                            output_status=output_status,
                            output_error_code=error_code,
                            output_job_id=output_job_id,
                        )
                    )

            final_response = stream.get_final_response()
            response_id = final_response.id
            for output_item in final_response.output or []:
                item = _dump_model(output_item)
                if item.get("type") != "mcp_call":
                    continue
                tool_name = item.get("name")
                if isinstance(tool_name, str):
                    called_tools.append(tool_name)
                    output_status, error_code, output_job_id = _extract_backtest_output_fields(
                        item.get("output")
                    )
                    call_records.append(
                        OpenAICallRecord(
                            tool=tool_name,
                            status=(str(item.get("status")) if isinstance(item.get("status"), str) else None),
                            received_at_monotonic=time.monotonic(),
                            output_status=output_status,
                            output_error_code=error_code,
                            output_job_id=output_job_id,
                        )
                    )

    except APIError as exc:
        api_error = f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # noqa: BLE001
        api_error = f"{type(exc).__name__}: {exc}"

    get_job_times = [
        record.received_at_monotonic
        for record in call_records
        if record.tool == "backtest_get_job" and record.status == "in_progress"
    ]
    intervals: list[float] = []
    if len(get_job_times) >= 2:
        for left, right in zip(get_job_times[:-1], get_job_times[1:], strict=True):
            intervals.append(round(right - left, 3))

    observed_sleep_min = min(intervals) if intervals else None
    has_create = "backtest_create_job" in called_tools
    has_get = "backtest_get_job" in called_tools
    has_done_or_failed = any(
        (item.output_status in {"done", "failed"})
        for item in call_records
        if item.tool in {"backtest_create_job", "backtest_get_job"}
    )

    ok = (
        api_error is None
        and has_create
        and has_get
        and has_done_or_failed
    )
    reason_parts: list[str] = []
    if not has_create:
        reason_parts.append("missing backtest_create_job call")
    if not has_get:
        reason_parts.append("missing backtest_get_job call")
    if not has_done_or_failed:
        reason_parts.append("no terminal status observed in tool outputs")
    if api_error:
        reason_parts.append(api_error)
    reason = "ok" if not reason_parts else "; ".join(reason_parts)

    return OpenAICaseResult(
        name=name,
        ok=ok,
        reason=reason,
        response_id=response_id,
        api_error=api_error,
        event_counts=dict(event_counter),
        called_tools=called_tools,
        call_records=call_records,
        assistant_text="".join(assistant_text_parts).strip(),
        get_job_intervals_s=intervals,
        asked_sleep_seconds=asked_sleep_seconds,
        observed_sleep_seconds_min=observed_sleep_min,
    )


def _start_process(
    cmd: list[str],
    *,
    cwd: str,
    stdout: Any = None,
    stderr: Any = None,
) -> subprocess.Popen[str]:
    return subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=stdout if stdout is not None else subprocess.DEVNULL,
        stderr=stderr if stderr is not None else subprocess.DEVNULL,
        text=True,
    )


def _wait_for_port(host: str, port: int, timeout_seconds: float) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            with socket_connection(host, port, timeout=1.0):
                return
        except OSError:
            time.sleep(0.4)
    raise RuntimeError(f"Timed out waiting for {host}:{port}")


def socket_connection(host: str, port: int, timeout: float):
    import socket

    sock = socket.create_connection((host, port), timeout=timeout)
    return sock


def _matches_expected_port(*, addr: str, mcp_port: int) -> bool:
    normalized = addr.strip()
    if not normalized:
        return False
    return re.search(rf":{mcp_port}(?:/|$)", normalized) is not None


def _get_ngrok_public_url(
    *,
    timeout_seconds: float,
    mcp_port: int,
    ngrok_proc: subprocess.Popen[Any] | None = None,
) -> str:
    deadline = time.time() + timeout_seconds
    api_ports = [4040, 4041, 4042, 4043, 4044]
    while time.time() < deadline:
        for port in api_ports:
            try:
                response = httpx.get(
                    f"http://127.0.0.1:{port}/api/tunnels",
                    timeout=2.0,
                    trust_env=False,
                )
                response.raise_for_status()
                payload = response.json()
                tunnels = payload.get("tunnels", [])
                for tunnel in tunnels:
                    if not isinstance(tunnel, dict):
                        continue
                    public_url = tunnel.get("public_url")
                    config = tunnel.get("config")
                    addr = ""
                    if isinstance(config, dict):
                        raw_addr = config.get("addr")
                        if isinstance(raw_addr, str):
                            addr = raw_addr
                    if (
                        isinstance(public_url, str)
                        and public_url.startswith("https://")
                        and _matches_expected_port(addr=addr, mcp_port=mcp_port)
                    ):
                        return public_url
            except Exception:  # noqa: BLE001
                continue
        time.sleep(0.5)

    details = ""
    if ngrok_proc is not None and ngrok_proc.poll() is not None:
        try:
            _, stderr = ngrok_proc.communicate(timeout=1.0)
        except Exception:  # noqa: BLE001
            stderr = ""
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        if isinstance(stderr, str) and stderr.strip():
            details = f" ngrok_stderr={stderr.strip()}"
    raise RuntimeError(f"ngrok public URL not ready within timeout.{details}")


def _tail_lines(path: str, *, line_count: int = 80) -> str:
    try:
        raw = Path(path).read_text(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return ""
    lines = raw.splitlines()
    return "\n".join(lines[-line_count:])


async def _enqueue_blocker_jobs(
    *,
    server_url: str,
    strategy_id: str,
    count: int,
) -> list[str]:
    if count <= 0:
        return []
    submissions = await _submit_jobs_with_concurrency(
        strategy_ids=[strategy_id for _ in range(count)],
        server_url=server_url,
        timeout_seconds=60.0,
        concurrency=min(4, count),
    )
    return [item.job_id for item in submissions]


def _print_queue_result(result: QueuePressureResult) -> None:
    print("=== Queue Pressure Test ===")
    print(
        f"ok={result.ok} total={result.total_jobs} completed={result.completed_jobs} "
        f"failed={result.failed_jobs} timeout={result.timed_out_jobs}"
    )
    print(
        f"submit_latency_ms p50={result.submit_latency_ms_p50} "
        f"p95={result.submit_latency_ms_p95}"
    )
    print(
        f"first_running_after_s p50={result.first_running_after_s_p50} "
        f"p95={result.first_running_after_s_p95}"
    )
    print(
        f"terminal_after_s p50={result.terminal_after_s_p50} "
        f"p95={result.terminal_after_s_p95}"
    )
    if result.errors:
        print("errors:")
        for error in result.errors:
            print(f"  - {error}")
    print()


def _print_openai_case(result: OpenAICaseResult) -> None:
    print(f"=== OpenAI Case: {result.name} ===")
    print(f"ok={result.ok} reason={result.reason}")
    print(f"called_tools={result.called_tools}")
    print(f"get_job_intervals_s={result.get_job_intervals_s}")
    print(f"observed_sleep_seconds_min={result.observed_sleep_seconds_min}")
    if result.api_error:
        print(f"api_error={result.api_error}")
    print()


def main() -> int:
    args = _parse_args()
    server_url = args.mcp_server_url.strip() or f"http://{args.mcp_host}:{args.mcp_port}/mcp"
    openai_server_url = args.openai_mcp_server_url.strip()
    report_path = Path(args.json_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    if not (os.getenv("OPENAI_API_KEY") or settings.openai_api_key):
        print("OPENAI_API_KEY is missing.", file=sys.stderr)
        return 1

    mcp_proc: subprocess.Popen[str] | None = None
    worker_proc: subprocess.Popen[str] | None = None
    ngrok_proc: subprocess.Popen[str] | None = None
    worker_log = NamedTemporaryFile(prefix="queue_openai_worker_", suffix=".log", delete=False)
    mcp_log = NamedTemporaryFile(prefix="queue_openai_mcp_", suffix=".log", delete=False)
    ngrok_log = NamedTemporaryFile(prefix="queue_openai_ngrok_", suffix=".log", delete=False)
    overall_ok = True

    try:
        if args.start_local_worker:
            print("[stage] starting local celery worker ...", flush=True)
            worker_cmd = [
                sys.executable,
                "-m",
                "celery",
                "-A",
                "src.workers.celery_app:celery_app",
                "worker",
                "-Q",
                "backtest",
                "--concurrency",
                str(args.worker_concurrency),
                "--loglevel",
                "INFO",
            ]
            worker_proc = _start_process(
                worker_cmd,
                cwd=os.getcwd(),
                stdout=worker_log,
                stderr=worker_log,
            )

        if args.start_local_mcp:
            print("[stage] starting local mcp server ...", flush=True)
            mcp_cmd = [
                sys.executable,
                "-m",
                "src.mcp.server",
                "--transport",
                "streamable-http",
                "--host",
                args.mcp_host,
                "--port",
                str(args.mcp_port),
            ]
            mcp_proc = _start_process(
                mcp_cmd,
                cwd=os.getcwd(),
                stdout=mcp_log,
                stderr=mcp_log,
            )
            _wait_for_port(args.mcp_host, args.mcp_port, args.startup_timeout_seconds)
            print(f"[stage] mcp ready at {server_url}", flush=True)

        if not openai_server_url:
            if args.start_ngrok:
                print("[stage] starting ngrok tunnel for openai ...", flush=True)
                ngrok_cmd = ["ngrok", "http", str(args.mcp_port), "--pooling-enabled"]
                ngrok_proc = _start_process(
                    ngrok_cmd,
                    cwd=os.getcwd(),
                    stdout=ngrok_log,
                    stderr=ngrok_log,
                )
                public_url = _get_ngrok_public_url(
                    timeout_seconds=args.ngrok_timeout_seconds,
                    mcp_port=args.mcp_port,
                    ngrok_proc=ngrok_proc,
                )
                openai_server_url = f"{public_url}/mcp"
                print(f"[stage] ngrok public mcp url: {openai_server_url}", flush=True)
            else:
                openai_server_url = server_url

        print("[stage] running queue pressure test ...", flush=True)
        queue_result = asyncio.run(
            run_queue_pressure_test(
                server_url=server_url,
                total_jobs=args.pressure_jobs,
                submit_concurrency=args.pressure_submit_concurrency,
                poll_interval_seconds=args.poll_interval_seconds,
                timeout_seconds=args.pressure_timeout_seconds,
            )
        )
        _print_queue_result(queue_result)
        overall_ok = overall_ok and queue_result.ok

        openai_cases: list[OpenAICaseResult] = []
        sleep_tool_recommended: bool | None = None
        if not args.skip_openai:
            print("[stage] preparing strategy for openai tool-use tests ...", flush=True)
            strategy_for_ai = asyncio.run(_create_backtest_strategy_ids(1))[0]
            _ = asyncio.run(
                _enqueue_blocker_jobs(
                    server_url=server_url,
                    strategy_id=strategy_for_ai,
                    count=args.ai_blocker_jobs,
                )
            )

            print("[stage] running openai case #1 (baseline polling) ...", flush=True)
            openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY") or settings.openai_api_key)
            case_fast = run_openai_case(
                openai_client,
                model=args.model,
                server_url=openai_server_url,
                strategy_id=strategy_for_ai,
                timeout_seconds=args.openai_timeout_seconds,
                name="ai_poll_without_sleep_tool_baseline",
                prompt=(
                    "Use MCP tools to run a backtest for strategy_id {strategy_id}. "
                    "Call backtest_create_job first with run_now=false. "
                    "Then keep polling backtest_get_job until status becomes done or failed. "
                    "Do not ask follow-up questions."
                ),
                asked_sleep_seconds=None,
            )
            _print_openai_case(case_fast)
            overall_ok = overall_ok and case_fast.ok
            openai_cases.append(case_fast)

            _ = asyncio.run(
                _enqueue_blocker_jobs(
                    server_url=server_url,
                    strategy_id=strategy_for_ai,
                    count=args.ai_blocker_jobs,
                )
            )
            print(
                "[stage] running openai case #2 (asked 2s interval, no sleep tool) ...",
                flush=True,
            )
            case_sleep_ask = run_openai_case(
                openai_client,
                model=args.model,
                server_url=openai_server_url,
                strategy_id=strategy_for_ai,
                timeout_seconds=args.openai_timeout_seconds,
                name="ai_poll_ask_for_2s_sleep_without_sleep_tool",
                prompt=(
                    "Use MCP tools to run a backtest for strategy_id {strategy_id}. "
                    "Call backtest_create_job first with run_now=false. "
                    "Then poll backtest_get_job repeatedly until done or failed, "
                    "and wait about 2 seconds between polls. "
                    "Do not ask follow-up questions."
                ),
                asked_sleep_seconds=2.0,
            )
            _print_openai_case(case_sleep_ask)
            overall_ok = overall_ok and case_sleep_ask.ok
            openai_cases.append(case_sleep_ask)

            if case_sleep_ask.get_job_intervals_s:
                min_interval = min(case_sleep_ask.get_job_intervals_s)
                # If model cannot keep >=1.5s when asked for 2s waits, treat as needing a sleep tool.
                sleep_tool_recommended = min_interval < 1.5
            else:
                sleep_tool_recommended = True
        else:
            print("[stage] skip_openai=true, only queue pressure test executed.", flush=True)

        summary = {
            "overall_ok": overall_ok,
            "sleep_tool_recommended": sleep_tool_recommended,
            "sleep_tool_reason": (
                "OpenAI test skipped."
                if args.skip_openai
                else (
                    "Model did not consistently produce >=1.5s polling intervals without a sleep tool."
                    if bool(sleep_tool_recommended)
                    else "Model achieved usable polling intervals without a sleep tool in this run."
                )
            ),
        }

        report = Report(
            ok=overall_ok,
            timestamp_utc=datetime.now(UTC).isoformat(),
            model=args.model,
            server_url=server_url,
            openai_server_url=openai_server_url,
            queue_pressure=asdict(queue_result),
            openai_cases=[item.to_json() for item in openai_cases],
            summary=summary,
        )
        with report_path.open("w", encoding="utf-8") as fp:
            json.dump(asdict(report), fp, ensure_ascii=False, indent=2)
        print(f"[report] wrote {report_path}")
        print(f"[overall] ok={report.ok} sleep_tool_recommended={sleep_tool_recommended}")
        if not report.ok:
            print(f"[debug] worker_log={worker_log.name}")
            print(f"[debug] mcp_log={mcp_log.name}")
            if ngrok_proc is not None:
                print(f"[debug] ngrok_log={ngrok_log.name}")

        if args.always_zero:
            return 0
        return 0 if report.ok else 1
    finally:
        for fp in (worker_log, mcp_log, ngrok_log):
            with suppress(Exception):
                fp.flush()
            with suppress(Exception):
                fp.close()
        if mcp_proc is not None and mcp_proc.poll() is None:
            mcp_proc.terminate()
            try:
                mcp_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                mcp_proc.kill()
                mcp_proc.wait(timeout=5)
        if worker_proc is not None and worker_proc.poll() is None:
            worker_proc.terminate()
            try:
                worker_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                worker_proc.kill()
                worker_proc.wait(timeout=5)
        if ngrok_proc is not None and ngrok_proc.poll() is None:
            ngrok_proc.terminate()
            try:
                ngrok_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                ngrok_proc.kill()
                ngrok_proc.wait(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
