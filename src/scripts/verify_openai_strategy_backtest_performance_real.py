#!/usr/bin/env python3
"""Real E2E verification: AI strategy upsert -> backtest -> performance MCP tools via ngrok.

This script validates three layers:
1) Simulated test suite (pytest) for strategy/backtest/performance code.
2) Real OpenAI call for strategy submission through MCP (`strategy_upsert_dsl`).
3) Real OpenAI calls for backtest execution and all performance analytics tools.

Local MCP is exposed via ngrok so OpenAI can reach it without deployed VM MCP.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from uuid import uuid4

import httpx
from openai import APIError, OpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.engine.strategy import EXAMPLE_PATH, load_strategy_payload
from src.models import database as db_module
from src.models.session import Session as AgentSession
from src.models.user import User

DEFAULT_MODEL = settings.openai_response_model
DEFAULT_MCP_HOST = "127.0.0.1"
DEFAULT_MCP_PORT = 8111
DEFAULT_REPORT_PATH = "logs/openai_strategy_backtest_performance_real_report.json"

PERFORMANCE_TOOL_NAMES: tuple[str, ...] = (
    "backtest_entry_hour_pnl_heatmap",
    "backtest_entry_weekday_pnl",
    "backtest_monthly_return_table",
    "backtest_holding_period_pnl_bins",
    "backtest_long_short_breakdown",
    "backtest_exit_reason_breakdown",
    "backtest_underwater_curve",
    "backtest_rolling_metrics",
)


@dataclass
class ToolCallRecord:
    tool: str
    status: str | None
    output_ok: bool | None
    output_error_code: str | None
    output_job_id: str | None
    output_strategy_id: str | None


@dataclass
class OpenAICaseResult:
    name: str
    ok: bool
    reason: str
    response_id: str | None
    api_error: str | None
    called_tools: list[str]
    call_records: list[dict[str, Any]]
    parsed_outputs_by_tool: dict[str, list[dict[str, Any]]]
    event_counts: dict[str, int]
    assistant_text: str


@dataclass
class FullReport:
    ok: bool
    timestamp_utc: str
    model: str
    mcp_server_url: str
    openai_mcp_server_url: str
    simulated_tests_ok: bool
    simulated_test_command: str
    simulated_test_output_tail: str
    strategy_case: dict[str, Any]
    backtest_case: dict[str, Any]
    performance_cases: list[dict[str, Any]]
    summary: dict[str, Any]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify AI strategy/backtest/performance MCP flow with real OpenAI + ngrok.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--mcp-host", default=DEFAULT_MCP_HOST)
    parser.add_argument("--mcp-port", type=int, default=DEFAULT_MCP_PORT)
    parser.add_argument(
        "--mcp-server-url",
        default="",
        help="Optional local MCP URL. If empty, uses host/port.",
    )
    parser.add_argument(
        "--openai-mcp-server-url",
        default="",
        help="Optional public MCP URL for OpenAI. If empty and --start-ngrok is set, auto-detect via ngrok API.",
    )
    parser.add_argument("--openai-timeout-seconds", type=float, default=240.0)
    parser.add_argument("--startup-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--ngrok-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--poll-wait-seconds", type=float, default=1.2)
    parser.add_argument("--max-backtest-polls", type=int, default=90)
    parser.add_argument("--start-local-mcp", action="store_true")
    parser.add_argument("--start-local-worker", action="store_true")
    parser.add_argument("--start-ngrok", action="store_true")
    parser.add_argument(
        "--run-simulated-tests",
        action="store_true",
        help="Run local pytest suite before real OpenAI verification.",
    )
    parser.add_argument("--report", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--always-zero", action="store_true")
    return parser.parse_args()


def _dump_model(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json", exclude_none=True)
    if isinstance(obj, dict):
        return obj
    return {"value": str(obj)}


def _extract_json_payload_from_tool_output(raw_output: Any) -> dict[str, Any] | None:
    if not isinstance(raw_output, str):
        return None
    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _extract_tool_outcome(parsed_output: dict[str, Any] | None) -> tuple[bool | None, str | None, str | None, str | None]:
    if not isinstance(parsed_output, dict):
        return (None, None, None, None)
    ok = parsed_output.get("ok")
    output_ok = bool(ok) if isinstance(ok, bool) else None
    output_job_id = parsed_output.get("job_id")
    output_strategy_id = parsed_output.get("strategy_id")

    error_code: str | None = None
    raw_error = parsed_output.get("error")
    if isinstance(raw_error, dict):
        maybe = raw_error.get("code")
        if isinstance(maybe, str) and maybe.strip():
            error_code = maybe.strip()

    return (
        output_ok,
        error_code,
        output_job_id if isinstance(output_job_id, str) else None,
        output_strategy_id if isinstance(output_strategy_id, str) else None,
    )


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
    api_ports = [4040, 4041, 4042, 4043, 4044, 4045]
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
    import socket

    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        sock: socket.socket | None = None
        try:
            sock = socket.create_connection((host, port), timeout=1.0)
            return
        except OSError:
            time.sleep(0.4)
        finally:
            if sock is not None:
                sock.close()
    raise RuntimeError(f"Timed out waiting for {host}:{port}")


def _tail_lines(path: str, *, line_count: int = 120) -> str:
    try:
        raw = Path(path).read_text(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return ""
    lines = raw.splitlines()
    return "\n".join(lines[-line_count:])


async def _prepare_session_and_payload() -> tuple[str, str]:
    await db_module.close_postgres()
    await db_module.init_postgres()
    assert db_module.AsyncSessionLocal is not None

    async with db_module.AsyncSessionLocal() as db:
        user = await _insert_test_user(db)
        session = AgentSession(
            user_id=user.id,
            current_phase="strategy",
            status="active",
            artifacts={},
            metadata_={},
        )
        db.add(session)
        await db.flush()

        dsl_payload = load_strategy_payload(EXAMPLE_PATH)
        dsl_payload["strategy"]["name"] = f"openai_e2e_{uuid4().hex[:8]}"
        dsl_payload["universe"]["market"] = "crypto"
        dsl_payload["universe"]["tickers"] = ["BTCUSD"]
        dsl_payload["timeframe"] = "5m"

        await db.commit()
        session_id = str(session.id)
        dsl_json = json.dumps(dsl_payload, ensure_ascii=False, separators=(",", ":"))

    await db_module.close_postgres()
    return (session_id, dsl_json)


async def _insert_test_user(db: AsyncSession) -> User:
    email = f"openai_e2e_{uuid4().hex[:10]}@example.com"
    user = User(email=email, password_hash="hash", name=email)
    db.add(user)
    await db.flush()
    return user


def _run_openai_case(
    *,
    client: OpenAI,
    model: str,
    server_url: str,
    prompt: str,
    allowed_tools: list[str],
    timeout_seconds: float,
    case_name: str,
) -> OpenAICaseResult:
    event_counter: Counter[str] = Counter()
    assistant_text_parts: list[str] = []
    called_tools: list[str] = []
    call_records: list[ToolCallRecord] = []
    parsed_outputs_by_tool: dict[str, list[dict[str, Any]]] = {}
    response_id: str | None = None
    api_error: str | None = None

    seen_call_keys: set[str] = set()

    def _record_item(item: dict[str, Any]) -> None:
        if item.get("type") != "mcp_call":
            return
        tool_name = item.get("name")
        if not isinstance(tool_name, str) or not tool_name:
            return

        call_id = item.get("id") if isinstance(item.get("id"), str) else ""
        status = item.get("status") if isinstance(item.get("status"), str) else None
        dedupe_key = f"{call_id}:{tool_name}:{status}"
        if dedupe_key in seen_call_keys:
            return
        seen_call_keys.add(dedupe_key)

        parsed_output = _extract_json_payload_from_tool_output(item.get("output"))
        output_ok, error_code, output_job_id, output_strategy_id = _extract_tool_outcome(parsed_output)

        called_tools.append(tool_name)
        call_records.append(
            ToolCallRecord(
                tool=tool_name,
                status=status,
                output_ok=output_ok,
                output_error_code=error_code,
                output_job_id=output_job_id,
                output_strategy_id=output_strategy_id,
            )
        )
        if isinstance(parsed_output, dict):
            parsed_outputs_by_tool.setdefault(tool_name, []).append(parsed_output)

    try:
        with client.responses.stream(
            model=model,
            input=prompt,
            tools=[
                {
                    "type": "mcp",
                    "server_label": "minsy_local",
                    "server_url": server_url,
                    "allowed_tools": allowed_tools,
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
                if isinstance(item, dict):
                    _record_item(item)

            final_response = stream.get_final_response()
            response_id = final_response.id
            for output_item in final_response.output or []:
                item = _dump_model(output_item)
                _record_item(item)

    except APIError as exc:
        api_error = f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # noqa: BLE001
        api_error = f"{type(exc).__name__}: {exc}"

    missing = [tool for tool in allowed_tools if tool not in called_tools]
    bad_outputs: list[str] = []
    for record in call_records:
        if record.output_ok is False:
            bad_outputs.append(f"{record.tool}:{record.output_error_code or 'UNKNOWN'}")

    ok = api_error is None and not missing and not bad_outputs
    reason_parts: list[str] = []
    if missing:
        reason_parts.append(f"missing_tools={missing}")
    if bad_outputs:
        reason_parts.append(f"tool_errors={bad_outputs}")
    if api_error:
        reason_parts.append(api_error)
    reason = "ok" if not reason_parts else "; ".join(reason_parts)

    return OpenAICaseResult(
        name=case_name,
        ok=ok,
        reason=reason,
        response_id=response_id,
        api_error=api_error,
        called_tools=called_tools,
        call_records=[asdict(item) for item in call_records],
        parsed_outputs_by_tool=parsed_outputs_by_tool,
        event_counts=dict(event_counter),
        assistant_text="".join(assistant_text_parts).strip(),
    )


def _extract_strategy_id(result: OpenAICaseResult) -> str:
    outputs = result.parsed_outputs_by_tool.get("strategy_upsert_dsl", [])
    for payload in outputs:
        strategy_id = payload.get("strategy_id")
        if isinstance(strategy_id, str) and strategy_id:
            return strategy_id
    raise RuntimeError(f"No strategy_id from strategy_upsert_dsl output: {outputs}")


def _extract_terminal_job_id(result: OpenAICaseResult) -> str:
    create_outputs = result.parsed_outputs_by_tool.get("backtest_create_job", [])
    get_outputs = result.parsed_outputs_by_tool.get("backtest_get_job", [])

    for payload in reversed(get_outputs):
        status = payload.get("status")
        job_id = payload.get("job_id")
        if isinstance(status, str) and status in {"done", "failed"} and isinstance(job_id, str):
            return job_id

    for payload in reversed(create_outputs):
        job_id = payload.get("job_id")
        if isinstance(job_id, str) and job_id:
            return job_id

    raise RuntimeError(
        f"No usable job_id from backtest_create_job/get_job outputs: create={create_outputs} get={get_outputs}",
    )


def _run_simulated_tests() -> tuple[bool, str, str]:
    cmd = (
        "uv run pytest "
        "tests/test_engine/test_backtest_analytics.py "
        "tests/test_mcp/test_backtest_tools.py "
        "tests/test_mcp/test_strategy_tools.py -q"
    )
    proc = subprocess.run(  # noqa: S603
        cmd,
        cwd=str(Path(__file__).resolve().parents[2]),
        shell=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    output = proc.stdout or ""
    tail = "\n".join(output.splitlines()[-120:])
    return (proc.returncode == 0, cmd, tail)


def main() -> int:
    args = _parse_args()
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    api_key = os.getenv("OPENAI_API_KEY") or settings.openai_api_key
    if not isinstance(api_key, str) or not api_key.strip():
        print("OPENAI_API_KEY is missing.", file=sys.stderr)
        return 1

    mcp_server_url = args.mcp_server_url.strip() or f"http://{args.mcp_host}:{args.mcp_port}/mcp"
    openai_server_url = args.openai_mcp_server_url.strip()

    mcp_proc: subprocess.Popen[str] | None = None
    worker_proc: subprocess.Popen[str] | None = None
    ngrok_proc: subprocess.Popen[str] | None = None
    mcp_log = NamedTemporaryFile(prefix="openai_e2e_mcp_", suffix=".log", delete=False)
    worker_log = NamedTemporaryFile(prefix="openai_e2e_worker_", suffix=".log", delete=False)
    ngrok_log = NamedTemporaryFile(prefix="openai_e2e_ngrok_", suffix=".log", delete=False)

    simulated_ok = True
    simulated_cmd = ""
    simulated_tail = ""

    strategy_case: OpenAICaseResult | None = None
    backtest_case: OpenAICaseResult | None = None
    performance_cases: list[OpenAICaseResult] = []

    overall_ok = True

    try:
        if args.run_simulated_tests:
            print("[stage] running simulated tests ...", flush=True)
            simulated_ok, simulated_cmd, simulated_tail = _run_simulated_tests()
            overall_ok = overall_ok and simulated_ok
            print(f"[stage] simulated tests ok={simulated_ok}", flush=True)

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
                "1",
                "--loglevel",
                "INFO",
            ]
            worker_proc = _start_process(
                worker_cmd,
                cwd=str(Path(__file__).resolve().parents[2]),
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
                cwd=str(Path(__file__).resolve().parents[2]),
                stdout=mcp_log,
                stderr=mcp_log,
            )
            _wait_for_port(args.mcp_host, args.mcp_port, args.startup_timeout_seconds)
            print(f"[stage] mcp ready at {mcp_server_url}", flush=True)

        if not openai_server_url:
            if args.start_ngrok:
                print("[stage] starting ngrok tunnel ...", flush=True)
                ngrok_cmd = ["ngrok", "http", str(args.mcp_port), "--pooling-enabled"]
                ngrok_proc = _start_process(
                    ngrok_cmd,
                    cwd=str(Path(__file__).resolve().parents[2]),
                    stdout=ngrok_log,
                    stderr=ngrok_log,
                )
                public_url = _get_ngrok_public_url(
                    timeout_seconds=args.ngrok_timeout_seconds,
                    mcp_port=args.mcp_port,
                    ngrok_proc=ngrok_proc,
                )
                openai_server_url = f"{public_url}/mcp"
                print(f"[stage] ngrok public mcp url={openai_server_url}", flush=True)
            else:
                openai_server_url = mcp_server_url

        session_id, dsl_json = asyncio.run(_prepare_session_and_payload())
        print(f"[stage] prepared strategy session_id={session_id}", flush=True)

        openai_client = OpenAI(api_key=api_key)

        strategy_prompt = (
            "You must call MCP tool strategy_upsert_dsl exactly once and then stop. "
            "Use arguments: "
            f"session_id='{session_id}', "
            f"dsl_json='{dsl_json}'. "
            "Return a short confirmation after tool call."
        )
        print("[stage] running openai strategy submission case ...", flush=True)
        strategy_case = _run_openai_case(
            client=openai_client,
            model=args.model,
            server_url=openai_server_url,
            prompt=strategy_prompt,
            allowed_tools=["strategy_upsert_dsl"],
            timeout_seconds=args.openai_timeout_seconds,
            case_name="openai_strategy_upsert",
        )
        print(f"[case] strategy ok={strategy_case.ok} reason={strategy_case.reason}", flush=True)
        overall_ok = overall_ok and strategy_case.ok

        strategy_id = _extract_strategy_id(strategy_case)

        backtest_prompt = (
            "Use MCP tools to run a backtest and wait until terminal status. "
            "Steps: "
            "1) call backtest_create_job with "
            f"strategy_id='{strategy_id}', start_date='2024-01-01T00:00:00+00:00', "
            "end_date='2024-03-01T00:00:00+00:00', initial_capital=100000, "
            "commission_rate=0.0, slippage_bps=0.0, run_now=false. "
            "2) call backtest_get_job repeatedly until status is done or failed. "
            f"Wait about {args.poll_wait_seconds:.1f} seconds between polls. "
            f"Do not exceed {args.max_backtest_polls} polls. "
            "Do not ask follow-up questions."
        )
        print("[stage] running openai backtest case ...", flush=True)
        backtest_case = _run_openai_case(
            client=openai_client,
            model=args.model,
            server_url=openai_server_url,
            prompt=backtest_prompt,
            allowed_tools=["backtest_create_job", "backtest_get_job"],
            timeout_seconds=args.openai_timeout_seconds,
            case_name="openai_backtest_create_and_poll",
        )
        print(f"[case] backtest ok={backtest_case.ok} reason={backtest_case.reason}", flush=True)
        overall_ok = overall_ok and backtest_case.ok

        job_id = _extract_terminal_job_id(backtest_case)

        for tool_name in PERFORMANCE_TOOL_NAMES:
            prompt = (
                f"Call MCP tool {tool_name} exactly once with job_id='{job_id}'. "
                "Then return a one-line confirmation."
            )
            print(f"[stage] running performance case tool={tool_name} ...", flush=True)
            case_result = _run_openai_case(
                client=openai_client,
                model=args.model,
                server_url=openai_server_url,
                prompt=prompt,
                allowed_tools=[tool_name],
                timeout_seconds=args.openai_timeout_seconds,
                case_name=f"openai_{tool_name}",
            )
            print(f"[case] {tool_name} ok={case_result.ok} reason={case_result.reason}", flush=True)
            performance_cases.append(case_result)
            overall_ok = overall_ok and case_result.ok

        summary = {
            "overall_ok": overall_ok,
            "strategy_upsert_ok": strategy_case.ok if strategy_case else False,
            "backtest_flow_ok": backtest_case.ok if backtest_case else False,
            "performance_tools_ok_count": sum(1 for item in performance_cases if item.ok),
            "performance_tools_total": len(performance_cases),
            "missing_performance_tools": [
                item.name for item in performance_cases if not item.ok
            ],
        }

        report = FullReport(
            ok=overall_ok,
            timestamp_utc=datetime.now(UTC).isoformat(),
            model=args.model,
            mcp_server_url=mcp_server_url,
            openai_mcp_server_url=openai_server_url,
            simulated_tests_ok=simulated_ok,
            simulated_test_command=simulated_cmd,
            simulated_test_output_tail=simulated_tail,
            strategy_case=asdict(strategy_case) if strategy_case else {},
            backtest_case=asdict(backtest_case) if backtest_case else {},
            performance_cases=[asdict(item) for item in performance_cases],
            summary=summary,
        )
        report_path.write_text(json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"[report] wrote {report_path}", flush=True)
        print(f"[overall] ok={report.ok}", flush=True)

        if not report.ok:
            print(f"[debug] worker_log={worker_log.name}", flush=True)
            print(f"[debug] mcp_log={mcp_log.name}", flush=True)
            if ngrok_proc is not None:
                print(f"[debug] ngrok_log={ngrok_log.name}", flush=True)

        if args.always_zero:
            return 0
        return 0 if report.ok else 1

    finally:
        for fp in (worker_log, mcp_log, ngrok_log):
            try:
                fp.flush()
            except Exception:  # noqa: BLE001
                pass
            try:
                fp.close()
            except Exception:  # noqa: BLE001
                pass

        if ngrok_proc is not None and ngrok_proc.poll() is None:
            ngrok_proc.terminate()
            try:
                ngrok_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                ngrok_proc.kill()
                ngrok_proc.wait(timeout=5)

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


if __name__ == "__main__":
    raise SystemExit(main())
