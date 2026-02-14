#!/usr/bin/env python3
"""Verify strategy DSL validate/upsert MCP calls via ngrok and real OpenAI endpoint."""

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
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
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


@dataclass(frozen=True, slots=True)
class Case:
    name: str
    session_id: str
    dsl_json: str
    tool_name: str
    expect_ok: bool
    expected_error_code: str | None = None


@dataclass
class CaseResult:
    name: str
    expect_ok: bool
    expected_error_code: str | None
    pass_expectation: bool = False
    response_id: str | None = None
    called_tools: list[str] | None = None
    statuses: list[str] | None = None
    output_text: str = ""
    parsed_payload: dict[str, Any] | None = None
    parse_error: str | None = None
    api_error: str | None = None
    event_counts: dict[str, int] | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify strategy DSL validate/upsert MCP flow with real OpenAI + ngrok"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--mcp-host", default=DEFAULT_MCP_HOST)
    parser.add_argument("--mcp-port", type=int, default=DEFAULT_MCP_PORT)
    parser.add_argument(
        "--mcp-server-url",
        default="",
        help="Optional public MCP server URL. If set, ngrok auto-discovery is skipped.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--output", default="logs/strategy_mcp_id_flow_report.json")
    parser.add_argument("--always-zero", action="store_true")
    return parser.parse_args()


def _extract_json_text(raw: str) -> str:
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, flags=re.DOTALL)
    if fenced:
        raw = fenced.group(1)

    start = raw.find("{")
    if start == -1:
        return raw

    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(raw)):
        ch = raw[idx]
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return raw[start : idx + 1]
    return raw[start:]


def _dump_model(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json", exclude_none=True)
    if isinstance(obj, dict):
        return obj
    return {"value": str(obj)}


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
                )
                response.raise_for_status()
                payload = response.json()
                tunnels = payload.get("tunnels", [])
                for tunnel in tunnels:
                    public_url = tunnel.get("public_url")
                    config = tunnel.get("config") if isinstance(tunnel, dict) else None
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


def _matches_expected_port(*, addr: str, mcp_port: int) -> bool:
    normalized = addr.strip()
    if not normalized:
        return False
    return re.search(rf":{mcp_port}(?:/|$)", normalized) is not None


def _extract_jsonrpc_messages(response_text: str) -> list[dict[str, Any]]:
    stripped = response_text.strip()
    if not stripped:
        return []

    if stripped.startswith("{"):
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return []
        return [parsed] if isinstance(parsed, dict) else []

    messages: list[dict[str, Any]] = []
    for line in response_text.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line.split("data:", 1)[1].strip()
        if not payload:
            continue
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            messages.append(parsed)
    return messages


def _mcp_rpc_call(
    *,
    server_url: str,
    method: str,
    request_id: str,
    params: dict[str, Any],
    timeout_seconds: float,
    session_id: str | None = None,
) -> tuple[dict[str, Any], str | None]:
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
    response = httpx.post(
        server_url,
        headers=headers,
        json=payload,
        timeout=max(5.0, timeout_seconds),
        trust_env=False,
    )
    response.raise_for_status()
    session_id_resp = response.headers.get("mcp-session-id") or session_id

    for message in _extract_jsonrpc_messages(response.text):
        if message.get("id") != request_id:
            continue
        if "result" in message and isinstance(message["result"], dict):
            return message["result"], session_id_resp
        if "error" in message:
            raise RuntimeError(f"MCP method={method} error={message['error']}")

    raise RuntimeError(f"MCP method={method} missing result for id={request_id}")


def _assert_required_tool_available(
    *,
    server_url: str,
    required_tool: str,
    timeout_seconds: float,
) -> None:
    init_result, session_id = _mcp_rpc_call(
        server_url=server_url,
        method="initialize",
        request_id="init-1",
        params={
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "verify-strategy-mcp-id-flow", "version": "0.1.0"},
        },
        timeout_seconds=timeout_seconds,
    )
    if "protocolVersion" not in init_result:
        raise RuntimeError(f"Invalid initialize result: {init_result}")

    tools_result, _ = _mcp_rpc_call(
        server_url=server_url,
        method="tools/list",
        request_id="tools-1",
        params={},
        timeout_seconds=timeout_seconds,
        session_id=session_id,
    )
    raw_tools = tools_result.get("tools")
    if not isinstance(raw_tools, list):
        raise RuntimeError(f"Invalid tools/list result: {tools_result}")
    tool_names = sorted(
        str(item.get("name"))
        for item in raw_tools
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    )
    if required_tool not in tool_names:
        raise RuntimeError(
            "Required MCP tool is missing from remote server "
            f"'{required_tool}'. available_tools={tool_names}"
        )


async def _prepare_strategy() -> tuple[str, str, str]:
    await db_module.close_postgres()
    await db_module.init_postgres()
    assert db_module.AsyncSessionLocal is not None

    async with db_module.AsyncSessionLocal() as db:
        session_id = await _create_strategy_session(db)
        payload = load_strategy_payload(EXAMPLE_PATH)
        valid_dsl_json = json.dumps(payload, ensure_ascii=False)
        invalid_payload = deepcopy(payload)
        invalid_payload.pop("timeframe", None)
        invalid_dsl_json = json.dumps(invalid_payload, ensure_ascii=False)
        await db.commit()
    await db_module.close_postgres()
    return session_id, valid_dsl_json, invalid_dsl_json


async def _create_strategy_session(db: AsyncSession) -> str:
    email = f"verify_mcp_{uuid4().hex[:10]}@example.com"
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
    return str(session.id)


def _run_case(
    *,
    client: OpenAI,
    case: Case,
    model: str,
    mcp_server_url: str,
    timeout_seconds: float,
) -> CaseResult:
    result = CaseResult(
        name=case.name,
        expect_ok=case.expect_ok,
        expected_error_code=case.expected_error_code,
        called_tools=[],
        statuses=[],
        event_counts={},
    )
    counter: Counter[str] = Counter()
    try:
        escaped_dsl_json = json.dumps(case.dsl_json, ensure_ascii=False)
        prompt = (
            f"Call MCP tool {case.tool_name} exactly once, "
            f"with session_id='{case.session_id}' and dsl_json={escaped_dsl_json}. "
            "Then output only the tool JSON result."
        )
        with client.responses.stream(
            model=model,
            input=prompt,
            tools=[
                {
                    "type": "mcp",
                    "server_label": "strategy_remote",
                    "server_url": mcp_server_url,
                    "allowed_tools": [case.tool_name],
                    "require_approval": "never",
                }
            ],
            tool_choice={
                "type": "mcp",
                "server_label": "strategy_remote",
                "name": case.tool_name,
            },
            timeout=timeout_seconds,
        ) as stream:
            for event in stream:
                event_type = getattr(event, "type", "unknown")
                counter[event_type] += 1

                if event_type == "response.output_text.delta":
                    delta = getattr(event, "delta", "")
                    if isinstance(delta, str):
                        result.output_text += delta
                    continue

                if event_type != "response.output_item.added":
                    continue
                payload = _dump_model(event)
                item = payload.get("item") or {}
                if item.get("type") == "mcp_call":
                    name = item.get("name")
                    status = item.get("status")
                    if isinstance(name, str):
                        result.called_tools.append(name)
                    if isinstance(status, str):
                        result.statuses.append(status)

            final_response = stream.get_final_response()
            result.response_id = final_response.id
            for output_item in final_response.output or []:
                payload = _dump_model(output_item)
                if payload.get("type") != "mcp_call":
                    continue
                name = payload.get("name")
                status = payload.get("status")
                if isinstance(name, str):
                    result.called_tools.append(name)
                if isinstance(status, str):
                    result.statuses.append(status)

        json_text = _extract_json_text(result.output_text)
        parsed = json.loads(json_text)
        if not isinstance(parsed, dict):
            raise ValueError("Model output is not a JSON object")
        result.parsed_payload = parsed

        ok_flag = bool(parsed.get("ok"))
        error_code = None
        error = parsed.get("error")
        if isinstance(error, dict):
            raw_code = error.get("code")
            if isinstance(raw_code, str):
                error_code = raw_code

        if case.expect_ok:
            result.pass_expectation = ok_flag is True
        else:
            result.pass_expectation = ok_flag is False and (
                case.expected_error_code is None
                or error_code == case.expected_error_code
            )
    except (json.JSONDecodeError, ValueError) as exc:
        result.parse_error = str(exc)
        result.pass_expectation = False
    except APIError as exc:
        result.api_error = f"{type(exc).__name__}: {exc}"
        result.pass_expectation = False
    except Exception as exc:  # noqa: BLE001
        result.api_error = f"{type(exc).__name__}: {exc}"
        result.pass_expectation = False
    result.event_counts = dict(counter)
    return result


def main() -> int:
    args = _parse_args()
    api_key = os.getenv("OPENAI_API_KEY") or settings.openai_api_key
    if not api_key:
        print("OPENAI_API_KEY is missing.", file=sys.stderr)
        return 2

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
    ngrok_cmd = ["ngrok", "http", str(args.mcp_port), "--pooling-enabled"]

    mcp_proc: subprocess.Popen[Any] | None = None
    ngrok_proc: subprocess.Popen[Any] | None = None
    started_at = datetime.now(UTC).isoformat()

    try:
        session_id, valid_dsl_json, invalid_dsl_json = asyncio.run(_prepare_strategy())

        mcp_proc = subprocess.Popen(  # noqa: S603
            mcp_cmd,
            cwd=Path(__file__).resolve().parents[2],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(1.5)

        if args.mcp_server_url.strip():
            mcp_server_url = args.mcp_server_url.strip().rstrip("/")
        else:
            ngrok_proc = subprocess.Popen(  # noqa: S603
                ngrok_cmd,
                cwd=Path(__file__).resolve().parents[2],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            public_url = _get_ngrok_public_url(
                timeout_seconds=20.0,
                mcp_port=args.mcp_port,
                ngrok_proc=ngrok_proc,
            )
            mcp_server_url = f"{public_url}/mcp"

        _assert_required_tool_available(
            server_url=mcp_server_url,
            required_tool="strategy_upsert_dsl",
            timeout_seconds=15.0,
        )

        client = OpenAI(api_key=api_key)
        cases = [
            Case(
                name="upsert_valid_dsl",
                session_id=session_id,
                dsl_json=valid_dsl_json,
                tool_name="strategy_upsert_dsl",
                expect_ok=True,
            ),
            Case(
                name="upsert_invalid_dsl",
                session_id=session_id,
                dsl_json=invalid_dsl_json,
                tool_name="strategy_upsert_dsl",
                expect_ok=False,
                expected_error_code="STRATEGY_VALIDATION_FAILED",
            ),
        ]

        results = []
        for index, case in enumerate(cases, start=1):
            print(
                f"[verify-mcp] running case {index}/{len(cases)}: {case.name}",
                file=sys.stderr,
            )
            case_result = _run_case(
                client=client,
                case=case,
                model=args.model,
                mcp_server_url=mcp_server_url,
                timeout_seconds=args.timeout_seconds,
            )
            results.append(case_result)
            print(
                f"[verify-mcp] done case {index}/{len(cases)}: {case.name} pass={case_result.pass_expectation}",
                file=sys.stderr,
            )

        passed = sum(1 for item in results if item.pass_expectation)
        report = {
            "started_at_utc": started_at,
            "finished_at_utc": datetime.now(UTC).isoformat(),
            "model": args.model,
            "mcp_server_url": mcp_server_url,
            "total_cases": len(results),
            "passed_cases": passed,
            "failed_cases": len(results) - passed,
            "results": [item.to_json() for item in results],
        }
        text = json.dumps(report, ensure_ascii=False, indent=2)
        print(text)

        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")

        if args.always_zero:
            return 0
        return 0 if passed == len(results) else 1
    finally:
        if ngrok_proc is not None and ngrok_proc.poll() is None:
            ngrok_proc.terminate()
        if mcp_proc is not None and mcp_proc.poll() is None:
            mcp_proc.terminate()


if __name__ == "__main__":
    raise SystemExit(main())
