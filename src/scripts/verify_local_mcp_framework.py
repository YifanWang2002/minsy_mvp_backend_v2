#!/usr/bin/env python3
"""Verify the modular local MCP server and OpenAI allowed_tools behavior."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

import httpx
from openai import APIError, OpenAI

from src.mcp.server import ALL_REGISTERED_TOOL_NAMES

DEFAULT_MODEL = "gpt-5.2"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8111
DEFAULT_ALLOWED_TOOLS = ("market_data_get_quote", "backtest_create_job")
DEFAULT_BLOCKED_TOOL = "strategy_generate_outline"


@dataclass
class ToolSummary:
    name: str
    description: str


@dataclass
class OpenAIAttemptResult:
    name: str
    prompt: str
    allowed_tools: list[str]
    tool_choice: str
    expect_success: bool
    ok: bool = False
    reason: str = ""
    response_id: str | None = None
    event_counts: dict[str, int] = field(default_factory=dict)
    called_tools: list[str] = field(default_factory=list)
    statuses: list[str] = field(default_factory=list)
    item_errors: list[str] = field(default_factory=list)
    api_error: str | None = None
    output_text: str = ""

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _dump_model(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json", exclude_none=True)
    if isinstance(obj, dict):
        return obj
    return {"value": str(obj)}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify local modular MCP server list_tools and OpenAI Responses "
            "allowed_tools behavior."
        )
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Local MCP host.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Local MCP port.")
    parser.add_argument(
        "--server-url",
        default="",
        help=(
            "MCP server URL for OpenAI call. "
            "Default uses local URL http://{host}:{port}/mcp."
        ),
    )
    parser.add_argument(
        "--no-start-server",
        action="store_true",
        help="Do not start local server subprocess. Assume server already running.",
    )
    parser.add_argument(
        "--startup-timeout-seconds",
        type=float,
        default=20.0,
        help="Timeout while waiting local server to become listable.",
    )
    parser.add_argument(
        "--openai-timeout-seconds",
        type=float,
        default=120.0,
        help="Per-attempt OpenAI stream timeout.",
    )
    parser.add_argument(
        "--skip-openai",
        action="store_true",
        help="Only verify local list_tools, skip OpenAI calls.",
    )
    parser.add_argument(
        "--json-report",
        default="",
        help="Optional path to write JSON report.",
    )
    parser.add_argument(
        "--always-zero",
        action="store_true",
        help="Always exit code 0 even on failures.",
    )
    return parser.parse_args()


def _extract_jsonrpc_messages(response_text: str) -> list[dict[str, Any]]:
    stripped = response_text.strip()
    if not stripped:
        return []

    # Some MCP servers may return raw JSON in non-stream mode.
    if stripped.startswith("{"):
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return []
        return [parsed] if isinstance(parsed, dict) else []

    # Stream mode: parse SSE lines.
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

    messages = _extract_jsonrpc_messages(response.text)
    for message in messages:
        if message.get("id") != request_id:
            continue
        if "result" in message and isinstance(message["result"], dict):
            return message["result"], session_id_resp
        if "error" in message:
            error = message["error"]
            raise RuntimeError(f"MCP method={method} error={error}")

    raise RuntimeError(f"MCP method={method} missing result for id={request_id}")


def _list_tools(server_url: str, timeout_seconds: float) -> list[ToolSummary]:
    initialize_result, session_id = _mcp_rpc_call(
        server_url=server_url,
        method="initialize",
        request_id="init-1",
        params={
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "verify-local-framework", "version": "0.1.0"},
        },
        timeout_seconds=timeout_seconds,
    )
    if "protocolVersion" not in initialize_result:
        raise RuntimeError(f"Invalid initialize result: {initialize_result}")

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

    summaries: list[ToolSummary] = []
    for item in raw_tools:
        if not isinstance(item, dict):
            continue
        name = item.get("name", "")
        description = item.get("description", "")
        summaries.append(ToolSummary(name=str(name), description=str(description).strip()))
    return summaries


def _wait_for_server_ready(server_url: str, timeout_seconds: float) -> list[ToolSummary]:
    deadline = time.time() + max(1.0, timeout_seconds)
    last_error: str | None = None
    while time.time() < deadline:
        try:
            return _list_tools(server_url, timeout_seconds=5.0)
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(0.5)
    raise RuntimeError(f"Server not ready at {server_url}. last_error={last_error}")


def _run_openai_attempt(
    client: OpenAI,
    *,
    model: str,
    server_url: str,
    prompt: str,
    allowed_tools: list[str],
    tool_choice_name: str,
    expect_success: bool,
    timeout_seconds: float,
    server_label: str = "minsy_local",
) -> OpenAIAttemptResult:
    result = OpenAIAttemptResult(
        name=tool_choice_name,
        prompt=prompt,
        allowed_tools=allowed_tools,
        tool_choice=tool_choice_name,
        expect_success=expect_success,
    )
    event_counter: Counter[str] = Counter()

    tool_def: dict[str, Any] = {
        "type": "mcp",
        "server_label": server_label,
        "server_url": server_url,
        "allowed_tools": allowed_tools,
        "require_approval": "never",
    }

    try:
        with client.responses.stream(
            model=model,
            input=prompt,
            tools=[tool_def],
            tool_choice={
                "type": "mcp",
                "server_label": server_label,
                "name": tool_choice_name,
            },
            timeout=timeout_seconds,
        ) as stream:
            for event in stream:
                event_type = getattr(event, "type", "unknown")
                event_counter[event_type] += 1
                if event_type == "response.output_text.delta":
                    delta = getattr(event, "delta", "")
                    if delta:
                        result.output_text += delta
                    continue

                if event_type != "response.output_item.added":
                    continue

                event_payload = _dump_model(event)
                item = event_payload.get("item") or {}
                item_type = item.get("type")

                if item_type == "mcp_call":
                    name = item.get("name")
                    if isinstance(name, str):
                        result.called_tools.append(name)
                    status = item.get("status")
                    if isinstance(status, str):
                        result.statuses.append(status)
                    error_text = item.get("error")
                    if isinstance(error_text, str) and error_text:
                        result.item_errors.append(error_text)

                if item_type == "mcp_list_tools":
                    error_text = item.get("error")
                    if isinstance(error_text, str) and error_text:
                        result.item_errors.append(error_text)

            final_response = stream.get_final_response()
            result.response_id = final_response.id
            for output_item in final_response.output or []:
                output_payload = _dump_model(output_item)
                if output_payload.get("type") != "mcp_call":
                    continue
                name = output_payload.get("name")
                if isinstance(name, str):
                    result.called_tools.append(name)
                status = output_payload.get("status")
                if isinstance(status, str):
                    result.statuses.append(status)
                error_text = output_payload.get("error")
                if isinstance(error_text, str) and error_text:
                    result.item_errors.append(error_text)

    except APIError as exc:
        result.api_error = f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # noqa: BLE001
        result.api_error = f"{type(exc).__name__}: {exc}"
    finally:
        result.event_counts = dict(event_counter)

    called_expected = tool_choice_name in result.called_tools
    called_only_allowed = all(name in allowed_tools for name in result.called_tools)
    has_completed = "completed" in result.statuses
    has_failed_event = any(
        key.endswith(".failed") and count > 0 for key, count in result.event_counts.items()
    )

    if expect_success:
        if (
            called_expected
            and called_only_allowed
            and has_completed
            and not has_failed_event
            and not result.item_errors
            and not result.api_error
        ):
            result.ok = True
            result.reason = "Tool call completed and respected allowed_tools."
        else:
            failures: list[str] = []
            if not called_expected:
                failures.append("expected tool not observed")
            if not called_only_allowed:
                failures.append("unexpected non-allowed tool observed")
            if not has_completed:
                failures.append("no completed status in mcp_call")
            if has_failed_event:
                failures.append("failed stream event detected")
            if result.item_errors:
                failures.append(f"mcp item error: {result.item_errors[-1]}")
            if result.api_error:
                failures.append(f"api error: {result.api_error}")
            result.ok = False
            result.reason = "; ".join(failures) if failures else "unknown failure"
    else:
        blocked_called = called_expected
        if (result.api_error and not blocked_called) or (not blocked_called and not has_completed):
            result.ok = True
            result.reason = "Blocked tool was not successfully executed."
        else:
            result.ok = False
            result.reason = "Blocked tool unexpectedly executed."
    return result


def _print_tool_list(tools: list[ToolSummary]) -> None:
    print("=== Local MCP list_tools ===")
    for idx, tool in enumerate(sorted(tools, key=lambda item: item.name), start=1):
        description = tool.description or "(no description)"
        print(f"[{idx:02d}] {tool.name} :: {description}")
    print()


def _start_server_subprocess(host: str, port: int) -> subprocess.Popen[str]:
    cmd = [
        sys.executable,
        "-m",
        "src.mcp.server",
        "--transport",
        "streamable-http",
        "--host",
        host,
        "--port",
        str(port),
    ]
    return subprocess.Popen(
        cmd,
        cwd=os.getcwd(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )


def main() -> int:
    args = _parse_args()
    local_server_url = f"http://{args.host}:{args.port}/mcp"
    openai_server_url = args.server_url.strip() or local_server_url

    expected_tool_names = set(ALL_REGISTERED_TOOL_NAMES)
    report: dict[str, Any] = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "model": args.model,
        "local_server_url": local_server_url,
        "openai_server_url": openai_server_url,
        "list_tools": {},
        "openai_attempts": [],
    }

    server_proc: subprocess.Popen[str] | None = None
    try:
        if not args.no_start_server:
            server_proc = _start_server_subprocess(args.host, args.port)

        listed_tools = _wait_for_server_ready(local_server_url, args.startup_timeout_seconds)
        _print_tool_list(listed_tools)
        listed_names = {tool.name for tool in listed_tools}

        missing = sorted(expected_tool_names - listed_names)
        extras = sorted(listed_names - expected_tool_names)
        list_ok = not missing
        print(
            f"[list_tools] ok={list_ok} expected={len(expected_tool_names)} "
            f"listed={len(listed_names)}"
        )
        if missing:
            print(f"[list_tools] missing={missing}")
        if extras:
            print(f"[list_tools] extra={extras}")
        print()

        report["list_tools"] = {
            "ok": list_ok,
            "expected_count": len(expected_tool_names),
            "listed_count": len(listed_names),
            "missing": missing,
            "extra": extras,
            "tools": [asdict(tool) for tool in listed_tools],
        }

        all_ok = list_ok
        if not args.skip_openai:
            if not os.getenv("OPENAI_API_KEY"):
                print("[openai] skipped: OPENAI_API_KEY missing")
                all_ok = False
            else:
                client = OpenAI()
                allowed_tools = list(DEFAULT_ALLOWED_TOOLS)
                attempts = [
                    (
                        "market_data_get_quote",
                        (
                            "Call MCP tool market_data_get_quote exactly once with symbol AAPL "
                            "and venue US. Then summarize in one line."
                        ),
                        True,
                    ),
                    (
                        "backtest_create_job",
                        (
                            "Call MCP tool backtest_create_job exactly once with strategy_id "
                            "00000000-0000-0000-0000-000000000001 and run_now=false. "
                            "Then summarize in one line."
                        ),
                        True,
                    ),
                    (
                        DEFAULT_BLOCKED_TOOL,
                        (
                            "Call MCP tool strategy_generate_outline exactly once with goal "
                            "momentum strategy for US equities. Then summarize in one line."
                        ),
                        False,
                    ),
                ]

                print("=== OpenAI allowed_tools verification ===")
                for idx, (tool_choice, prompt, expect_success) in enumerate(attempts, start=1):
                    print(
                        f"[{idx:02d}/{len(attempts):02d}] choice={tool_choice} "
                        f"expect_success={expect_success}"
                    )
                    attempt_result = _run_openai_attempt(
                        client,
                        model=args.model,
                        server_url=openai_server_url,
                        prompt=prompt,
                        allowed_tools=allowed_tools,
                        tool_choice_name=tool_choice,
                        expect_success=expect_success,
                        timeout_seconds=args.openai_timeout_seconds,
                    )
                    report["openai_attempts"].append(attempt_result.to_json())
                    all_ok = all_ok and attempt_result.ok

                    status = "PASS" if attempt_result.ok else "FAIL"
                    print(f"  -> {status}: {attempt_result.reason}")
                    if attempt_result.called_tools:
                        print(f"  -> called_tools={attempt_result.called_tools}")
                    if attempt_result.statuses:
                        print(f"  -> statuses={attempt_result.statuses}")
                    if attempt_result.item_errors:
                        print(f"  -> item_errors={attempt_result.item_errors}")
                    if attempt_result.api_error:
                        print(f"  -> api_error={attempt_result.api_error}")
                    if attempt_result.event_counts:
                        print(f"  -> events={json.dumps(attempt_result.event_counts)}")
                    print()

        report["ok"] = all_ok
        if args.json_report:
            with open(args.json_report, "w", encoding="utf-8") as fp:
                json.dump(report, fp, ensure_ascii=False, indent=2)
            print(f"[report] wrote JSON report: {args.json_report}")

        print(f"[overall] ok={report['ok']}")
        if args.always_zero:
            return 0
        return 0 if report["ok"] else 1
    finally:
        if server_proc is not None and server_proc.poll() is None:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_proc.kill()
                server_proc.wait(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
