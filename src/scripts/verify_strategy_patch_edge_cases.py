#!/usr/bin/env python3
"""Stress-test strategy patch workflow on edge cases via real OpenAI + MCP."""

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
from uuid import UUID, uuid4

import httpx
from openai import APIError, OpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.engine.strategy import EXAMPLE_PATH, load_strategy_payload, upsert_strategy_dsl
from src.models import database as db_module
from src.models.session import Session as AgentSession
from src.models.strategy import Strategy
from src.models.user import User

DEFAULT_MODEL = settings.openai_response_model
DEFAULT_MCP_HOST = "127.0.0.1"
DEFAULT_MCP_PORT = 8111


@dataclass(frozen=True, slots=True)
class EdgeCase:
    name: str
    strategy_id: str
    task: str
    validator_key: str


@dataclass
class EdgeCaseResult:
    name: str
    strategy_id: str
    pass_expectation: bool = False
    response_id: str | None = None
    called_tools: list[str] | None = None
    statuses: list[str] | None = None
    output_text: str = ""
    parsed_payload: dict[str, Any] | None = None
    parse_error: str | None = None
    api_error: str | None = None
    event_counts: dict[str, int] | None = None
    validation_ok: bool = False
    validation_message: str = ""

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stress test strategy patch edge cases via real OpenAI + MCP.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--mcp-host", default=DEFAULT_MCP_HOST)
    parser.add_argument("--mcp-port", type=int, default=DEFAULT_MCP_PORT)
    parser.add_argument("--mcp-server-url", default="")
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--output", default="logs/strategy_patch_edge_cases_report.json")
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


def _assert_required_tools_available(
    *,
    server_url: str,
    required_tools: list[str],
    timeout_seconds: float,
) -> None:
    init_result, rpc_session_id = _mcp_rpc_call(
        server_url=server_url,
        method="initialize",
        request_id="init-1",
        params={
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "verify-strategy-patch-edge-cases", "version": "0.1.0"},
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
        session_id=rpc_session_id,
    )
    raw_tools = tools_result.get("tools")
    if not isinstance(raw_tools, list):
        raise RuntimeError(f"Invalid tools/list result: {tools_result}")
    available = {
        str(item.get("name"))
        for item in raw_tools
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    missing = sorted(tool for tool in required_tools if tool not in available)
    if missing:
        raise RuntimeError(
            f"Required MCP tools missing: {missing}. available_tools={sorted(available)}",
        )


async def _prepare_edge_case_strategies() -> tuple[str, dict[str, str]]:
    await db_module.close_postgres()
    await db_module.init_postgres()
    assert db_module.AsyncSessionLocal is not None

    async with db_module.AsyncSessionLocal() as db:
        session_id = await _create_strategy_session(db)
        base_payload = load_strategy_payload(EXAMPLE_PATH)

        ids: dict[str, str] = {}

        case1 = deepcopy(base_payload)
        case1["strategy"]["name"] = "Edge Case 1"
        created1 = await upsert_strategy_dsl(db, session_id=session_id, dsl_payload=case1)
        ids["multi_update"] = str(created1.strategy.id)

        case2 = deepcopy(base_payload)
        case2["strategy"]["name"] = "Edge Case 2 Long Only"
        case2["trade"].pop("short", None)
        created2 = await upsert_strategy_dsl(db, session_id=session_id, dsl_payload=case2)
        ids["add_short"] = str(created2.strategy.id)

        case3 = deepcopy(base_payload)
        case3["strategy"]["name"] = "Edge Case 3 Factor Migration"
        created3 = await upsert_strategy_dsl(db, session_id=session_id, dsl_payload=case3)
        ids["factor_migration"] = str(created3.strategy.id)

        case4 = deepcopy(base_payload)
        case4["strategy"]["name"] = "Edge Case 4 Append Conditions"
        created4 = await upsert_strategy_dsl(db, session_id=session_id, dsl_payload=case4)
        ids["append_filters"] = str(created4.strategy.id)

        await db.commit()

    await db_module.close_postgres()
    return str(session_id), ids


async def _create_strategy_session(db: AsyncSession) -> UUID:
    email = f"verify_patch_edge_{uuid4().hex[:10]}@example.com"
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
    return session.id


async def _load_strategy_payload(strategy_id: str) -> dict[str, Any]:
    await db_module.init_postgres()
    assert db_module.AsyncSessionLocal is not None
    async with db_module.AsyncSessionLocal() as db:
        strategy = await db.get(Strategy, UUID(strategy_id))
        if strategy is None:
            raise LookupError(f"Strategy not found: {strategy_id}")
        payload = strategy.dsl_payload if isinstance(strategy.dsl_payload, dict) else {}
    await db_module.close_postgres()
    return payload


def _collect_refs(node: Any) -> list[str]:
    refs: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "ref" and isinstance(value, str):
                refs.append(value)
            else:
                refs.extend(_collect_refs(value))
    elif isinstance(node, list):
        for item in node:
            refs.extend(_collect_refs(item))
    return refs


def _find_exit_by_type(side: dict[str, Any], exit_type: str) -> dict[str, Any] | None:
    exits = side.get("exits")
    if not isinstance(exits, list):
        return None
    for item in exits:
        if isinstance(item, dict) and item.get("type") == exit_type:
            return item
    return None


def _find_rsi_threshold(side: dict[str, Any], *, op: str) -> float | int | None:
    entry = side.get("entry")
    if not isinstance(entry, dict):
        return None
    condition = entry.get("condition")
    if not isinstance(condition, dict):
        return None
    all_nodes = condition.get("all")
    if not isinstance(all_nodes, list):
        return None
    for node in all_nodes:
        if not isinstance(node, dict):
            continue
        cmp_node = node.get("cmp")
        if not isinstance(cmp_node, dict):
            continue
        left = cmp_node.get("left")
        if not isinstance(left, dict):
            continue
        if left.get("ref") != "rsi_14":
            continue
        if cmp_node.get("op") != op:
            continue
        return cmp_node.get("right")
    return None


def _has_atr_filter(side: dict[str, Any], threshold: float) -> bool:
    entry = side.get("entry")
    if not isinstance(entry, dict):
        return False
    condition = entry.get("condition")
    if not isinstance(condition, dict):
        return False
    all_nodes = condition.get("all")
    if not isinstance(all_nodes, list):
        return False
    for node in all_nodes:
        if not isinstance(node, dict):
            continue
        cmp_node = node.get("cmp")
        if not isinstance(cmp_node, dict):
            continue
        left = cmp_node.get("left")
        if not isinstance(left, dict):
            continue
        if left.get("ref") != "atr_14":
            continue
        if cmp_node.get("op") != "gt":
            continue
        if cmp_node.get("right") == threshold:
            return True
    return False


def _validate_case_payload(*, key: str, payload: dict[str, Any]) -> tuple[bool, str]:
    trade = payload.get("trade")
    if not isinstance(trade, dict):
        return False, "trade block missing"

    if key == "multi_update":
        long_side = trade.get("long")
        short_side = trade.get("short")
        if not isinstance(long_side, dict) or not isinstance(short_side, dict):
            return False, "long/short side missing"
        if payload.get("timeframe") != "4h":
            return False, "timeframe not updated to 4h"
        if _find_rsi_threshold(long_side, op="lt") != 65:
            return False, "long RSI threshold is not 65"
        if _find_rsi_threshold(short_side, op="gt") != 35:
            return False, "short RSI threshold is not 35"
        stop_loss = _find_exit_by_type(long_side, "stop_loss")
        if not isinstance(stop_loss, dict):
            return False, "long stop_loss exit missing"
        stop = stop_loss.get("stop")
        if not isinstance(stop, dict) or stop.get("multiple") != 2.5:
            return False, "long stop_loss atr multiple is not 2.5"
        bracket = _find_exit_by_type(long_side, "bracket_rr")
        if not isinstance(bracket, dict) or bracket.get("risk_reward") != 3.0:
            return False, "long bracket_rr risk_reward is not 3.0"
        sizing = long_side.get("position_sizing")
        if not isinstance(sizing, dict) or sizing.get("pct") != 0.30:
            return False, "long position sizing pct is not 0.30"
        return True, "all multi-update checks passed"

    if key == "add_short":
        short_side = trade.get("short")
        if not isinstance(short_side, dict):
            return False, "short side was not added"
        entry = short_side.get("entry")
        exits = short_side.get("exits")
        sizing = short_side.get("position_sizing")
        if not isinstance(entry, dict) or not isinstance(exits, list) or not isinstance(sizing, dict):
            return False, "short side shape invalid"
        if len(exits) < 3:
            return False, "short exits count < 3"
        if sizing.get("mode") != "pct_equity" or sizing.get("pct") != 0.2:
            return False, "short position sizing mismatch"
        entry_refs = _collect_refs(entry)
        if "ema_9" not in entry_refs or "ema_21" not in entry_refs:
            return False, "short entry refs missing ema_9/ema_21"
        return True, "short side added with expected structure"

    if key == "factor_migration":
        factors = payload.get("factors")
        if not isinstance(factors, dict):
            return False, "factors missing"
        if "ema_34" not in factors:
            return False, "ema_34 not added"
        if "ema_21" in factors:
            return False, "ema_21 still present"
        refs = _collect_refs(trade)
        if any(ref == "ema_21" for ref in refs):
            return False, "ema_21 refs still exist in trade block"
        if not any(ref == "ema_34" for ref in refs):
            return False, "ema_34 refs not found in trade block"
        return True, "factor migration checks passed"

    if key == "append_filters":
        long_side = trade.get("long")
        short_side = trade.get("short")
        if not isinstance(long_side, dict) or not isinstance(short_side, dict):
            return False, "long/short side missing"
        if not _has_atr_filter(long_side, 1.0):
            return False, "long ATR filter not found"
        if not _has_atr_filter(short_side, 1.0):
            return False, "short ATR filter not found"
        return True, "append filter checks passed"

    return False, f"unknown validator key: {key}"


def _build_case_prompt(*, session_id: str, strategy_id: str, task: str) -> str:
    return (
        "You are updating an existing strategy via MCP.\n"
        "Use this strict workflow:\n"
        "1) Call strategy_get_dsl(session_id, strategy_id) first.\n"
        "2) Build minimal RFC 6902 patch operations.\n"
        "3) Call strategy_patch_dsl(session_id, strategy_id, patch_json, expected_version) once.\n"
        "4) expected_version must equal metadata.version from step 1.\n"
        "5) Patch path root is the DSL object itself. "
        "Valid examples: /timeframe, /trade/long/exits/1/stop/multiple.\n"
        "6) NEVER use /dsl_json prefix in patch paths.\n"
        "Return ONLY the JSON result from strategy_patch_dsl.\n\n"
        f"session_id={session_id}\n"
        f"strategy_id={strategy_id}\n"
        f"Task:\n{task}\n"
    )


def _run_case(
    *,
    client: OpenAI,
    case: EdgeCase,
    session_id: str,
    model: str,
    mcp_server_url: str,
    timeout_seconds: float,
) -> EdgeCaseResult:
    result = EdgeCaseResult(
        name=case.name,
        strategy_id=case.strategy_id,
        called_tools=[],
        statuses=[],
        event_counts={},
    )
    counter: Counter[str] = Counter()

    try:
        prompt = _build_case_prompt(
            session_id=session_id,
            strategy_id=case.strategy_id,
            task=case.task,
        )
        with client.responses.stream(
            model=model,
            input=prompt,
            tools=[
                {
                    "type": "mcp",
                    "server_label": "strategy_remote",
                    "server_url": mcp_server_url,
                    "allowed_tools": ["strategy_get_dsl", "strategy_patch_dsl"],
                    "require_approval": "never",
                }
            ],
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
        if not ok_flag:
            result.validation_ok = False
            result.validation_message = "strategy_patch_dsl returned ok=false"
            result.pass_expectation = False
            return result

        names = list(result.called_tools or [])
        first_get = next((i for i, name in enumerate(names) if name == "strategy_get_dsl"), -1)
        first_patch = next((i for i, name in enumerate(names) if name == "strategy_patch_dsl"), -1)
        if first_get == -1 or first_patch == -1 or first_get > first_patch:
            result.validation_ok = False
            result.validation_message = "tool call order mismatch: expected get before patch"
            result.pass_expectation = False
            return result

        payload_after = asyncio.run(_load_strategy_payload(case.strategy_id))
        validation_ok, validation_message = _validate_case_payload(
            key=case.validator_key,
            payload=payload_after,
        )
        result.validation_ok = validation_ok
        result.validation_message = validation_message
        result.pass_expectation = validation_ok
        return result

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
        session_id, strategy_ids = asyncio.run(_prepare_edge_case_strategies())

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

        _assert_required_tools_available(
            server_url=mcp_server_url,
            required_tools=["strategy_get_dsl", "strategy_patch_dsl"],
            timeout_seconds=20.0,
        )

        cases = [
            EdgeCase(
                name="edge_multi_update_6_params",
                strategy_id=strategy_ids["multi_update"],
                validator_key="multi_update",
                task=(
                    "- change timeframe from 1d to 4h\n"
                    "- change long RSI threshold from 70 to 65\n"
                    "- change short RSI threshold from 30 to 35\n"
                    "- change long stop_loss atr multiple from 2.0 to 2.5\n"
                    "- change long bracket_rr risk_reward from 2.0 to 3.0\n"
                    "- change long position sizing pct from 0.25 to 0.30\n"
                ),
            ),
            EdgeCase(
                name="edge_add_full_short_to_long_only",
                strategy_id=strategy_ids["add_short"],
                validator_key="add_short",
                task=(
                    "The strategy is currently long-only. Add trade.short with:\n"
                    "- entry condition: ema_9 cross_below ema_21 AND rsi_14 > 30 "
                    "AND macd_12_26_9.macd_line < macd_12_26_9.signal\n"
                    "- exits: signal_exit cross_above ema_9/ema_21, "
                    "stop_loss atr_multiple atr_14 x2, bracket_rr risk_reward 2 using same stop\n"
                    "- position_sizing: mode pct_equity, pct 0.2\n"
                    "- keep long side unchanged\n"
                ),
            ),
            EdgeCase(
                name="edge_factor_migration_ema21_to_ema34",
                strategy_id=strategy_ids["factor_migration"],
                validator_key="factor_migration",
                task=(
                    "- add factor ema_34 (type ema, period 34, source close)\n"
                    "- remove factor ema_21\n"
                    "- update all trade refs from ema_21 to ema_34\n"
                    "- keep ema_21_typical unchanged\n"
                ),
            ),
            EdgeCase(
                name="edge_append_filters_both_sides",
                strategy_id=strategy_ids["append_filters"],
                validator_key="append_filters",
                task=(
                    "- append one condition to long entry condition.all: atr_14 > 1.0\n"
                    "- append one condition to short entry condition.all: atr_14 > 1.0\n"
                    "- do not change other logic\n"
                ),
            ),
        ]

        client = OpenAI(api_key=api_key)
        results: list[EdgeCaseResult] = []
        for index, case in enumerate(cases, start=1):
            print(
                f"[verify-edge] running case {index}/{len(cases)}: {case.name}",
                file=sys.stderr,
            )
            case_result = _run_case(
                client=client,
                case=case,
                session_id=session_id,
                model=args.model,
                mcp_server_url=mcp_server_url,
                timeout_seconds=args.timeout_seconds,
            )
            results.append(case_result)
            print(
                f"[verify-edge] done case {index}/{len(cases)}: {case.name} "
                f"pass={case_result.pass_expectation} validation={case_result.validation_message}",
                file=sys.stderr,
            )

        passed = sum(1 for item in results if item.pass_expectation)
        report = {
            "started_at_utc": started_at,
            "finished_at_utc": datetime.now(UTC).isoformat(),
            "model": args.model,
            "mcp_server_url": mcp_server_url,
            "session_id": session_id,
            "strategy_ids": strategy_ids,
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
        asyncio.run(db_module.close_postgres())


if __name__ == "__main__":
    raise SystemExit(main())
