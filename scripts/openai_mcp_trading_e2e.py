#!/usr/bin/env python3
"""Real OpenAI -> trading MCP end-to-end probe with user-scoped context."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from dotenv import load_dotenv
from openai import APIError, OpenAI
from sqlalchemy import func, select

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mcp.context_auth import MCP_CONTEXT_HEADER, create_mcp_context_token
from src.models import database as db_module
from src.models.deployment import Deployment
from src.models.session import Session as ChatSession
from src.models.user import User

DEFAULT_SERVER_URL = "https://dev.minsyai.com/trading/mcp"
DEFAULT_MODEL = "gpt-5.2"


@dataclass
class ProbeCase:
    tool: str
    args: dict[str, Any]
    assistant_reply: str = ""
    call_status: str = ""
    call_error: str | None = None
    output_ok: bool | None = None
    output_preview: str | None = None


@dataclass
class RuntimeContext:
    user_id: UUID
    session_id: UUID | None
    deployment_id: UUID


def _load_env() -> None:
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        load_dotenv(env_file)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_server_url = (
        os.getenv("MCP_SERVER_URL_TRADING")
        or os.getenv("MCP_SERVER_URL_TRADING_DEV")
        or os.getenv("MCP_SERVER_URL_TRADING_PROD")
        or DEFAULT_SERVER_URL
    )
    parser.add_argument("--server-url", default=default_server_url)
    parser.add_argument("--model", default=os.getenv("OPENAI_RESPONSE_MODEL", DEFAULT_MODEL))
    parser.add_argument("--server-label", default="trading_mcp")
    parser.add_argument("--user-email", default="2@test.com")
    parser.add_argument("--deployment-id", default="")
    parser.add_argument(
        "--artifact-file",
        default="artifacts/openai_mcp_trading_e2e/latest.json",
        help="Output artifact JSON path.",
    )
    parser.add_argument(
        "--show-all-events",
        action="store_true",
        help="Print all stream events; default only prints MCP events.",
    )
    return parser.parse_args()


async def _resolve_runtime_context(*, user_email: str, deployment_id: str) -> RuntimeContext:
    await db_module.init_postgres(ensure_schema=False)
    assert db_module.AsyncSessionLocal is not None

    try:
        async with db_module.AsyncSessionLocal() as db:
            user = await db.scalar(select(User).where(func.lower(User.email) == user_email.strip().lower()))
            if user is None:
                raise RuntimeError(f"user not found by email: {user_email}")

            latest_session_id = await db.scalar(
                select(ChatSession.id)
                .where(ChatSession.user_id == user.id)
                .order_by(ChatSession.updated_at.desc())
                .limit(1),
            )

            resolved_deployment_id: UUID
            if deployment_id.strip():
                try:
                    resolved_deployment_id = UUID(deployment_id.strip())
                except ValueError as exc:
                    raise RuntimeError(f"invalid deployment id: {deployment_id}") from exc
            else:
                latest_deployment_id = await db.scalar(
                    select(Deployment.id)
                    .where(Deployment.user_id == user.id)
                    .order_by(Deployment.updated_at.desc())
                    .limit(1),
                )
                if latest_deployment_id is None:
                    raise RuntimeError(
                        f"no deployment found for user {user_email}; run e2e setup first",
                    )
                resolved_deployment_id = latest_deployment_id

            return RuntimeContext(
                user_id=user.id,
                session_id=latest_session_id,
                deployment_id=resolved_deployment_id,
            )
    finally:
        await db_module.close_postgres()


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "model_dump"):
        try:
            return _to_jsonable(value.model_dump(mode="json", exclude_none=True, warnings=False))
        except Exception:  # noqa: BLE001
            pass
    return str(value)


def _coerce_payload(event: Any) -> dict[str, Any]:
    payload = _to_jsonable(event)
    return payload if isinstance(payload, dict) else {"type": str(getattr(event, "type", "unknown"))}


def _truncate(value: Any, *, max_len: int = 400) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if len(text) <= max_len:
        return text
    return text[:max_len] + "...<truncated>"


def _extract_output_payload(item: dict[str, Any]) -> dict[str, Any] | None:
    output = item.get("output")
    if isinstance(output, str):
        text = output.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None

    if isinstance(output, list):
        for chunk in output:
            if not isinstance(chunk, dict):
                continue
            text = chunk.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            try:
                parsed = json.loads(text.strip())
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue
    return None


def _build_prompt(*, tool: str, args: dict[str, Any]) -> str:
    payload = json.dumps(args, ensure_ascii=True, separators=(",", ":"))
    return (
        f"Call MCP tool {tool} exactly once with JSON arguments {payload}. "
        "After receiving the tool result, summarize whether it succeeded in one sentence."
    )


def _print_event(payload: dict[str, Any], *, show_all_events: bool) -> None:
    event_type = str(payload.get("type", "unknown"))
    if not show_all_events and "mcp" not in event_type.lower():
        return
    print(f"[event] {event_type}")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _run_case(
    *,
    client: OpenAI,
    model: str,
    server_url: str,
    server_label: str,
    headers: dict[str, str],
    case: ProbeCase,
    show_all_events: bool,
) -> ProbeCase:
    tool_def: dict[str, Any] = {
        "type": "mcp",
        "server_label": server_label,
        "server_url": server_url,
        "require_approval": "never",
        "headers": headers,
        "allowed_tools": [case.tool],
    }
    call_seen = False

    try:
        with client.responses.stream(
            model=model,
            input=_build_prompt(tool=case.tool, args=case.args),
            tools=[tool_def],
            tool_choice={"type": "mcp", "server_label": server_label, "name": case.tool},
            reasoning={"effort": "low"},
        ) as stream:
            for event in stream:
                payload = _coerce_payload(event)
                _print_event(payload, show_all_events=show_all_events)

                if str(payload.get("type", "")) not in {"response.output_item.added", "response.output_item.done"}:
                    continue
                item = payload.get("item")
                if not isinstance(item, dict):
                    continue
                if str(item.get("type", "")).strip().lower() != "mcp_call":
                    continue
                if str(item.get("name", "")).strip() != case.tool:
                    continue
                call_seen = True
                case.call_status = str(item.get("status", "")).strip()
                case.call_error = _truncate(item.get("error"), max_len=700)
                output_payload = _extract_output_payload(item)
                if output_payload is not None:
                    case.output_ok = output_payload.get("ok") if isinstance(output_payload.get("ok"), bool) else None
                case.output_preview = _truncate(item.get("output"), max_len=600)

            final_response = stream.get_final_response()
            output_text = getattr(final_response, "output_text", "")
            case.assistant_reply = output_text.strip() if isinstance(output_text, str) else ""
    except APIError as exc:
        case.call_status = "api_error"
        case.call_error = _truncate(f"{type(exc).__name__}: {exc}", max_len=900)
    except Exception as exc:  # noqa: BLE001
        case.call_status = "client_error"
        case.call_error = _truncate(f"{type(exc).__name__}: {exc}", max_len=900)

    if not call_seen and case.call_status == "":
        case.call_status = "missing_call"
        case.call_error = case.call_error or "No MCP call captured for forced tool choice."
    return case


def main() -> int:
    _load_env()
    args = _parse_args()

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("ERROR: OPENAI_API_KEY is missing.")
        return 2

    resolved = asyncio.run(
        _resolve_runtime_context(
            user_email=args.user_email,
            deployment_id=args.deployment_id,
        ),
    )
    token = create_mcp_context_token(
        user_id=resolved.user_id,
        session_id=resolved.session_id,
        phase="deployment",
        request_id="openai_mcp_trading_e2e",
    )
    mcp_headers = {MCP_CONTEXT_HEADER: token}

    cases: list[ProbeCase] = [
        ProbeCase(tool="trading_ping", args={}),
        ProbeCase(tool="trading_capabilities", args={}),
        ProbeCase(tool="trading_list_deployments", args={"status": "", "limit": 20}),
        ProbeCase(
            tool="trading_get_orders",
            args={"deployment_id": str(resolved.deployment_id), "limit": 20},
        ),
        ProbeCase(
            tool="trading_get_positions",
            args={"deployment_id": str(resolved.deployment_id), "limit": 20},
        ),
        ProbeCase(tool="trading_start_deployment", args={"deployment_id": str(resolved.deployment_id)}),
        ProbeCase(tool="trading_pause_deployment", args={"deployment_id": str(resolved.deployment_id)}),
        ProbeCase(tool="trading_stop_deployment", args={"deployment_id": str(resolved.deployment_id)}),
    ]

    client = OpenAI(api_key=api_key)
    for case in cases:
        print(f"\n=== running {case.tool} ===")
        _run_case(
            client=client,
            model=args.model,
            server_url=args.server_url,
            server_label=args.server_label,
            headers=mcp_headers,
            case=case,
            show_all_events=args.show_all_events,
        )

    summary: dict[str, Any] = {
        "started_at": datetime.now(UTC).isoformat(),
        "server_url": args.server_url,
        "model": args.model,
        "user_email": args.user_email,
        "user_id": str(resolved.user_id),
        "session_id": str(resolved.session_id) if resolved.session_id is not None else None,
        "deployment_id": str(resolved.deployment_id),
        "cases": [asdict(case) for case in cases],
    }

    all_ok = all(
        case.call_status == "completed" and case.call_error in {None, ""} and case.output_ok is True for case in cases
    )
    summary["all_ok"] = all_ok

    artifact = Path(args.artifact_file).resolve()
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")

    print("\n=== summary ===")
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    print(f"artifact={artifact}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
