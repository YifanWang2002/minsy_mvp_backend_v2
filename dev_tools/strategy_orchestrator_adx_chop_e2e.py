#!/usr/bin/env python
"""Strategy-phase orchestrator E2E probes for ADX/CHOP range-market requests.

This script drives the real strategy-stage orchestrator path (without using
strategy_failure_diagnostics.py), and exports:
- instructions/enriched_input/tools actually sent to model
- prompt + skills files used for strategy/schema_only
- MCP endpoint exposure
- per-call strategy_validate_dsl arguments/output (including raw DSL input)
- persisted chat records in DB
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import re
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import select


def _detect_backend_dir() -> Path:
    candidate = Path(__file__).resolve().parent
    for path in (candidate, *candidate.parents):
        if (path / "apps").is_dir() and (path / "packages").is_dir():
            return path
    raise RuntimeError("Cannot detect backend root (expected apps/ and packages/).")


BACKEND_DIR = _detect_backend_dir()
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


def _setup_env() -> None:
    os.environ.setdefault("MINSY_SERVICE", "api")
    os.environ.setdefault(
        "MINSY_ENV_FILES",
        ",".join([
            "env/.env.secrets",
            "env/.env.common",
            "env/.env.dev",
            "env/.env.dev.api",
        ]),
    )
    for key in (
        "ALL_PROXY",
        "all_proxy",
        "HTTP_PROXY",
        "http_proxy",
        "HTTPS_PROXY",
        "https_proxy",
    ):
        os.environ.pop(key, None)


_setup_env()

from apps.api.orchestration import ChatOrchestrator
from apps.api.orchestration.openai_stream_service import OpenAIResponsesEventStreamer
from apps.api.orchestration.types import _TurnPostProcessResult, _TurnPreparation, _TurnStreamState
from apps.api.schemas.requests import ChatSendRequest
from packages.infra.db.models.session import Message, Session
from packages.infra.db.models.user import User
from packages.infra.db import session as db_session_module
from packages.infra.db.session import init_postgres
from packages.shared_settings.schema.settings import settings


_SSE_DATA_PATTERN = re.compile(r"^data:\s*(.+)$", flags=re.MULTILINE)


class ProbeChatOrchestrator(ChatOrchestrator):
    """Capture one-turn preparation/stream/postprocess internals for diagnostics."""

    def __init__(self, db: Any) -> None:
        super().__init__(db)
        self.capture: dict[str, Any] = {}

    async def _prepare_turn_context(self, **kwargs: Any) -> _TurnPreparation:
        preparation = await super()._prepare_turn_context(**kwargs)
        self.capture["phase_before"] = preparation.phase_before
        self.capture["phase_stage"] = preparation.ctx.runtime_policy.phase_stage
        self.capture["phase_turn_count"] = preparation.phase_turn_count
        self.capture["instructions"] = preparation.prompt.instructions
        self.capture["enriched_input"] = preparation.prompt.enriched_input
        self.capture["tools"] = copy.deepcopy(preparation.tools)
        self.capture["tool_choice"] = copy.deepcopy(preparation.prompt.tool_choice)
        self.capture["request_model"] = preparation.prompt.model
        self.capture["request_reasoning"] = copy.deepcopy(preparation.prompt.reasoning)
        self.capture["request_max_output_tokens"] = preparation.prompt.max_output_tokens
        return preparation

    async def _stream_openai_and_collect(self, **kwargs: Any):  # type: ignore[override]
        stream_state: _TurnStreamState = kwargs["stream_state"]
        async for event in super()._stream_openai_and_collect(**kwargs):
            yield event
        self.capture["raw_response_text"] = stream_state.full_text
        self.capture["completed_model"] = stream_state.completed_model
        self.capture["completed_usage"] = copy.deepcopy(stream_state.completed_usage)
        self.capture["stream_error_message"] = stream_state.stream_error_message
        self.capture["stream_error_detail"] = copy.deepcopy(stream_state.stream_error_detail)
        self.capture["mcp_call_records"] = copy.deepcopy(stream_state.mcp_call_records)
        self.capture["mcp_call_order"] = list(stream_state.mcp_call_order)

    async def _post_process_turn(self, **kwargs: Any) -> _TurnPostProcessResult:
        result = await super()._post_process_turn(**kwargs)
        self.capture["cleaned_text"] = result.cleaned_text
        self.capture["assistant_text"] = result.assistant_text
        self.capture["missing_fields"] = list(result.missing_fields)
        self.capture["transition_from_phase"] = result.transition_from_phase
        self.capture["transition_to_phase"] = result.transition_to_phase
        self.capture["persisted_tool_calls"] = copy.deepcopy(result.persisted_tool_calls)
        return result


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _json_load_if_possible(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def _collect_prompt_skill_files() -> dict[str, str]:
    strategy_skills_dir = BACKEND_DIR / "apps" / "api" / "agents" / "skills"
    strategy_dir = strategy_skills_dir / "strategy"
    assets_dir = BACKEND_DIR / "packages" / "domain" / "strategy" / "assets"
    files = [
        strategy_dir / "skills.md",
        strategy_dir / "stages" / "schema_only.md",
        strategy_skills_dir / "utils" / "skills.md",
        assets_dir / "DSL_SPEC.md",
        assets_dir / "strategy_dsl_schema.json",
    ]
    output: dict[str, str] = {}
    for path in files:
        if path.is_file():
            output[str(path)] = _read_text(path)
    return output


def _collect_indicator_skill_files() -> dict[str, str]:
    indicator_dir = (
        BACKEND_DIR / "apps" / "mcp" / "domains" / "strategy" / "skills" / "indicators"
    )
    output: dict[str, str] = {}
    if not indicator_dir.is_dir():
        return output
    for path in sorted(indicator_dir.glob("*.md")):
        output[str(path)] = _read_text(path)
    return output


def _extract_validate_calls(mcp_calls: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for call in mcp_calls:
        if str(call.get("name", "")).strip() != "strategy_validate_dsl":
            continue
        arguments = _json_load_if_possible(call.get("arguments"))
        parsed_arguments = arguments if isinstance(arguments, dict) else {"raw": arguments}
        dsl_json = parsed_arguments.get("dsl_json")
        dsl_parsed = _json_load_if_possible(dsl_json)
        raw_output = _json_load_if_possible(call.get("output"))
        parsed_output = raw_output if isinstance(raw_output, dict) else {"raw": raw_output}

        output.append(
            {
                "call_id": call.get("id") or call.get("call_id"),
                "status": call.get("status"),
                "arguments": parsed_arguments,
                "dsl_json_raw": dsl_json,
                "dsl_json_parsed": dsl_parsed,
                "output": parsed_output,
                "error": call.get("error"),
            }
        )
    return output


def _extract_sse_payloads(events: list[str]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for raw in events:
        for match in _SSE_DATA_PATTERN.findall(raw):
            try:
                parsed = json.loads(match)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                output.append(parsed)
    return output


async def _load_chat_messages(db: Any, session_id: UUID) -> list[dict[str, Any]]:
    stmt = (
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(Message.created_at.asc(), Message.id.asc())
    )
    rows = (await db.execute(stmt)).scalars().all()
    messages: list[dict[str, Any]] = []
    for row in rows:
        messages.append(
            {
                "id": str(row.id),
                "role": row.role,
                "phase": row.phase,
                "content": row.content,
                "response_id": row.response_id,
                "tool_calls": row.tool_calls,
                "token_usage": row.token_usage,
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }
        )
    return messages


def _build_strategy_artifacts() -> dict[str, Any]:
    return {
        "kyc": {
            "profile": {
                "trading_years_bucket": "years_3_5",
                "risk_tolerance": "moderate",
                "return_expectation": "return_15_25",
            },
            "missing_fields": [],
        },
        "pre_strategy": {
            "profile": {
                "market": "US",
                "instrument": "stock",
                "tickers": ["AAPL"],
                "tickers_csv": "AAPL",
                "timeframe": "1d",
                "holding_period": "days_3_7",
            },
            "runtime": {
                "instrument_data_status": "ready",
                "instrument_data_symbol": "AAPL",
                "instrument_data_market": "US",
                "instrument_available_locally": True,
            },
            "missing_fields": [],
        },
        "strategy": {
            "profile": {},
            "missing_fields": ["strategy_id"],
        },
    }


async def _run_single_probe(
    *,
    db: Any,
    user: User,
    streamer: OpenAIResponsesEventStreamer,
    language: str,
    label: str,
    user_prompt: str,
) -> dict[str, Any]:
    session = Session(
        id=uuid4(),
        user_id=user.id,
        current_phase="strategy",
        status="active",
        artifacts=_build_strategy_artifacts(),
    )
    db.add(session)
    await db.commit()

    orchestrator = ProbeChatOrchestrator(db)
    payload = ChatSendRequest(
        session_id=session.id,
        message=user_prompt,
    )

    sse_events: list[str] = []
    async for event in orchestrator.handle_message_stream(
        user,
        payload,
        streamer,
        language=language,
    ):
        sse_events.append(event)

    sse_payloads = _extract_sse_payloads(sse_events)
    mcp_order = list(orchestrator.capture.get("mcp_call_order") or [])
    mcp_records = dict(orchestrator.capture.get("mcp_call_records") or {})
    mcp_tool_calls = [mcp_records[call_id] for call_id in mcp_order if call_id in mcp_records]
    validate_calls = _extract_validate_calls(mcp_tool_calls)
    any_validate_success = any(
        isinstance(item.get("output"), dict) and item["output"].get("ok") is True
        for item in validate_calls
    )

    messages = await _load_chat_messages(db, session.id)

    return {
        "label": label,
        "session_id": str(session.id),
        "user_prompt": user_prompt,
        "phase": orchestrator.capture.get("phase_before"),
        "phase_stage": orchestrator.capture.get("phase_stage"),
        "phase_turn_count": orchestrator.capture.get("phase_turn_count"),
        "model": orchestrator.capture.get("completed_model")
        or orchestrator.capture.get("request_model"),
        "token_usage": orchestrator.capture.get("completed_usage"),
        "stream_error_message": orchestrator.capture.get("stream_error_message"),
        "stream_error_detail": orchestrator.capture.get("stream_error_detail"),
        "instructions": orchestrator.capture.get("instructions"),
        "enriched_input": orchestrator.capture.get("enriched_input"),
        "tools_offered": orchestrator.capture.get("tools"),
        "tool_choice": orchestrator.capture.get("tool_choice"),
        "assistant_cleaned_text": orchestrator.capture.get("cleaned_text"),
        "assistant_raw_text": orchestrator.capture.get("raw_response_text"),
        "mcp_tool_calls": mcp_tool_calls,
        "validate_calls": validate_calls,
        "validate_success": any_validate_success,
        "persisted_tool_calls": orchestrator.capture.get("persisted_tool_calls"),
        "missing_fields_after_turn": orchestrator.capture.get("missing_fields"),
        "transition_from_phase": orchestrator.capture.get("transition_from_phase"),
        "transition_to_phase": orchestrator.capture.get("transition_to_phase"),
        "sse_payloads": sse_payloads,
        "chat_messages": messages,
    }


async def run(args: argparse.Namespace) -> Path:
    await init_postgres(ensure_schema=True)
    session_maker = db_session_module.AsyncSessionLocal
    if session_maker is None:
        raise RuntimeError("AsyncSessionLocal is not initialized.")

    probes = [
        (
            "adx_range_market",
            (
                "请直接在 strategy 阶段生成一个完整可运行的 DSL 策略。"
                "目标是美股 AAPL 日线，在震荡市场表现更好，必须使用 ADX 做过滤。"
                "请先调用 get_indicator_catalog，再调用 strategy_validate_dsl；"
                "若失败请按错误自动修复直到通过。"
            ),
        ),
        (
            "chop_range_market",
            (
                "继续给我一个震荡市策略，这次必须使用 CHOP 过滤。"
                "请直接给出完整 DSL，并在本轮中调用 strategy_validate_dsl 直到通过。"
            ),
        ),
    ]

    async with session_maker() as db:
        user_stmt = select(User).where(User.email == args.email).limit(1)
        user = (await db.execute(user_stmt)).scalars().first()
        if user is None:
            raise RuntimeError(f"User not found: {args.email}")
        user_id = user.id

        streamer = OpenAIResponsesEventStreamer()
        scenario_reports: list[dict[str, Any]] = []
        for label, prompt in probes:
            scenario_report = await _run_single_probe(
                db=db,
                user=user,
                streamer=streamer,
                language=args.language,
                label=label,
                user_prompt=prompt,
            )
            scenario_reports.append(scenario_report)

    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "user": {
            "email": args.email,
            "id": str(user_id),
        },
        "runtime": {
            "strategy_mcp_server_url": settings.strategy_mcp_server_url,
            "backtest_mcp_server_url": settings.backtest_mcp_server_url,
            "market_data_mcp_server_url": settings.market_data_mcp_server_url,
            "openai_response_model": settings.openai_response_model,
        },
        "prompt_skill_files_used": _collect_prompt_skill_files(),
        "mcp_indicator_skill_files": _collect_indicator_skill_files(),
        "scenarios": scenario_reports,
    }

    out_dir = (BACKEND_DIR / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%SZ")
    out_path = out_dir / f"strategy_orchestrator_adx_chop_e2e_{timestamp}.json"
    out_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run strategy-phase orchestrator E2E probes for ADX/CHOP."
    )
    parser.add_argument(
        "--email",
        default="7@test.com",
        help="Target user email in local DB.",
    )
    parser.add_argument(
        "--language",
        default="zh",
        help="Conversation language (default: zh).",
    )
    parser.add_argument(
        "--output-dir",
        default="runtime/diag",
        help="Output directory relative to backend root.",
    )
    return parser.parse_args()


async def _main() -> None:
    args = _parse_args()
    out_path = await run(args)
    print(f"[OK] E2E report written: {out_path}")


if __name__ == "__main__":
    asyncio.run(_main())
