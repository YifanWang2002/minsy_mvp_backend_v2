#!/usr/bin/env python3
"""Batch harness for strategy_validate_dsl quality metrics.

Runs >=10 real orchestrator conversations and reports:
1) First-attempt strategy_validate_dsl pass rate.
2) Average strategy_validate_dsl calls needed until first success.
3) Prompt observability for strategy/schema_only turn.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from uuid import UUID, uuid4

# Must be set before importing app modules (settings load at import-time).
os.environ.setdefault("MINSY_SERVICE", "api")
os.environ.setdefault(
    "MINSY_ENV_FILES",
    ",".join(
        [
            "env/.env.secrets",
            "env/.env.common",
            "env/.env.dev",
            "env/.env.dev.api",
            "env/.env.dev.localtest",
        ]
    ),
)

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from apps.api.schemas.requests import ChatSendRequest, ExecutionPolicy
from packages.domain.session.services.openai_stream_service import OpenAIResponsesEventStreamer
from packages.infra.db import session as db_session_module
from packages.infra.db.models.session import Session
from packages.infra.db.models.user import User
from packages.infra.db.session import init_postgres
from tests.test_agents.harness.observable_orchestrator import ObservableChatOrchestrator
from tests.test_agents.harness.observer import TurnObserver


@dataclass(slots=True)
class CaseResult:
    case_id: str
    strategy_request: str
    session_id: str | None
    success: bool
    error: str | None
    first_strategy_turn_number: int | None
    first_strategy_phase_stage: str | None
    strategy_instructions_chars: int | None
    strategy_instructions_has_dsl_spec: bool
    strategy_instructions_has_dsl_schema: bool
    strategy_instructions_is_v2: bool
    strategy_enriched_input_chars: int | None
    strategy_turn_input_tokens: int | None
    strategy_turn_output_tokens: int | None
    strategy_turn_total_tokens: int | None
    validate_call_count_in_strategy_turn: int
    validate_call_statuses_in_strategy_turn: list[str]
    first_validate_status: str | None
    first_validate_pass: bool
    calls_until_first_validate_success: int | None
    strategy_turn_any_validate_success: bool
    conversation_total_tokens: int
    conversation_total_input_tokens: int
    conversation_total_output_tokens: int


STRATEGY_REQUESTS: list[str] = [
    "请生成一个均线趋势跟踪策略：EMA 20 上穿 EMA 50 做多，下穿离场。",
    "请做一个RSI均值回归策略：RSI<30入场，RSI>55止盈，带2%止损。",
    "请做布林带回归：收盘跌破下轨买入，回到中轨卖出。",
    "请做MACD趋势策略：MACD线上穿signal做多，跌破signal平仓。",
    "请做ATR波动突破：突破过去20根高点入场，2ATR止损。",
    "请做短线动量策略：5日收益为正且价格在EMA20上方才开仓。",
    "请做只做空策略：价格跌破EMA50后做空，反穿EMA50平仓。",
    "请做双条件策略：RSI>55且EMA20>EMA50才允许开多，反条件离场。",
    "请做保守趋势策略：仅在日线EMA50向上时，4h级别回踩EMA20入场。",
    "请做风险控制优先策略：固定仓位20%，设置1.5%止损和3%止盈。",
]


def _setup_env() -> None:
    os.environ["MINSY_SERVICE"] = "api"
    os.environ["MINSY_ENV_FILES"] = ",".join(
        [
            "env/.env.secrets",
            "env/.env.common",
            "env/.env.dev",
            "env/.env.dev.api",
            "env/.env.dev.localtest",
        ]
    )


def _extract_validate_metrics(tool_calls: list[dict]) -> tuple[list[str], int | None]:
    statuses: list[str] = []
    calls_until_success: int | None = None
    for idx, call in enumerate(tool_calls, start=1):
        if str(call.get("name", "")).strip() != "strategy_validate_dsl":
            continue
        status = str(call.get("status", "")).strip().lower() or "unknown"
        statuses.append(status)
        if calls_until_success is None and status == "success":
            calls_until_success = len(statuses)
    return statuses, calls_until_success


def _build_seed_strategy_artifacts() -> dict:
    return {
        "kyc": {
            "profile": {
                "trading_years_bucket": "years_3_5",
                "risk_tolerance": "moderate",
                "return_expectation": "return_15_25",
            }
        },
        "pre_strategy": {
            "profile": {
                "target_market": "us_stocks",
                "target_instrument": "AAPL",
                "timeframe": "1d",
                "holding_period": "days",
            },
            "runtime": {
                "instrument_data_status": "ready",
                "instrument_data_symbol": "AAPL",
                "instrument_data_market": "us_stocks",
                "instrument_available_locally": True,
            },
        },
        "strategy": {
            "profile": {},
            "runtime": {},
        },
    }


async def _run_case(
    *,
    db,
    streamer: OpenAIResponsesEventStreamer,
    case_id: str,
    strategy_request: str,
) -> CaseResult:
    user = User(
        id=uuid4(),
        email=f"harness_batch_{uuid4().hex[:10]}@test.com",
        password_hash="test_hash",
        name=f"Harness Batch {case_id}",
        is_active=True,
    )
    db.add(user)
    await db.flush()

    observer = TurnObserver(uuid4(), user.id)
    orchestrator = ObservableChatOrchestrator(db, observer)
    seeded_session = Session(
        user_id=user.id,
        current_phase="strategy",
        status="active",
        artifacts=_build_seed_strategy_artifacts(),
        metadata_={},
    )
    db.add(seeded_session)
    await db.flush()
    session_id: UUID | None = seeded_session.id

    try:
        payload = ChatSendRequest(
            session_id=session_id,
            message=strategy_request,
            execution_policy=ExecutionPolicy(
                model="gpt-5.2",
                reasoning_effort="medium",
            ),
        )
        async for event in orchestrator.handle_message_stream(
            user,
            payload,
            streamer,
            language="zh",
        ):
            if '"type": "stream_start"' in event or '"type":"stream_start"' in event:
                for line in event.splitlines():
                    if not line.startswith("data: "):
                        continue
                    try:
                        data = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue
                    if data.get("type") == "stream_start":
                        raw_sid = data.get("session_id")
                        if isinstance(raw_sid, str) and raw_sid.strip():
                            session_id = UUID(raw_sid)

        conversation = observer.finalize()
        strategy_turn = next((t for t in conversation.turns if t.phase == "strategy"), None)
        if strategy_turn is None:
            return CaseResult(
                case_id=case_id,
                strategy_request=strategy_request,
                session_id=str(session_id) if session_id is not None else None,
                success=False,
                error="No strategy turn reached.",
                first_strategy_turn_number=None,
                first_strategy_phase_stage=None,
                strategy_instructions_chars=None,
                strategy_instructions_has_dsl_spec=False,
                strategy_instructions_has_dsl_schema=False,
                strategy_instructions_is_v2=False,
                strategy_enriched_input_chars=None,
                strategy_turn_input_tokens=None,
                strategy_turn_output_tokens=None,
                strategy_turn_total_tokens=None,
                validate_call_count_in_strategy_turn=0,
                validate_call_statuses_in_strategy_turn=[],
                first_validate_status=None,
                first_validate_pass=False,
                calls_until_first_validate_success=None,
                strategy_turn_any_validate_success=False,
                conversation_total_tokens=conversation.total_tokens,
                conversation_total_input_tokens=conversation.total_input_tokens,
                conversation_total_output_tokens=conversation.total_output_tokens,
            )

        statuses, calls_until_success = _extract_validate_metrics(strategy_turn.mcp_tool_calls)
        first_status = statuses[0] if statuses else None
        first_pass = first_status == "success"
        has_spec = "[DSL SPEC]" in strategy_turn.instructions
        has_schema = "[DSL JSON SCHEMA]" in strategy_turn.instructions
        is_v2 = "Token-Optimized" in strategy_turn.instructions

        return CaseResult(
            case_id=case_id,
            strategy_request=strategy_request,
            session_id=str(session_id) if session_id is not None else None,
            success=True,
            error=None,
            first_strategy_turn_number=strategy_turn.turn_number,
            first_strategy_phase_stage=strategy_turn.phase_stage,
            strategy_instructions_chars=len(strategy_turn.instructions),
            strategy_instructions_has_dsl_spec=has_spec,
            strategy_instructions_has_dsl_schema=has_schema,
            strategy_instructions_is_v2=is_v2,
            strategy_enriched_input_chars=len(strategy_turn.enriched_input),
            strategy_turn_input_tokens=strategy_turn.input_tokens,
            strategy_turn_output_tokens=strategy_turn.output_tokens,
            strategy_turn_total_tokens=strategy_turn.total_tokens,
            validate_call_count_in_strategy_turn=len(statuses),
            validate_call_statuses_in_strategy_turn=statuses,
            first_validate_status=first_status,
            first_validate_pass=first_pass,
            calls_until_first_validate_success=calls_until_success,
            strategy_turn_any_validate_success=(calls_until_success is not None),
            conversation_total_tokens=conversation.total_tokens,
            conversation_total_input_tokens=conversation.total_input_tokens,
            conversation_total_output_tokens=conversation.total_output_tokens,
        )
    except Exception as exc:  # noqa: BLE001
        return CaseResult(
            case_id=case_id,
            strategy_request=strategy_request,
            session_id=str(session_id) if session_id is not None else None,
            success=False,
            error=f"{type(exc).__name__}: {exc}",
            first_strategy_turn_number=None,
            first_strategy_phase_stage=None,
            strategy_instructions_chars=None,
            strategy_instructions_has_dsl_spec=False,
            strategy_instructions_has_dsl_schema=False,
            strategy_instructions_is_v2=False,
            strategy_enriched_input_chars=None,
            strategy_turn_input_tokens=None,
            strategy_turn_output_tokens=None,
            strategy_turn_total_tokens=None,
            validate_call_count_in_strategy_turn=0,
            validate_call_statuses_in_strategy_turn=[],
            first_validate_status=None,
            first_validate_pass=False,
            calls_until_first_validate_success=None,
            strategy_turn_any_validate_success=False,
            conversation_total_tokens=0,
            conversation_total_input_tokens=0,
            conversation_total_output_tokens=0,
        )


def _build_summary(results: list[CaseResult]) -> dict:
    valid_runs = [r for r in results if r.success and r.first_strategy_turn_number is not None]
    if not valid_runs:
        return {
            "total_runs": len(results),
            "valid_runs": 0,
        }

    first_pass_count = sum(1 for r in valid_runs if r.first_validate_pass)
    pass_runs = [r for r in valid_runs if r.calls_until_first_validate_success is not None]
    calls_until_pass = [int(r.calls_until_first_validate_success) for r in pass_runs]
    validate_calls_per_run = [r.validate_call_count_in_strategy_turn for r in valid_runs]
    instructions_chars = [
        int(r.strategy_instructions_chars or 0)
        for r in valid_runs
        if r.strategy_instructions_chars is not None
    ]

    return {
        "total_runs": len(results),
        "valid_runs": len(valid_runs),
        "first_validate_pass_rate": round(first_pass_count / len(valid_runs), 4),
        "first_validate_pass_count": first_pass_count,
        "avg_validate_calls_until_first_success_successful_runs": (
            round(mean(calls_until_pass), 4) if calls_until_pass else None
        ),
        "avg_validate_calls_in_strategy_turn": round(mean(validate_calls_per_run), 4),
        "strategy_turn_any_validate_success_rate": round(len(pass_runs) / len(valid_runs), 4),
        "avg_strategy_instructions_chars": round(mean(instructions_chars), 2),
        "all_strategy_turn_has_dsl_spec": all(r.strategy_instructions_has_dsl_spec for r in valid_runs),
        "all_strategy_turn_has_dsl_schema": all(r.strategy_instructions_has_dsl_schema for r in valid_runs),
        "all_strategy_turn_is_v2": all(r.strategy_instructions_is_v2 for r in valid_runs),
        "total_strategy_turn_tokens": sum(int(r.strategy_turn_total_tokens or 0) for r in valid_runs),
    }


async def main() -> None:
    _setup_env()
    await init_postgres(ensure_schema=False)
    if db_session_module.AsyncSessionLocal is None:
        raise RuntimeError("AsyncSessionLocal unavailable after init_postgres")

    started_at = datetime.now(UTC)
    results: list[CaseResult] = []
    streamer = OpenAIResponsesEventStreamer()

    async with db_session_module.AsyncSessionLocal() as db:
        for idx, req in enumerate(STRATEGY_REQUESTS, start=1):
            case_id = f"case_{idx:02d}"
            print(f"[{case_id}] running ...", flush=True)
            result = await _run_case(
                db=db,
                streamer=streamer,
                case_id=case_id,
                strategy_request=req,
            )
            results.append(result)
            print(
                f"[{case_id}] success={result.success} "
                f"first_pass={result.first_validate_pass} "
                f"validate_calls={result.validate_call_count_in_strategy_turn} "
                f"calls_until_success={result.calls_until_first_validate_success} "
                f"error={result.error}",
                flush=True,
            )

        summary = _build_summary(results)
        ended_at = datetime.now(UTC)
        payload = {
            "started_at": started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "duration_seconds": round((ended_at - started_at).total_seconds(), 3),
            "summary": summary,
            "results": [asdict(r) for r in results],
        }

        out_dir = Path("runtime/reports")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"strategy_validate_batch_{started_at.strftime('%Y%m%d_%H%M%S')}.json"
        out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        print("\n=== SUMMARY ===", flush=True)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        print(f"report_file={out_file}", flush=True)

        await db.rollback()


if __name__ == "__main__":
    asyncio.run(main())
