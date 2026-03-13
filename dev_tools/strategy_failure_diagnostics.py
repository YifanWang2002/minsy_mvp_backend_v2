#!/usr/bin/env python3
"""Diagnose strategy generation failures with grouped prompt difficulty tests.

Groups:
1) Minimal requirement prompts x3
2) Indicator-specific prompts x3
3) Complex prompts x4

Outputs:
- Real orchestrator/model evidence
- Offered strategy tool list (including factor catalog tools)
- strategy_validate_dsl first-pass rate and call counts
- Failure reason hints from MCP outputs
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from uuid import UUID, uuid4

# Settings load at import-time; initialize env first.
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


INDICATOR_TOOL_NAMES: frozenset[str] = frozenset(
    {"get_indicator_catalog", "get_indicator_detail"}
)


@dataclass(slots=True)
class PromptCase:
    case_id: str
    group: str
    message: str


@dataclass(slots=True)
class CaseDiagnosis:
    case_id: str
    group: str
    message: str
    success: bool
    error: str | None
    session_id: str | None
    reached_strategy: bool
    phase_stage: str | None
    response_model: str | None
    response_id: str | None
    stream_error: str | None
    used_real_gpt52_model: bool
    offered_tools: list[str]
    offered_has_indicator_catalog: bool
    offered_has_indicator_detail: bool
    called_tools: list[str]
    indicator_tool_call_count: int
    validate_call_count: int
    validate_statuses: list[str]
    first_validate_pass: bool
    any_validate_success: bool
    calls_until_first_validate_success: int | None
    validate_failure_reasons: list[str]
    strategy_id_created: str | None
    strategy_turn_tokens: int | None
    strategy_turn_input_tokens: int | None
    strategy_turn_output_tokens: int | None
    root_cause_hint: str


def _build_cases() -> list[PromptCase]:
    return [
        PromptCase("g1_01", "simple", "请你随便帮我做一个策略。"),
        PromptCase("g1_02", "simple", "不设限制，帮我随便做一个可运行策略。"),
        PromptCase("g1_03", "simple", "请你随便生成一个交易策略。"),
        PromptCase("g2_01", "indicator", "请帮我做一个用EMA的趋势策略，日线AAPL。"),
        PromptCase("g2_02", "indicator", "请帮我做一个用ATR止损的趋势策略，日线AAPL。"),
        PromptCase("g2_03", "indicator", "请帮我做一个用SMA均线金叉死叉的策略，日线AAPL。"),
        PromptCase(
            "g3_01",
            "complex",
            "请做一个多空双向策略：EMA20/EMA60判趋势，ATR做止损，1:2盈亏比，4小时级别。",
        ),
        PromptCase(
            "g3_02",
            "complex",
            "请做一个稳健策略：仅在日线趋势向上时开多，入场触发放在1小时回踩，要求仓位20%。",
        ),
        PromptCase(
            "g3_03",
            "complex",
            "请做一个条件较多的策略：SMA趋势过滤 + RSI动量确认 + ATR止损，禁止连续反向开仓。",
        ),
        PromptCase(
            "g3_04",
            "complex",
            "请构建一个可回测的完整策略，包含entry/exit/position_sizing，并明确stop_loss与take_profit。",
        ),
    ]


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


def _flatten_offered_tools(strategy_turn) -> list[str]:
    output: list[str] = []
    for tool in strategy_turn.tools:
        if not isinstance(tool, dict):
            continue
        allowed = tool.get("allowed_tools")
        if not isinstance(allowed, list):
            continue
        for name in allowed:
            text = str(name).strip()
            if text and text not in output:
                output.append(text)
    return output


def _extract_validate_calls(strategy_turn) -> list[dict]:
    return [
        item
        for item in strategy_turn.mcp_tool_calls
        if str(item.get("name", "")).strip() == "strategy_validate_dsl"
    ]


def _extract_indicator_call_count(strategy_turn) -> int:
    return sum(
        1
        for item in strategy_turn.mcp_tool_calls
        if str(item.get("name", "")).strip() in INDICATOR_TOOL_NAMES
    )


def _extract_reason_text(call: dict) -> str | None:
    def normalize(value) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            text = value.strip()
            return text or None
        try:
            text = json.dumps(value, ensure_ascii=False)
        except TypeError:
            text = str(value)
        text = text.strip()
        return text or None

    candidates: list[str | None] = [normalize(call.get("error"))]
    output = call.get("output")
    output_obj = None
    if isinstance(output, dict):
        output_obj = output
    elif isinstance(output, str):
        try:
            parsed = json.loads(output)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            output_obj = parsed
    if isinstance(output_obj, dict):
        for key in ("error", "message", "detail", "description", "code"):
            candidates.append(normalize(output_obj.get(key)))
        data = output_obj.get("data")
        if isinstance(data, dict):
            for key in ("error", "message", "detail"):
                candidates.append(normalize(data.get(key)))
        issues = output_obj.get("issues")
        candidates.append(normalize(issues))

    for item in candidates:
        if isinstance(item, str) and item:
            return item[:280]
    return None


def _classify_root_cause(
    *,
    reached_strategy: bool,
    stream_error: str | None,
    offered_has_indicator_catalog: bool,
    validate_call_count: int,
    any_validate_success: bool,
    indicator_tool_call_count: int,
) -> str:
    if not reached_strategy:
        return "未进入 strategy 阶段。"
    if stream_error:
        return f"OpenAI 流错误：{stream_error}"
    if not offered_has_indicator_catalog:
        return "strategy toollist 未开放因子目录工具。"
    if validate_call_count <= 0:
        return "模型未调用 strategy_validate_dsl。"
    if any_validate_success:
        return "validate 至少成功一次。"
    if indicator_tool_call_count <= 0:
        return "未调用因子目录工具就反复 validate，导致持续失败。"
    return "已调用因子工具但 validate 仍持续失败，需看 validate 输出错误。"


async def _run_case(
    *,
    db,
    streamer: OpenAIResponsesEventStreamer,
    case: PromptCase,
) -> CaseDiagnosis:
    user = User(
        id=uuid4(),
        email=f"strategy_diag_{uuid4().hex[:10]}@test.com",
        password_hash="test_hash",
        name=f"StrategyDiag-{case.case_id}",
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
            message=case.message,
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
            return CaseDiagnosis(
                case_id=case.case_id,
                group=case.group,
                message=case.message,
                success=False,
                error="No strategy turn reached.",
                session_id=str(session_id) if session_id is not None else None,
                reached_strategy=False,
                phase_stage=None,
                response_model=None,
                response_id=None,
                stream_error=None,
                used_real_gpt52_model=False,
                offered_tools=[],
                offered_has_indicator_catalog=False,
                offered_has_indicator_detail=False,
                called_tools=[],
                indicator_tool_call_count=0,
                validate_call_count=0,
                validate_statuses=[],
                first_validate_pass=False,
                any_validate_success=False,
                calls_until_first_validate_success=None,
                validate_failure_reasons=[],
                strategy_id_created=None,
                strategy_turn_tokens=None,
                strategy_turn_input_tokens=None,
                strategy_turn_output_tokens=None,
                root_cause_hint="未进入 strategy 阶段。",
            )

        offered_tools = _flatten_offered_tools(strategy_turn)
        validate_calls = _extract_validate_calls(strategy_turn)
        validate_statuses = [
            str(call.get("status", "")).strip().lower() or "unknown"
            for call in validate_calls
        ]
        first_validate_pass = bool(validate_statuses and validate_statuses[0] == "success")
        calls_until_success = None
        for idx, status in enumerate(validate_statuses, start=1):
            if status == "success":
                calls_until_success = idx
                break
        any_validate_success = calls_until_success is not None
        reasons = [
            text
            for text in (_extract_reason_text(call) for call in validate_calls)
            if isinstance(text, str) and text
        ]
        dedup_reasons: list[str] = []
        for text in reasons:
            if text not in dedup_reasons:
                dedup_reasons.append(text)

        strategy_profile = {}
        if isinstance(strategy_turn.artifacts_after, dict):
            block = strategy_turn.artifacts_after.get("strategy")
            if isinstance(block, dict):
                profile = block.get("profile")
                if isinstance(profile, dict):
                    strategy_profile = profile
        strategy_id_created = str(strategy_profile.get("strategy_id", "")).strip() or None

        called_tools = [
            str(item.get("name", "")).strip()
            for item in strategy_turn.mcp_tool_calls
            if str(item.get("name", "")).strip()
        ]
        indicator_call_count = _extract_indicator_call_count(strategy_turn)
        stream_error = (
            str(strategy_turn.stream_error).strip()
            if isinstance(strategy_turn.stream_error, str) and strategy_turn.stream_error.strip()
            else None
        )
        root_cause = _classify_root_cause(
            reached_strategy=True,
            stream_error=stream_error,
            offered_has_indicator_catalog=("get_indicator_catalog" in offered_tools),
            validate_call_count=len(validate_calls),
            any_validate_success=any_validate_success,
            indicator_tool_call_count=indicator_call_count,
        )
        model = str(strategy_turn.model or "").strip() or None

        return CaseDiagnosis(
            case_id=case.case_id,
            group=case.group,
            message=case.message,
            success=True,
            error=None,
            session_id=str(session_id) if session_id is not None else None,
            reached_strategy=True,
            phase_stage=strategy_turn.phase_stage,
            response_model=model,
            response_id=str(strategy_turn.response_id or "").strip() or None,
            stream_error=stream_error,
            used_real_gpt52_model=bool(model and model.startswith("gpt-5.2")),
            offered_tools=offered_tools,
            offered_has_indicator_catalog=("get_indicator_catalog" in offered_tools),
            offered_has_indicator_detail=("get_indicator_detail" in offered_tools),
            called_tools=called_tools,
            indicator_tool_call_count=indicator_call_count,
            validate_call_count=len(validate_calls),
            validate_statuses=validate_statuses,
            first_validate_pass=first_validate_pass,
            any_validate_success=any_validate_success,
            calls_until_first_validate_success=calls_until_success,
            validate_failure_reasons=dedup_reasons[:5],
            strategy_id_created=strategy_id_created,
            strategy_turn_tokens=strategy_turn.total_tokens,
            strategy_turn_input_tokens=strategy_turn.input_tokens,
            strategy_turn_output_tokens=strategy_turn.output_tokens,
            root_cause_hint=root_cause,
        )
    except Exception as exc:  # noqa: BLE001
        return CaseDiagnosis(
            case_id=case.case_id,
            group=case.group,
            message=case.message,
            success=False,
            error=f"{type(exc).__name__}: {exc}",
            session_id=str(session_id) if session_id is not None else None,
            reached_strategy=False,
            phase_stage=None,
            response_model=None,
            response_id=None,
            stream_error=None,
            used_real_gpt52_model=False,
            offered_tools=[],
            offered_has_indicator_catalog=False,
            offered_has_indicator_detail=False,
            called_tools=[],
            indicator_tool_call_count=0,
            validate_call_count=0,
            validate_statuses=[],
            first_validate_pass=False,
            any_validate_success=False,
            calls_until_first_validate_success=None,
            validate_failure_reasons=[],
            strategy_id_created=None,
            strategy_turn_tokens=None,
            strategy_turn_input_tokens=None,
            strategy_turn_output_tokens=None,
            root_cause_hint=f"运行异常：{type(exc).__name__}",
        )


def _build_summary(results: list[CaseDiagnosis]) -> dict:
    valid = [r for r in results if r.success and r.reached_strategy]
    if not valid:
        return {"total_runs": len(results), "valid_runs": 0}

    def summarize(group: str) -> dict:
        subset = [r for r in valid if r.group == group]
        if not subset:
            return {"runs": 0}
        first_pass = sum(1 for r in subset if r.first_validate_pass)
        any_pass = sum(1 for r in subset if r.any_validate_success)
        validate_counts = [r.validate_call_count for r in subset]
        return {
            "runs": len(subset),
            "first_validate_pass_rate": round(first_pass / len(subset), 4),
            "any_validate_success_rate": round(any_pass / len(subset), 4),
            "avg_validate_calls": round(mean(validate_counts), 4),
        }

    all_first = sum(1 for r in valid if r.first_validate_pass)
    all_any = sum(1 for r in valid if r.any_validate_success)
    root_causes = Counter(r.root_cause_hint for r in valid)
    models = Counter(r.response_model or "unknown" for r in valid)

    return {
        "total_runs": len(results),
        "valid_runs": len(valid),
        "all_used_real_gpt52_model": all(r.used_real_gpt52_model for r in valid),
        "all_offered_indicator_catalog_tool": all(r.offered_has_indicator_catalog for r in valid),
        "all_offered_indicator_detail_tool": all(r.offered_has_indicator_detail for r in valid),
        "global_first_validate_pass_rate": round(all_first / len(valid), 4),
        "global_any_validate_success_rate": round(all_any / len(valid), 4),
        "global_avg_validate_calls": round(mean(r.validate_call_count for r in valid), 4),
        "global_avg_strategy_tokens": round(mean(r.strategy_turn_tokens or 0 for r in valid), 2),
        "groups": {
            "simple": summarize("simple"),
            "indicator": summarize("indicator"),
            "complex": summarize("complex"),
        },
        "response_models": dict(models),
        "root_cause_counts": dict(root_causes),
    }


async def main() -> None:
    started_at = datetime.now(UTC)
    await init_postgres(ensure_schema=False)
    if db_session_module.AsyncSessionLocal is None:
        raise RuntimeError("AsyncSessionLocal unavailable after init_postgres")

    cases = _build_cases()
    streamer = OpenAIResponsesEventStreamer()
    results: list[CaseDiagnosis] = []

    async with db_session_module.AsyncSessionLocal() as db:
        for idx, case in enumerate(cases, start=1):
            print(f"[{idx:02d}/{len(cases)}] {case.case_id} ({case.group}) running ...", flush=True)
            diag = await _run_case(
                db=db,
                streamer=streamer,
                case=case,
            )
            results.append(diag)
            print(
                f"[{case.case_id}] reached_strategy={diag.reached_strategy} "
                f"model={diag.response_model} "
                f"offered_catalog={diag.offered_has_indicator_catalog} "
                f"validate_calls={diag.validate_call_count} "
                f"first_pass={diag.first_validate_pass} "
                f"any_pass={diag.any_validate_success}",
                flush=True,
            )

        summary = _build_summary(results)
        ended_at = datetime.now(UTC)
        report = {
            "started_at": started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "duration_seconds": round((ended_at - started_at).total_seconds(), 3),
            "summary": summary,
            "results": [asdict(item) for item in results],
        }

        out_dir = Path("runtime/reports")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"strategy_failure_diag_{started_at.strftime('%Y%m%d_%H%M%S')}.json"
        out_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        print("\n=== SUMMARY ===", flush=True)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        print(f"report_file={out_file}", flush=True)

        await db.rollback()


if __name__ == "__main__":
    asyncio.run(main())

