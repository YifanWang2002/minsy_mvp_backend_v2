"""Strategy confirmation endpoints for frontend-reviewed DSL artifacts."""

from __future__ import annotations

import copy
import json
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.agents.orchestrator import ChatOrchestrator
from src.agents.phases import Phase, can_transition
from src.api.middleware.auth import get_current_user
from src.api.schemas.events import StrategyConfirmResponse
from src.api.schemas.requests import ChatSendRequest, StrategyConfirmRequest
from src.dependencies import get_db, get_responses_event_streamer
from src.engine.strategy import (
    StrategyDslValidationException,
    StrategyStorageNotFoundError,
    upsert_strategy_dsl,
)
from src.models.phase_transition import PhaseTransition
from src.models.session import Session
from src.models.user import User
from src.services.openai_stream_service import ResponsesEventStreamer

router = APIRouter(prefix="/strategies", tags=["strategies"])


@router.post("/confirm", response_model=StrategyConfirmResponse)
async def confirm_strategy(
    payload: StrategyConfirmRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    streamer: ResponsesEventStreamer = Depends(get_responses_event_streamer),
) -> StrategyConfirmResponse:
    session = await db.scalar(
        select(Session).where(
            Session.id == payload.session_id,
            Session.user_id == user.id,
        ),
    )
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.")
    if session.archived_at is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Session is archived. Unarchive it before confirming strategy.",
        )

    try:
        persistence = await upsert_strategy_dsl(
            db,
            session_id=session.id,
            dsl_payload=payload.dsl_json,
            strategy_id=payload.strategy_id,
            auto_commit=False,
        )
    except StrategyDslValidationException as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "STRATEGY_VALIDATION_FAILED",
                "errors": [
                    {
                        "code": item.code,
                        "message": item.message,
                        "path": item.path,
                        "value": item.value,
                    }
                    for item in exc.errors
                ],
            },
        ) from exc
    except StrategyStorageNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "STRATEGY_STORAGE_NOT_FOUND",
                "message": str(exc),
            },
        ) from exc

    receipt = persistence.receipt

    artifacts = ChatOrchestrator._ensure_phase_keyed(copy.deepcopy(session.artifacts or {}))
    strategy_block = artifacts.setdefault(Phase.STRATEGY.value, {"profile": {}, "missing_fields": []})
    strategy_profile = dict(strategy_block.get("profile", {}))
    strategy_profile["strategy_id"] = str(receipt.strategy_id)
    strategy_profile["strategy_confirmed"] = True
    strategy_profile["strategy_last_confirmed_at"] = receipt.last_updated_at.isoformat()
    strategy_block["profile"] = strategy_profile
    strategy_block["missing_fields"] = []

    stress_block = artifacts.setdefault(
        Phase.STRESS_TEST.value,
        {"profile": {}, "missing_fields": ["strategy_id", "backtest_job_id", "backtest_status"]},
    )
    stress_profile = dict(stress_block.get("profile", {}))
    stress_profile["strategy_id"] = str(receipt.strategy_id)
    stress_profile.pop("backtest_job_id", None)
    stress_profile.pop("backtest_status", None)
    stress_profile.pop("backtest_error_code", None)
    stress_block["profile"] = stress_profile
    stress_block["missing_fields"] = ["backtest_job_id", "backtest_status"]

    session.artifacts = artifacts
    session.last_activity_at = datetime.now(UTC)

    if session.current_phase != Phase.STRESS_TEST.value:
        if not can_transition(session.current_phase, Phase.STRESS_TEST.value):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"Invalid phase transition for confirmation: "
                    f"{session.current_phase} -> {Phase.STRESS_TEST.value}"
                ),
            )

        from_phase = session.current_phase
        session.current_phase = Phase.STRESS_TEST.value

        next_meta = dict(session.metadata_ or {})
        next_meta.update(
            {
                "reason": "strategy_confirmed_from_frontend",
                "strategy_id": str(receipt.strategy_id),
                "phase_transition_at": datetime.now(UTC).isoformat(),
            },
        )
        session.metadata_ = next_meta

        db.add(
            PhaseTransition(
                session_id=session.id,
                from_phase=from_phase,
                to_phase=Phase.STRESS_TEST.value,
                trigger="user_action",
                metadata_={
                    "reason": "strategy_confirmed_from_frontend",
                    "strategy_id": str(receipt.strategy_id),
                },
            )
        )

    await db.commit()
    await db.refresh(session)

    auto_message_text: str | None = None
    auto_assistant_text: str | None = None
    auto_done_payload: dict[str, Any] | None = None
    auto_error: str | None = None
    auto_started = False

    if payload.auto_start_backtest:
        auto_message_text = payload.auto_message or _default_auto_message(
            language=payload.language,
            strategy_id=str(receipt.strategy_id),
        )
        orchestrator = ChatOrchestrator(db)
        follow_up_payload = ChatSendRequest(
            session_id=session.id,
            message=auto_message_text,
        )

        text_parts: list[str] = []
        try:
            async for chunk in orchestrator.handle_message_stream(
                user,
                follow_up_payload,
                streamer,
                language=payload.language,
            ):
                envelope = _parse_sse_payload(chunk)
                if envelope is None:
                    continue
                event_type = envelope.get("type")
                if event_type == "text_delta":
                    delta = envelope.get("delta")
                    if isinstance(delta, str) and delta:
                        text_parts.append(delta)
                elif event_type == "done":
                    auto_done_payload = envelope

            auto_assistant_text = "".join(text_parts).strip() or None
            auto_started = True
            await db.refresh(session)
        except Exception as exc:  # noqa: BLE001
            auto_error = f"{type(exc).__name__}: {exc}"

    return StrategyConfirmResponse(
        session_id=session.id,
        strategy_id=receipt.strategy_id,
        phase=session.current_phase,
        metadata={
            "user_id": str(receipt.user_id),
            "session_id": str(receipt.session_id),
            "strategy_name": receipt.strategy_name,
            "dsl_version": receipt.dsl_version,
            "version": receipt.version,
            "status": receipt.status,
            "timeframe": receipt.timeframe,
            "symbol_count": receipt.symbol_count,
            "payload_hash": receipt.payload_hash,
            "last_updated_at": receipt.last_updated_at.isoformat(),
        },
        auto_started=auto_started,
        auto_message=auto_message_text,
        auto_assistant_text=auto_assistant_text,
        auto_done_payload=auto_done_payload,
        auto_error=auto_error,
    )


def _default_auto_message(*, language: str, strategy_id: str) -> str:
    if language.startswith("zh"):
        return (
            "用户已经确认了策略细节，"
            f"策略已存储为 strategy_id {strategy_id}。"
            "请开始回测，创建并跟踪 backtest job。"
        )
    return (
        "The user has confirmed the strategy details and it is stored as "
        f"strategy_id {strategy_id}. "
        "Please start backtesting, create a backtest job, and track it to completion."
    )


def _parse_sse_payload(chunk: str) -> dict[str, Any] | None:
    data_lines: list[str] = []
    for line in chunk.splitlines():
        if line.startswith("data:"):
            data_lines.append(line.removeprefix("data:").lstrip())
    if not data_lines:
        return None

    raw_data = "\n".join(data_lines).strip()
    if not raw_data:
        return None

    try:
        parsed = json.loads(raw_data)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed
