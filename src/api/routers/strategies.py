"""Strategy confirmation endpoints for frontend-reviewed DSL artifacts."""

from __future__ import annotations

import copy
import json
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.agents.orchestrator import ChatOrchestrator
from src.agents.phases import Phase
from src.api.middleware.auth import get_current_user
from src.api.schemas.events import (
    StrategyConfirmResponse,
    StrategyDetailResponse,
    StrategyDraftDetailResponse,
)
from src.api.schemas.requests import ChatSendRequest, StrategyConfirmRequest
from src.dependencies import get_db, get_responses_event_streamer
from src.engine.strategy import (
    StrategyDslValidationException,
    StrategyStorageNotFoundError,
    get_strategy_draft,
    get_strategy_or_raise,
    upsert_strategy_dsl,
)
from src.models.session import Session
from src.models.user import User
from src.services.openai_stream_service import ResponsesEventStreamer

router = APIRouter(prefix="/strategies", tags=["strategies"])


@router.get("/drafts/{strategy_draft_id}", response_model=StrategyDraftDetailResponse)
async def get_strategy_draft_detail(
    strategy_draft_id: UUID,
    user: User = Depends(get_current_user),
) -> StrategyDraftDetailResponse:
    try:
        draft = await get_strategy_draft(strategy_draft_id)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "STRATEGY_DRAFT_STORE_UNAVAILABLE",
                "message": str(exc),
            },
        ) from exc
    if draft is None or draft.user_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "STRATEGY_DRAFT_NOT_FOUND",
                "message": "Strategy draft not found.",
            },
        )

    return StrategyDraftDetailResponse(
        strategy_draft_id=draft.strategy_draft_id,
        session_id=draft.session_id,
        dsl_json=draft.dsl_json,
        expires_at=draft.expires_at,
        metadata={
            "user_id": str(draft.user_id),
            "session_id": str(draft.session_id),
            "payload_hash": draft.payload_hash,
            "created_at": draft.created_at.isoformat(),
            "expires_at": draft.expires_at.isoformat(),
            "ttl_seconds": draft.ttl_seconds,
        },
    )


@router.get("/{strategy_id}", response_model=StrategyDetailResponse)
async def get_strategy_detail(
    strategy_id: UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> StrategyDetailResponse:
    try:
        strategy = await get_strategy_or_raise(db, strategy_id=strategy_id)
    except StrategyStorageNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "STRATEGY_NOT_FOUND",
                "message": str(exc),
            },
        ) from exc

    if strategy.user_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "STRATEGY_NOT_FOUND",
                "message": "Strategy not found.",
            },
        )

    updated_at = strategy.updated_at if strategy.updated_at is not None else strategy.created_at
    dsl_payload = strategy.dsl_payload if isinstance(strategy.dsl_payload, dict) else {}
    return StrategyDetailResponse(
        strategy_id=strategy.id,
        session_id=strategy.session_id,
        version=int(strategy.version),
        status=strategy.status,
        dsl_json=dsl_payload,
        metadata={
            "user_id": str(strategy.user_id),
            "session_id": str(strategy.session_id),
            "strategy_name": strategy.name,
            "dsl_version": strategy.dsl_version or "",
            "version": int(strategy.version),
            "status": strategy.status,
            "timeframe": strategy.timeframe,
            "symbol_count": len(strategy.symbols or []),
            "last_updated_at": updated_at.isoformat() if updated_at is not None else None,
        },
    )


@router.post("/confirm", response_model=StrategyConfirmResponse)
async def confirm_strategy(
    payload: StrategyConfirmRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    streamer: ResponsesEventStreamer = Depends(get_responses_event_streamer),
) -> StrategyConfirmResponse:
    session = await _load_active_session_for_confirm(
        db=db,
        session_id=payload.session_id,
        user_id=user.id,
    )
    receipt = await _upsert_confirmed_strategy(
        db=db,
        session=session,
        dsl_json=payload.dsl_json,
        strategy_id=payload.strategy_id,
    )
    strategy_market, strategy_tickers, strategy_timeframe = _extract_strategy_scope(
        payload.dsl_json
    )
    strategy_tickers_csv = ",".join(strategy_tickers) if strategy_tickers else None
    strategy_primary_symbol = strategy_tickers[0] if strategy_tickers else None
    scope_updates = _strategy_scope_updates(
        strategy_market=strategy_market,
        strategy_tickers=strategy_tickers,
        strategy_tickers_csv=strategy_tickers_csv,
        strategy_primary_symbol=strategy_primary_symbol,
        strategy_timeframe=strategy_timeframe,
    )
    _apply_strategy_confirm_artifacts(
        session=session,
        receipt=receipt,
        scope_updates=scope_updates,
    )
    _apply_strategy_confirm_metadata(
        session=session,
        receipt=receipt,
        scope_updates=scope_updates,
        advance_to_stress_test=payload.advance_to_stress_test,
    )

    await db.commit()
    await db.refresh(session)

    auto_started, auto_message_text, auto_assistant_text, auto_done_payload, auto_error = (
        await _run_auto_backtest_followup(
            db=db,
            user=user,
            session=session,
            streamer=streamer,
            auto_start_backtest=payload.auto_start_backtest,
            auto_message=payload.auto_message,
            language=payload.language,
            strategy_id=str(receipt.strategy_id),
        )
    )

    return StrategyConfirmResponse(
        session_id=session.id,
        strategy_id=receipt.strategy_id,
        phase=session.current_phase,
        metadata=_receipt_metadata(receipt),
        auto_started=auto_started,
        auto_message=auto_message_text,
        auto_assistant_text=auto_assistant_text,
        auto_done_payload=auto_done_payload,
        auto_error=auto_error,
    )


async def _load_active_session_for_confirm(
    *,
    db: AsyncSession,
    session_id: UUID,
    user_id: UUID,
) -> Session:
    session = await db.scalar(
        select(Session).where(
            Session.id == session_id,
            Session.user_id == user_id,
        ),
    )
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.")
    if session.archived_at is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Session is archived. Unarchive it before confirming strategy.",
        )
    return session


async def _upsert_confirmed_strategy(
    *,
    db: AsyncSession,
    session: Session,
    dsl_json: dict[str, Any],
    strategy_id: UUID | None,
) -> Any:
    try:
        persistence = await upsert_strategy_dsl(
            db,
            session_id=session.id,
            dsl_payload=dsl_json,
            strategy_id=strategy_id,
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
    return persistence.receipt


def _apply_strategy_confirm_artifacts(
    *,
    session: Session,
    receipt: Any,
    scope_updates: dict[str, Any],
) -> None:
    artifacts = ChatOrchestrator._ensure_phase_keyed(copy.deepcopy(session.artifacts or {}))
    strategy_block = artifacts.setdefault(Phase.STRATEGY.value, {"profile": {}, "missing_fields": []})
    strategy_profile = dict(strategy_block.get("profile", {}))
    strategy_profile["strategy_id"] = str(receipt.strategy_id)
    strategy_profile.update(scope_updates)
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
    stress_profile.update(scope_updates)
    stress_profile.pop("backtest_job_id", None)
    stress_profile.pop("backtest_status", None)
    stress_profile.pop("backtest_error_code", None)
    stress_block["profile"] = stress_profile
    stress_block["missing_fields"] = ["backtest_job_id", "backtest_status"]
    session.artifacts = artifacts


def _apply_strategy_confirm_metadata(
    *,
    session: Session,
    receipt: Any,
    scope_updates: dict[str, Any],
    advance_to_stress_test: bool,
) -> None:
    # Reset response-chain context so post-confirm turns always use refreshed
    # strategy runtime policy/toolset.
    session.previous_response_id = None
    session.current_phase = Phase.STRATEGY.value
    session.last_activity_at = datetime.now(UTC)
    next_meta = dict(session.metadata_ or {})
    next_meta["strategy_id"] = str(receipt.strategy_id)
    next_meta["strategy_confirmed_at"] = receipt.last_updated_at.isoformat()
    next_meta.update(scope_updates)

    # Keep the session in strategy phase for performance-driven iteration.
    # `advance_to_stress_test` is currently ignored until dedicated stress-test
    # tools are shipped.
    if advance_to_stress_test:
        next_meta["advance_to_stress_test_ignored"] = True
        next_meta["advance_to_stress_test_ignored_at"] = datetime.now(UTC).isoformat()
    session.metadata_ = next_meta


async def _run_auto_backtest_followup(
    *,
    db: AsyncSession,
    user: User,
    session: Session,
    streamer: ResponsesEventStreamer,
    auto_start_backtest: bool,
    auto_message: str | None,
    language: str,
    strategy_id: str,
) -> tuple[bool, str | None, str | None, dict[str, Any] | None, str | None]:
    auto_message_text: str | None = None
    auto_assistant_text: str | None = None
    auto_done_payload: dict[str, Any] | None = None
    auto_error: str | None = None
    auto_started = False
    if not auto_start_backtest:
        return auto_started, auto_message_text, auto_assistant_text, auto_done_payload, auto_error

    auto_message_text = auto_message or _default_auto_message(
        language=language,
        strategy_id=strategy_id,
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
            language=language,
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
    return auto_started, auto_message_text, auto_assistant_text, auto_done_payload, auto_error


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


def _extract_strategy_scope(
    dsl_json: dict[str, Any],
) -> tuple[str | None, list[str], str | None]:
    if not isinstance(dsl_json, dict):
        return None, [], None

    universe = dsl_json.get("universe")
    if not isinstance(universe, dict):
        universe = {}

    strategy_market = _coerce_non_empty_string(universe.get("market"))
    strategy_tickers = _coerce_ticker_list(universe.get("tickers"))
    strategy_timeframe = _coerce_non_empty_string(dsl_json.get("timeframe"))
    return strategy_market, strategy_tickers, strategy_timeframe


def _strategy_scope_updates(
    *,
    strategy_market: str | None,
    strategy_tickers: list[str],
    strategy_tickers_csv: str | None,
    strategy_primary_symbol: str | None,
    strategy_timeframe: str | None,
) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    if strategy_market:
        updates["strategy_market"] = strategy_market
    if strategy_tickers:
        updates["strategy_tickers"] = strategy_tickers
    if strategy_tickers_csv:
        updates["strategy_tickers_csv"] = strategy_tickers_csv
    if strategy_primary_symbol:
        updates["strategy_primary_symbol"] = strategy_primary_symbol
    if strategy_timeframe:
        updates["strategy_timeframe"] = strategy_timeframe
    return updates


def _receipt_metadata(receipt: Any) -> dict[str, Any]:
    return {
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
    }


def _coerce_non_empty_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _coerce_ticker_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []

    normalized: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        normalized.append(text)
    return normalized
