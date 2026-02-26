"""Strategy confirmation endpoints for frontend-reviewed DSL artifacts."""

from __future__ import annotations

import copy
import json
import math
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.orchestration import ChatOrchestrator
from apps.api.agents.phases import Phase, can_transition
from apps.api.middleware.auth import get_current_user
from apps.api.schemas.events import (
    StrategyBacktestSummary,
    StrategyConfirmResponse,
    StrategyDetailResponse,
    StrategyDraftDetailResponse,
    StrategyListItemResponse,
    StrategyVersionDiffResponse,
    StrategyVersionItemResponse,
)
from apps.api.schemas.requests import ChatSendRequest, StrategyConfirmRequest
from apps.api.dependencies import get_db, get_responses_event_streamer
from packages.domain.strategy import (
    StrategyDslValidationException,
    StrategyRevisionNotFoundError,
    StrategyStorageNotFoundError,
    diff_strategy_versions,
    get_strategy_draft,
    get_strategy_or_raise,
    get_strategy_version_payload,
    list_strategy_versions,
    upsert_strategy_dsl,
)
from packages.infra.db.models.backtest import BacktestJob
from packages.infra.db.models.phase_transition import PhaseTransition
from packages.infra.db.models.session import Session
from packages.infra.db.models.strategy import Strategy
from packages.infra.db.models.user import User
from apps.api.orchestration.openai_stream_service import ResponsesEventStreamer
from packages.domain.session.services.session_title_service import refresh_session_title

router = APIRouter(prefix="/strategies", tags=["strategies"])

_BACKTEST_STATUS_TO_EXTERNAL: dict[str, str] = {
    "queued": "pending",
    "running": "running",
    "completed": "done",
    "failed": "failed",
    "cancelled": "failed",
}


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


@router.get("", response_model=list[StrategyListItemResponse])
async def list_user_strategies(
    limit: int = Query(default=30, ge=1, le=200),
    offset: int = Query(default=0, ge=0, le=10_000),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[StrategyListItemResponse]:
    rows = (
        await db.scalars(
            select(Strategy)
            .where(Strategy.user_id == user.id)
            .order_by(Strategy.updated_at.desc(), Strategy.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
    ).all()

    items: list[StrategyListItemResponse] = []
    for strategy in rows:
        dsl_payload = strategy.dsl_payload if isinstance(strategy.dsl_payload, dict) else {}
        latest_backtest = await _resolve_latest_backtest_summary(
            db,
            strategy_id=strategy.id,
            target_version=int(strategy.version),
            current_version=int(strategy.version),
        )
        items.append(
            StrategyListItemResponse(
                strategy_id=strategy.id,
                session_id=strategy.session_id,
                version=int(strategy.version),
                status=strategy.status,
                dsl_json=dsl_payload,
                metadata=_strategy_metadata(strategy),
                latest_backtest=latest_backtest,
            )
        )
    return items


@router.get("/{strategy_id}/versions", response_model=list[StrategyVersionItemResponse])
async def list_strategy_versions_with_payload(
    strategy_id: UUID,
    limit: int = Query(default=20, ge=1, le=200),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[StrategyVersionItemResponse]:
    strategy = await _resolve_owned_strategy_or_404(
        db=db,
        strategy_id=strategy_id,
        user_id=user.id,
    )

    try:
        revisions = await list_strategy_versions(
            db,
            session_id=strategy.session_id,
            strategy_id=strategy.id,
            limit=limit,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "STRATEGY_INVALID_QUERY",
                "message": str(exc),
            },
        ) from exc

    items: list[StrategyVersionItemResponse] = []
    for revision in revisions:
        resolved = await get_strategy_version_payload(
            db,
            session_id=strategy.session_id,
            strategy_id=strategy.id,
            version=int(revision.version),
        )
        backtest = await _resolve_latest_backtest_summary(
            db,
            strategy_id=strategy.id,
            target_version=int(revision.version),
            current_version=int(strategy.version),
        )
        items.append(
            StrategyVersionItemResponse(
                strategy_id=strategy.id,
                version=int(revision.version),
                dsl_json=resolved.dsl_payload,
                revision=_revision_metadata(revision),
                backtest=backtest,
            )
        )
    return items


@router.get("/{strategy_id}/diff", response_model=StrategyVersionDiffResponse)
async def get_strategy_versions_diff(
    strategy_id: UUID,
    from_version: int = Query(ge=1),
    to_version: int = Query(ge=1),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> StrategyVersionDiffResponse:
    strategy = await _resolve_owned_strategy_or_404(
        db=db,
        strategy_id=strategy_id,
        user_id=user.id,
    )

    try:
        diff_result = await diff_strategy_versions(
            db,
            session_id=strategy.session_id,
            strategy_id=strategy.id,
            from_version=from_version,
            to_version=to_version,
        )
        left_payload = await get_strategy_version_payload(
            db,
            session_id=strategy.session_id,
            strategy_id=strategy.id,
            version=from_version,
        )
        right_payload = await get_strategy_version_payload(
            db,
            session_id=strategy.session_id,
            strategy_id=strategy.id,
            version=to_version,
        )
    except StrategyRevisionNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "STRATEGY_REVISION_NOT_FOUND",
                "message": str(exc),
            },
        ) from exc
    except StrategyStorageNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "STRATEGY_NOT_FOUND",
                "message": str(exc),
            },
        ) from exc

    diff_items = _build_display_diff_items(
        patch_ops=diff_result.patch_ops,
        from_payload=left_payload.dsl_payload,
        to_payload=right_payload.dsl_payload,
    )
    return StrategyVersionDiffResponse(
        strategy_id=diff_result.strategy_id,
        from_version=diff_result.from_version,
        to_version=diff_result.to_version,
        patch_op_count=diff_result.op_count,
        patch_ops=diff_result.patch_ops,
        diff_items=diff_items,
        from_payload_hash=diff_result.from_payload_hash,
        to_payload_hash=diff_result.to_payload_hash,
    )


@router.get("/{strategy_id}", response_model=StrategyDetailResponse)
async def get_strategy_detail(
    strategy_id: UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> StrategyDetailResponse:
    strategy = await _resolve_owned_strategy_or_404(
        db=db,
        strategy_id=strategy_id,
        user_id=user.id,
    )

    dsl_payload = strategy.dsl_payload if isinstance(strategy.dsl_payload, dict) else {}
    return StrategyDetailResponse(
        strategy_id=strategy.id,
        session_id=strategy.session_id,
        version=int(strategy.version),
        status=strategy.status,
        dsl_json=dsl_payload,
        metadata=_strategy_metadata(strategy),
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
    previous_phase = session.current_phase
    _apply_strategy_confirm_metadata(
        session=session,
        receipt=receipt,
        scope_updates=scope_updates,
        advance_to_stress_test=payload.advance_to_stress_test,
    )
    if previous_phase != session.current_phase and can_transition(previous_phase, session.current_phase):
        db.add(
            PhaseTransition(
                session_id=session.id,
                from_phase=previous_phase,
                to_phase=session.current_phase,
                trigger="user_action",
                metadata_=_build_phase_transition_metadata(
                    reason="strategy_confirm_to_strategy",
                    source="api",
                    context={
                        "strategy_id": str(receipt.strategy_id),
                        "strategy_name": receipt.strategy_name,
                    },
                ),
            )
        )
    await refresh_session_title(db=db, session=session)

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
                        "path_pointer": getattr(item, "pointer", ""),
                        "stage": getattr(item, "stage", ""),
                        "value": item.value,
                        "expected": getattr(item, "expected", None),
                        "actual": getattr(item, "actual", None),
                        "suggestion": getattr(item, "suggestion", ""),
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
    strategy_profile["strategy_name"] = receipt.strategy_name
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
    stress_profile["strategy_name"] = receipt.strategy_name
    stress_profile.update(scope_updates)
    stress_profile.pop("backtest_job_id", None)
    stress_profile.pop("backtest_status", None)
    stress_profile.pop("backtest_error_code", None)
    stress_block["profile"] = stress_profile
    stress_block["missing_fields"] = ["backtest_job_id", "backtest_status"]

    deployment_block = artifacts.setdefault(
        Phase.DEPLOYMENT.value,
        {"profile": {}, "missing_fields": ["deployment_status"], "runtime": {}},
    )
    deployment_profile = dict(deployment_block.get("profile", {}))
    deployment_profile["strategy_id"] = str(receipt.strategy_id)
    deployment_profile["strategy_name"] = receipt.strategy_name
    deployment_profile.update(scope_updates)
    deployment_profile["deployment_status"] = "ready"
    deployment_profile["deployment_prepared_at"] = receipt.last_updated_at.isoformat()
    deployment_block["profile"] = deployment_profile
    deployment_block["missing_fields"] = []
    deployment_runtime = dict(deployment_block.get("runtime", {}))
    deployment_runtime.update(
        {
            "strategy_id": str(receipt.strategy_id),
            "strategy_name": receipt.strategy_name,
            "deployment_status": "ready",
            "prepared_at": receipt.last_updated_at.isoformat(),
        }
    )
    deployment_block["runtime"] = deployment_runtime
    session.artifacts = artifacts


def _apply_strategy_confirm_metadata(
    *,
    session: Session,
    receipt: Any,
    scope_updates: dict[str, Any],
    advance_to_stress_test: bool,
) -> None:
    # Reset response-chain context so post-confirm turns always use refreshed
    # strategy runtime policy/toolset and continue in strategy iteration.
    session.previous_response_id = None
    session.current_phase = Phase.STRATEGY.value
    session.last_activity_at = datetime.now(UTC)
    next_meta = dict(session.metadata_ or {})
    next_meta["strategy_id"] = str(receipt.strategy_id)
    next_meta["strategy_name"] = receipt.strategy_name
    next_meta["strategy_confirmed_at"] = receipt.last_updated_at.isoformat()
    next_meta["deployment_status"] = "ready"
    next_meta["deployment_prepared_at"] = receipt.last_updated_at.isoformat()
    next_meta.update(scope_updates)

    # `advance_to_stress_test` is currently ignored because strategy iteration
    # remains inside strategy phase in the current product boundary.
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


def _build_phase_transition_metadata(
    *,
    reason: str,
    source: str,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "reason": reason,
        "source": source,
        "context": context or {},
        "recorded_at": datetime.now(UTC).isoformat(),
    }


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


async def _resolve_owned_strategy_or_404(
    *,
    db: AsyncSession,
    strategy_id: UUID,
    user_id: UUID,
) -> Strategy:
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

    if strategy.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "STRATEGY_NOT_FOUND",
                "message": "Strategy not found.",
            },
        )
    return strategy


def _strategy_metadata(strategy: Strategy) -> dict[str, Any]:
    updated_at = strategy.updated_at if strategy.updated_at is not None else strategy.created_at
    return {
        "user_id": str(strategy.user_id),
        "session_id": str(strategy.session_id),
        "strategy_name": strategy.name,
        "dsl_version": strategy.dsl_version or "",
        "version": int(strategy.version),
        "status": strategy.status,
        "timeframe": strategy.timeframe,
        "symbol_count": len(strategy.symbols or []),
        "last_updated_at": updated_at.isoformat() if updated_at is not None else None,
    }


def _revision_metadata(revision: Any) -> dict[str, Any]:
    return {
        "strategy_id": str(revision.strategy_id),
        "session_id": str(revision.session_id) if revision.session_id else None,
        "version": int(revision.version),
        "dsl_version": revision.dsl_version,
        "payload_hash": revision.payload_hash,
        "change_type": revision.change_type,
        "source_version": revision.source_version,
        "patch_op_count": int(revision.patch_op_count),
        "created_at": revision.created_at.isoformat(),
    }


async def _resolve_latest_backtest_summary(
    db: AsyncSession,
    *,
    strategy_id: UUID,
    target_version: int | None,
    current_version: int | None,
    max_curve_points: int = 160,
) -> StrategyBacktestSummary | None:
    jobs = (
        await db.scalars(
            select(BacktestJob)
            .where(BacktestJob.strategy_id == strategy_id)
            .order_by(BacktestJob.submitted_at.desc(), BacktestJob.created_at.desc())
            .limit(200),
        )
    ).all()
    if not jobs:
        return None

    selected: BacktestJob | None = None
    if target_version is None:
        selected = jobs[0]
    else:
        for job in jobs:
            if _parse_backtest_strategy_version(job) == target_version:
                selected = job
                break
        if selected is None and current_version is not None and target_version == current_version:
            selected = jobs[0]

    if selected is None:
        return None
    return _serialize_backtest_summary(selected, max_curve_points=max_curve_points)


def _serialize_backtest_summary(
    job: BacktestJob,
    *,
    max_curve_points: int,
) -> StrategyBacktestSummary:
    status = _BACKTEST_STATUS_TO_EXTERNAL.get(job.status, "pending")
    result = job.results if isinstance(job.results, dict) else {}
    summary = result.get("summary")
    if not isinstance(summary, dict):
        summary = {}
    performance = result.get("performance")
    if not isinstance(performance, dict):
        performance = {}
    metrics = performance.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}

    raw_equity = result.get("equity_curve")
    if not isinstance(raw_equity, list):
        raw_equity = []
    normalized_equity = _downsample_equity_curve(raw_equity, max_points=max_curve_points)

    return StrategyBacktestSummary(
        job_id=job.id,
        status=status,
        strategy_version=_parse_backtest_strategy_version(job),
        total_return_pct=_as_float(summary.get("total_return_pct")),
        max_drawdown_pct=_as_float(summary.get("max_drawdown_pct")),
        sharpe_ratio=_as_float(metrics.get("sharpe")),
        equity_curve=normalized_equity,
        completed_at=job.completed_at,
    )


def _parse_backtest_strategy_version(job: BacktestJob) -> int | None:
    config = job.config if isinstance(job.config, dict) else {}
    raw = config.get("strategy_version")
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw if raw > 0 else None
    if isinstance(raw, float):
        value = int(raw)
        return value if value > 0 else None
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        try:
            value = int(text)
        except ValueError:
            return None
        return value if value > 0 else None
    return None


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _downsample_equity_curve(
    rows: list[Any],
    *,
    max_points: int,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in rows:
        normalized_item = _normalize_equity_curve_point(item)
        if normalized_item is not None:
            normalized.append(normalized_item)

    if len(normalized) <= max_points:
        return normalized
    step = max(1, math.ceil(len(normalized) / max_points))
    sampled = [normalized[index] for index in range(0, len(normalized), step)]
    if sampled and sampled[-1] != normalized[-1]:
        sampled.append(normalized[-1])
    if len(sampled) <= max_points:
        return sampled
    return [*sampled[: max_points - 1], normalized[-1]]


def _normalize_equity_curve_point(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    timestamp = item.get("timestamp")
    equity = _as_float(item.get("equity"))
    if not isinstance(timestamp, str) or not timestamp.strip() or equity is None:
        return None
    return {
        "timestamp": timestamp,
        "equity": equity,
    }


def _build_display_diff_items(
    *,
    patch_ops: list[dict[str, Any]],
    from_payload: dict[str, Any],
    to_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for raw_op in patch_ops:
        if not isinstance(raw_op, dict):
            continue
        op_name = str(raw_op.get("op", "")).strip().lower()
        path = str(raw_op.get("path", "")).strip()
        if not op_name or not path:
            continue

        old_value = _json_pointer_get(from_payload, path)
        new_value = _json_pointer_get(to_payload, path)
        if op_name == "add":
            old_value = None
        elif op_name == "remove":
            new_value = None

        items.append(
            {
                "op": op_name,
                "path": path,
                "old_value": old_value,
                "new_value": new_value,
            }
        )
    return items


def _json_pointer_get(payload: Any, pointer: str) -> Any:
    if pointer == "":
        return payload
    if not pointer.startswith("/"):
        return None

    cursor = payload
    for raw_token in pointer.split("/")[1:]:
        token = _decode_pointer_token(raw_token)
        if isinstance(cursor, dict):
            if token not in cursor:
                return None
            cursor = cursor[token]
            continue
        if isinstance(cursor, list):
            if token == "-":
                return None
            try:
                index = int(token)
            except ValueError:
                return None
            if index < 0 or index >= len(cursor):
                return None
            cursor = cursor[index]
            continue
        return None
    return cursor


def _decode_pointer_token(token: str) -> str:
    return token.replace("~1", "/").replace("~0", "~")


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
