"""Canonical chart annotation CRUD and demo endpoints."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.dependencies import get_db
from apps.api.middleware.auth import get_current_user
from apps.api.schemas.chart_annotations import (
    ChartAnnotationBatchUpsertRequest,
    ChartAnnotationConflictResponse,
    ChartAnnotationCreateRequest,
    ChartAnnotationDeleteRequest,
    ChartAnnotationDemoSeedRequest,
    ChartAnnotationListResponse,
    ChartAnnotationUpdateRequest,
)
from packages.domain.chart_annotations.service import (
    ChartAnnotationConflictError,
    ChartAnnotationFilter,
    append_chart_annotation_snapshot,
    batch_upsert_chart_annotations,
    create_chart_annotation,
    delete_chart_annotation,
    generate_demo_annotations,
    project_backtest_annotations_for_scope,
    project_execution_annotations_for_scope,
    update_chart_annotation,
)
from packages.infra.db.models.user import User

router = APIRouter(prefix="/chart-annotations", tags=["chart-annotations"])


def _parse_dt(value: str | None) -> datetime | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        numeric = float(text)
    except ValueError:
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    if numeric > 9999999999:
        numeric = numeric / 1000.0
    try:
        return datetime.fromtimestamp(numeric, tz=UTC)
    except (OSError, OverflowError, ValueError):
        return None


def _build_filters(
    *,
    market: str | None,
    symbol: str | None,
    timeframe: str | None,
    deployment_id: UUID | None,
    strategy_id: UUID | None,
    backtest_id: UUID | None,
    chart_layout_id: str | None,
    from_raw: str | None,
    to_raw: str | None,
    include_deleted: bool = False,
) -> ChartAnnotationFilter:
    return ChartAnnotationFilter(
        market=market.lower() if market else None,
        symbol=symbol.upper() if symbol else None,
        timeframe=timeframe.lower() if timeframe else None,
        deployment_id=deployment_id,
        strategy_id=strategy_id,
        backtest_id=backtest_id,
        chart_layout_id=chart_layout_id,
        from_dt=_parse_dt(from_raw),
        to_dt=_parse_dt(to_raw),
        include_deleted=include_deleted,
    )


@router.get("", response_model=ChartAnnotationListResponse)
async def get_chart_annotations(
    market: str | None = Query(default=None),
    symbol: str | None = Query(default=None),
    timeframe: str | None = Query(default=None),
    deployment_id: UUID | None = Query(default=None),
    strategy_id: UUID | None = Query(default=None),
    backtest_id: UUID | None = Query(default=None),
    chart_layout_id: str | None = Query(default=None),
    from_raw: str | None = Query(default=None, alias="from"),
    to_raw: str | None = Query(default=None, alias="to"),
    include_deleted: bool = Query(default=False),
    include_projected: bool = Query(default=True),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ChartAnnotationListResponse:
    filters = _build_filters(
        market=market,
        symbol=symbol,
        timeframe=timeframe,
        deployment_id=deployment_id,
        strategy_id=strategy_id,
        backtest_id=backtest_id,
        chart_layout_id=chart_layout_id,
        from_raw=from_raw,
        to_raw=to_raw,
        include_deleted=include_deleted,
    )
    snapshot = await append_chart_annotation_snapshot(
        db,
        owner_user_id=user.id,
        filters=filters,
    )
    annotations = list(snapshot["annotations"])
    if include_projected:
        annotations.extend(
            await project_execution_annotations_for_scope(
                db,
                owner_user_id=user.id,
                filters=filters,
            )
        )
        annotations.extend(
            await project_backtest_annotations_for_scope(
                db,
                filters=filters,
            )
        )
    return ChartAnnotationListResponse(
        annotations=annotations,
        cursor=int(snapshot["cursor"]),
    )


@router.post("", response_model=dict)
async def create_chart_annotation_endpoint(
    request: ChartAnnotationCreateRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    return await create_chart_annotation(
        db,
        owner_user_id=user.id,
        payload=request.annotation.model_dump(mode="json", exclude_none=True),
    )


@router.patch("/{annotation_id}", response_model=dict)
async def update_chart_annotation_endpoint(
    annotation_id: UUID,
    request: ChartAnnotationUpdateRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    try:
        return await update_chart_annotation(
            db,
            owner_user_id=user.id,
            annotation_id=annotation_id,
            base_version=request.base_version,
            payload=request.annotation.model_dump(mode="json", exclude_none=True),
        )
    except ChartAnnotationConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=ChartAnnotationConflictResponse(
                message="Annotation version conflict.",
                latest=exc.latest,
            ).model_dump(mode="json"),
        ) from exc
    except PermissionError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"code": "CHART_ANNOTATION_LOCKED", "message": str(exc)},
        ) from exc
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "CHART_ANNOTATION_NOT_FOUND", "message": str(exc)},
        ) from exc


@router.delete("/{annotation_id}", response_model=dict)
async def delete_chart_annotation_endpoint(
    annotation_id: UUID,
    request: ChartAnnotationDeleteRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    try:
        return await delete_chart_annotation(
            db,
            owner_user_id=user.id,
            annotation_id=annotation_id,
            base_version=request.base_version,
        )
    except ChartAnnotationConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=ChartAnnotationConflictResponse(
                message="Annotation version conflict.",
                latest=exc.latest,
            ).model_dump(mode="json"),
        ) from exc
    except PermissionError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"code": "CHART_ANNOTATION_LOCKED", "message": str(exc)},
        ) from exc
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "CHART_ANNOTATION_NOT_FOUND", "message": str(exc)},
        ) from exc


@router.post("/batch-upsert", response_model=list[dict])
async def batch_upsert_chart_annotations_endpoint(
    request: ChartAnnotationBatchUpsertRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[dict]:
    return await batch_upsert_chart_annotations(
        db,
        owner_user_id=user.id,
        annotations=[
            item.model_dump(mode="json", exclude_none=True)
            for item in request.annotations
        ],
    )


@router.post("/demo-seed", response_model=list[dict])
async def demo_seed_chart_annotations_endpoint(
    request: ChartAnnotationDemoSeedRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[dict]:
    return await generate_demo_annotations(
        db,
        owner_user_id=user.id,
        market=request.market.lower(),
        symbol=request.symbol.upper(),
        timeframe=request.timeframe.lower(),
        scenario=request.scenario,
        reference=request.reference,
        persist=request.persist,
        deployment_id=request.deployment_id,
        strategy_id=request.strategy_id,
        backtest_id=request.backtest_id,
        chart_layout_id=request.chart_layout_id,
    )
