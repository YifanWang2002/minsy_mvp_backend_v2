"""Schemas for canonical chart annotations."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class ChartAnnotationSourcePayload(BaseModel):
    type: str
    source_id: str | None = None
    actor_user_id: UUID | None = None


class ChartAnnotationScopePayload(BaseModel):
    market: str
    symbol: str
    timeframe: str
    anchor_space: str = "time_price"
    deployment_id: UUID | None = None
    strategy_id: UUID | None = None
    backtest_id: UUID | None = None
    chart_layout_id: str | None = None


class ChartAnnotationSemanticPayload(BaseModel):
    kind: str
    role: str
    intent: str | None = None
    direction: str | None = None
    status: str | None = None


class ChartAnnotationToolPayload(BaseModel):
    family: str
    vendor: str = "tradingview"
    vendor_type: str | None = None


class ChartAnnotationDocumentPayload(BaseModel):
    id: UUID | None = None
    version: int | None = None
    source: ChartAnnotationSourcePayload
    scope: ChartAnnotationScopePayload
    semantic: ChartAnnotationSemanticPayload
    tool: ChartAnnotationToolPayload
    anchors: dict[str, Any] = Field(default_factory=dict)
    geometry: dict[str, Any] = Field(default_factory=dict)
    content: dict[str, Any] = Field(default_factory=dict)
    style: dict[str, Any] = Field(default_factory=dict)
    relations: dict[str, Any] = Field(default_factory=dict)
    lifecycle: dict[str, Any] = Field(default_factory=dict)
    vendor_native: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None


class ChartAnnotationListResponse(BaseModel):
    annotations: list[dict[str, Any]] = Field(default_factory=list)
    cursor: int = 0


class ChartAnnotationCreateRequest(BaseModel):
    annotation: ChartAnnotationDocumentPayload


class ChartAnnotationUpdateRequest(BaseModel):
    base_version: int = Field(ge=1)
    annotation: ChartAnnotationDocumentPayload


class ChartAnnotationDeleteRequest(BaseModel):
    base_version: int = Field(ge=1)


class ChartAnnotationBatchUpsertRequest(BaseModel):
    annotations: list[ChartAnnotationDocumentPayload] = Field(default_factory=list)


class ChartAnnotationConflictResponse(BaseModel):
    code: str = "CHART_ANNOTATION_VERSION_CONFLICT"
    message: str
    latest: dict[str, Any]


class ChartAnnotationDemoSeedRequest(BaseModel):
    market: str
    symbol: str
    timeframe: str
    scenario: str = "long"
    reference: str = "latest_closed_bar"
    persist: bool = True
    deployment_id: UUID | None = None
    strategy_id: UUID | None = None
    backtest_id: UUID | None = None
    chart_layout_id: str | None = None
