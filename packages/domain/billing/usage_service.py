"""Billing usage persistence and aggregate helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from math import ceil
from typing import Any
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from packages.infra.db.models.billing_usage_event import BillingUsageEvent
from packages.infra.db.models.billing_usage_monthly import BillingUsageMonthly
from packages.infra.db.models.deployment import Deployment
from packages.infra.db.models.strategy import Strategy
from packages.shared_settings.schema.settings import settings

_AI_METRIC_CODE = "ai_tokens_monthly_total"
_CPU_TOKENS_METRIC_CODE = "cpu_tokens_monthly_total"
_CPU_JOBS_METRIC_CODE_LEGACY = "cpu_jobs_monthly_total"


def month_window_start(value: datetime | None = None) -> date:
    now = value.astimezone(UTC) if isinstance(value, datetime) else datetime.now(UTC)
    return date(year=now.year, month=now.month, day=1)


def month_window_reset_at(window_start: date) -> datetime:
    if window_start.month == 12:
        return datetime(window_start.year + 1, 1, 1, tzinfo=UTC)
    return datetime(window_start.year, window_start.month + 1, 1, tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class MonthlyUsageSnapshot:
    """Current month usage counters for one user."""

    window_month: date
    ai_input_tokens: int
    ai_reasoning_tokens: int
    ai_output_tokens: int
    ai_total_tokens: int
    cpu_tokens_total: int

    @property
    def cpu_jobs_total(self) -> int:
        """Backward-compat alias; now represents CPU token usage."""
        return self.cpu_tokens_total


@dataclass(frozen=True, slots=True)
class CapacitySnapshot:
    """Current stock resource counters for one user."""

    strategies_current_count: int
    deployments_running_count: int


@dataclass(frozen=True, slots=True)
class WeightedAiUsageBreakdown:
    """Raw OpenAI tokens + weighted internal tokens for quota accounting."""

    input_tokens: int
    reasoning_tokens: int
    output_tokens: int
    raw_total_tokens: int
    weighted_total_tokens: int
    estimated_cost_usd: float
    model: str
    input_per_token_usd: float
    output_per_token_usd: float
    usd_per_internal_token: float


@dataclass(frozen=True, slots=True)
class CpuTokenUsageBreakdown:
    """Converted CPU token units for one compute workload estimate."""

    estimated_bars: int
    bars_per_token: int
    token_quantity: int


class UsageService:
    """Read/write usage counters used by quota guard and overview APIs."""

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def _get_or_create_monthly_row(
        self,
        *,
        user_id: UUID,
        window_month: date,
    ) -> BillingUsageMonthly:
        existing = await self._db.scalar(
            select(BillingUsageMonthly).where(
                BillingUsageMonthly.user_id == user_id,
                BillingUsageMonthly.window_month == window_month,
            )
        )
        if existing is not None:
            return existing

        created = BillingUsageMonthly(
            user_id=user_id,
            window_month=window_month,
        )
        self._db.add(created)
        await self._db.flush()
        return created

    async def get_monthly_usage(
        self,
        *,
        user_id: UUID,
        window_month: date | None = None,
    ) -> MonthlyUsageSnapshot:
        resolved_month = window_month or month_window_start()
        row = await self._db.scalar(
            select(BillingUsageMonthly).where(
                BillingUsageMonthly.user_id == user_id,
                BillingUsageMonthly.window_month == resolved_month,
            )
        )
        if row is None:
            return MonthlyUsageSnapshot(
                window_month=resolved_month,
                ai_input_tokens=0,
                ai_reasoning_tokens=0,
                ai_output_tokens=0,
                ai_total_tokens=0,
                cpu_tokens_total=0,
            )

        return MonthlyUsageSnapshot(
            window_month=row.window_month,
            ai_input_tokens=max(int(row.ai_input_tokens), 0),
            ai_reasoning_tokens=max(int(row.ai_reasoning_tokens), 0),
            ai_output_tokens=max(int(row.ai_output_tokens), 0),
            ai_total_tokens=max(int(row.ai_total_tokens), 0),
            cpu_tokens_total=max(int(row.cpu_jobs_total), 0),
        )

    async def get_capacity_snapshot(self, *, user_id: UUID) -> CapacitySnapshot:
        strategy_count = await self._db.scalar(
            select(func.count()).select_from(Strategy).where(Strategy.user_id == user_id),
        )
        deployment_count = await self._db.scalar(
            select(func.count()).select_from(Deployment).where(
                Deployment.user_id == user_id,
                Deployment.status.in_(("pending", "active", "paused", "error")),
            ),
        )
        return CapacitySnapshot(
            strategies_current_count=max(int(strategy_count or 0), 0),
            deployments_running_count=max(int(deployment_count or 0), 0),
        )

    async def record_ai_tokens(
        self,
        *,
        user_id: UUID,
        input_tokens: int,
        reasoning_tokens: int,
        output_tokens: int,
        total_tokens_override: int | None = None,
        source: str,
        reference_type: str | None = None,
        reference_id: str | None = None,
        metadata: dict | None = None,
    ) -> MonthlyUsageSnapshot:
        normalized_input = max(int(input_tokens), 0)
        normalized_reasoning = max(int(reasoning_tokens), 0)
        normalized_output = max(int(output_tokens), 0)
        calculated_total_tokens = normalized_input + normalized_reasoning + normalized_output
        total_tokens = (
            max(int(total_tokens_override), 0)
            if total_tokens_override is not None
            else calculated_total_tokens
        )

        window_month = month_window_start()
        if reference_type and reference_id:
            existing = await self._db.scalar(
                select(BillingUsageEvent.id).where(
                    BillingUsageEvent.user_id == user_id,
                    BillingUsageEvent.metric_code == _AI_METRIC_CODE,
                    BillingUsageEvent.reference_type == reference_type,
                    BillingUsageEvent.reference_id == reference_id,
                )
            )
            if existing is not None:
                return await self.get_monthly_usage(user_id=user_id, window_month=window_month)

        row = await self._get_or_create_monthly_row(
            user_id=user_id,
            window_month=window_month,
        )
        row.ai_input_tokens = int(row.ai_input_tokens) + normalized_input
        row.ai_reasoning_tokens = int(row.ai_reasoning_tokens) + normalized_reasoning
        row.ai_output_tokens = int(row.ai_output_tokens) + normalized_output
        row.ai_total_tokens = int(row.ai_total_tokens) + total_tokens

        event = BillingUsageEvent(
            user_id=user_id,
            metric_code=_AI_METRIC_CODE,
            quantity=total_tokens,
            window_month=window_month,
            source=source,
            reference_type=reference_type,
            reference_id=reference_id,
            metadata_=dict(metadata or {}),
        )
        self._db.add(event)
        await self._db.flush()
        return await self.get_monthly_usage(user_id=user_id, window_month=window_month)

    async def record_ai_tokens_from_openai_usage(
        self,
        *,
        user_id: UUID,
        raw_usage: dict,
        model: str | None = None,
        source: str,
        reference_type: str | None = None,
        reference_id: str | None = None,
        metadata: dict | None = None,
    ) -> MonthlyUsageSnapshot:
        weighted = compute_weighted_ai_usage_from_openai(
            raw_usage=raw_usage,
            model=model,
            pricing=settings.openai_pricing,
            billing_cost_model=settings.billing_cost_model,
        )

        metadata_payload = dict(metadata or {})
        metadata_payload.setdefault(
            "weighted_usage",
            {
                "model": weighted.model,
                "input_per_token_usd": weighted.input_per_token_usd,
                "output_per_token_usd": weighted.output_per_token_usd,
                "usd_per_internal_token": weighted.usd_per_internal_token,
                "estimated_cost_usd": weighted.estimated_cost_usd,
                "raw_total_tokens": weighted.raw_total_tokens,
                "weighted_total_tokens": weighted.weighted_total_tokens,
            },
        )
        metadata_payload.setdefault("raw_usage", dict(raw_usage))
        return await self.record_ai_tokens(
            user_id=user_id,
            input_tokens=weighted.input_tokens,
            reasoning_tokens=weighted.reasoning_tokens,
            output_tokens=weighted.output_tokens,
            total_tokens_override=weighted.weighted_total_tokens,
            source=source,
            reference_type=reference_type,
            reference_id=reference_id,
            metadata=metadata_payload,
        )

    async def record_cpu_tokens(
        self,
        *,
        user_id: UUID,
        quantity: int,
        source: str,
        reference_type: str | None = None,
        reference_id: str | None = None,
        metadata: dict | None = None,
    ) -> MonthlyUsageSnapshot:
        normalized_quantity = max(int(quantity), 0)
        window_month = month_window_start()
        if reference_type and reference_id:
            existing = await self._db.scalar(
                select(BillingUsageEvent.id).where(
                    BillingUsageEvent.user_id == user_id,
                    BillingUsageEvent.metric_code.in_(
                        (_CPU_TOKENS_METRIC_CODE, _CPU_JOBS_METRIC_CODE_LEGACY)
                    ),
                    BillingUsageEvent.reference_type == reference_type,
                    BillingUsageEvent.reference_id == reference_id,
                )
            )
            if existing is not None:
                return await self.get_monthly_usage(user_id=user_id, window_month=window_month)
        if normalized_quantity <= 0:
            return await self.get_monthly_usage(user_id=user_id, window_month=window_month)

        row = await self._get_or_create_monthly_row(
            user_id=user_id,
            window_month=window_month,
        )
        row.cpu_jobs_total = int(row.cpu_jobs_total) + normalized_quantity

        event = BillingUsageEvent(
            user_id=user_id,
            metric_code=_CPU_TOKENS_METRIC_CODE,
            quantity=normalized_quantity,
            window_month=window_month,
            source=source,
            reference_type=reference_type,
            reference_id=reference_id,
            metadata_=dict(metadata or {}),
        )
        self._db.add(event)
        await self._db.flush()
        return await self.get_monthly_usage(user_id=user_id, window_month=window_month)

    async def record_cpu_job(
        self,
        *,
        user_id: UUID,
        source: str,
        reference_type: str | None = None,
        reference_id: str | None = None,
        metadata: dict | None = None,
    ) -> MonthlyUsageSnapshot:
        """Backward-compat helper; records one CPU token unit."""
        return await self.record_cpu_tokens(
            user_id=user_id,
            quantity=1,
            source=source,
            reference_type=reference_type,
            reference_id=reference_id,
            metadata=metadata,
        )


def compute_weighted_ai_usage_from_openai(
    *,
    raw_usage: Mapping[str, Any] | None,
    model: str | None,
    pricing: Mapping[str, Any] | None,
    billing_cost_model: Mapping[str, Any] | None,
) -> WeightedAiUsageBreakdown:
    usage_map = dict(raw_usage) if isinstance(raw_usage, Mapping) else {}
    input_tokens = _to_int(usage_map.get("input_tokens") or usage_map.get("prompt_tokens"))
    output_tokens = _to_int(
        usage_map.get("output_tokens") or usage_map.get("completion_tokens")
    )
    reasoning_tokens = 0
    raw_total_tokens = input_tokens + output_tokens

    resolved_model = str(model or "").strip() or "default"
    input_per_token_usd, output_per_token_usd = _resolve_model_pricing(
        pricing=pricing,
        model=resolved_model,
    )

    usd_per_internal_token = _resolve_internal_ai_unit_usd(billing_cost_model)
    estimated_cost_usd = (
        input_tokens * input_per_token_usd
        + output_tokens * output_per_token_usd
    )
    if estimated_cost_usd > 0:
        weighted_total_tokens = (
            max(ceil(estimated_cost_usd / usd_per_internal_token), 1)
            if raw_total_tokens > 0
            else 0
        )
    else:
        weighted_total_tokens = raw_total_tokens
        estimated_cost_usd = raw_total_tokens * usd_per_internal_token

    return WeightedAiUsageBreakdown(
        input_tokens=input_tokens,
        reasoning_tokens=reasoning_tokens,
        output_tokens=output_tokens,
        raw_total_tokens=raw_total_tokens,
        weighted_total_tokens=weighted_total_tokens,
        estimated_cost_usd=round(estimated_cost_usd, 6),
        model=resolved_model,
        input_per_token_usd=input_per_token_usd,
        output_per_token_usd=output_per_token_usd,
        usd_per_internal_token=usd_per_internal_token,
    )


def compute_cpu_tokens_from_bars(
    *,
    estimated_bars: int | None,
    billing_cost_model: Mapping[str, Any] | None,
) -> CpuTokenUsageBreakdown:
    normalized_bars = max(int(estimated_bars or 0), 0)
    bars_per_token = _resolve_cpu_bars_per_token(billing_cost_model)
    effective_bars = normalized_bars if normalized_bars > 0 else 1
    token_quantity = max(ceil(effective_bars / bars_per_token), 1)
    return CpuTokenUsageBreakdown(
        estimated_bars=normalized_bars,
        bars_per_token=bars_per_token,
        token_quantity=token_quantity,
    )


class UsageMetric:
    """Metric keys shared between quota checks and API contracts."""

    AI_TOKENS_MONTHLY_TOTAL = "ai_tokens_monthly_total"
    CPU_TOKENS_MONTHLY_TOTAL = "cpu_tokens_monthly_total"
    LEGACY_CPU_JOBS_MONTHLY_TOTAL = "cpu_jobs_monthly_total"
    CPU_JOBS_MONTHLY_TOTAL = CPU_TOKENS_MONTHLY_TOTAL
    STRATEGIES_CURRENT_COUNT = "strategies_current_count"
    DEPLOYMENTS_RUNNING_COUNT = "deployments_running_count"


def _to_float(value: object) -> float:
    try:
        if value is None:
            return 0.0
        return max(float(value), 0.0)
    except (TypeError, ValueError):
        return 0.0


def _resolve_model_pricing(
    *,
    pricing: Mapping[str, Any] | None,
    model: str,
) -> tuple[float, float]:
    if not isinstance(pricing, Mapping):
        return (0.0, 0.0)

    model_entry = pricing.get(model)
    if not isinstance(model_entry, Mapping):
        model_entry = pricing.get("default")
    if not isinstance(model_entry, Mapping):
        return (0.0, 0.0)

    input_per_token = _to_float(model_entry.get("input_per_token"))
    output_per_token = _to_float(model_entry.get("output_per_token"))
    if input_per_token <= 0:
        input_per_token = _to_float(model_entry.get("input_per_1k_tokens")) / 1000.0
    if output_per_token <= 0:
        output_per_token = _to_float(model_entry.get("output_per_1k_tokens")) / 1000.0
    return (input_per_token, output_per_token)


def _resolve_internal_ai_unit_usd(
    billing_cost_model: Mapping[str, Any] | None,
) -> float:
    raw = (
        billing_cost_model
        if isinstance(billing_cost_model, Mapping)
        else {}
    )
    explicit_unit = _to_float(raw.get("ai_usage_unit_usd"))
    if explicit_unit > 0:
        return explicit_unit

    per_1k = _to_float(raw.get("token_cost_per_1k_usd"))
    if per_1k > 0:
        return per_1k / 1000.0
    return 0.01 / 1000.0


def _resolve_cpu_bars_per_token(
    billing_cost_model: Mapping[str, Any] | None,
) -> int:
    raw = (
        billing_cost_model
        if isinstance(billing_cost_model, Mapping)
        else {}
    )
    explicit = _to_int(raw.get("cpu_bars_per_token"))
    if explicit > 0:
        return explicit
    return 100_000


def _to_int(value: object) -> int:
    try:
        if value is None:
            return 0
        return max(int(value), 0)
    except (TypeError, ValueError):
        return 0
