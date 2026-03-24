"""Tier quota guard used by API routes and domain services."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

from packages.domain.billing.usage_service import (
    UsageMetric,
    UsageService,
    month_window_reset_at,
)
from packages.domain.exceptions import DomainError
from packages.shared_settings.schema.settings import settings


@dataclass(frozen=True, slots=True)
class QuotaSnapshot:
    """Resolved limit + usage tuple for one metric."""

    metric: str
    tier: str
    used: int
    limit: int
    remaining: int
    reset_at: datetime | None


class QuotaExceededError(DomainError):
    """402 error with quota metadata payload."""

    __slots__ = ("metric", "tier", "used", "limit", "remaining", "reset_at")

    def __init__(
        self,
        *,
        metric: str,
        tier: str,
        used: int,
        limit: int,
        remaining: int,
        reset_at: datetime | None,
    ) -> None:
        super().__init__(
            status_code=402,
            code="QUOTA_EXCEEDED",
            message=f"Quota exceeded for metric '{metric}'.",
        )
        self.metric = metric
        self.tier = tier
        self.used = used
        self.limit = limit
        self.remaining = remaining
        self.reset_at = reset_at

    @property
    def detail(self) -> dict[str, Any]:
        payload = super().detail
        payload.update(
            {
                "metric": self.metric,
                "tier": self.tier,
                "used": self.used,
                "limit": self.limit,
                "remaining": self.remaining,
                "reset_at": self.reset_at.isoformat() if self.reset_at else None,
            }
        )
        return payload


class FeatureNotIncludedError(DomainError):
    """403 guard reserved for future tier-gated features."""

    def __init__(self, *, feature: str, tier: str) -> None:
        super().__init__(
            status_code=403,
            code="FEATURE_NOT_INCLUDED",
            message=f"Feature '{feature}' is not included in tier '{tier}'.",
        )


class QuotaService:
    """Resolve per-tier limits and enforce metric quotas."""

    def __init__(self, usage_service: UsageService) -> None:
        self._usage = usage_service

    @staticmethod
    def normalize_tier(value: str | None) -> str:
        raw = (value or "").strip().lower()
        if raw in {"free", "go", "plus", "pro"}:
            return raw
        return "free"

    @staticmethod
    def normalize_metric(metric: str) -> str:
        raw = (metric or "").strip()
        if raw in {
            UsageMetric.CPU_TOKENS_MONTHLY_TOTAL,
            UsageMetric.CPU_JOBS_MONTHLY_TOTAL,
            UsageMetric.LEGACY_CPU_JOBS_MONTHLY_TOTAL,
        }:
            return UsageMetric.CPU_TOKENS_MONTHLY_TOTAL
        return raw

    def _limits_for_tier(self, tier: str) -> dict[str, int]:
        normalized_tier = self.normalize_tier(tier)
        limits = settings.billing_tier_limits.get(normalized_tier)
        if isinstance(limits, dict):
            return {
                key: max(int(value), 0)
                for key, value in limits.items()
                if isinstance(key, str)
            }
        return {}

    def _limit_with_user_override(
        self,
        *,
        user_id: UUID,
        metric: str,
        limit: int,
    ) -> int:
        normalized_limit = max(int(limit), 0)
        if normalized_limit <= 0:
            return normalized_limit

        override = settings.billing_user_quota_overrides.get(str(user_id))
        if not isinstance(override, dict):
            return normalized_limit

        metrics = override.get("metrics")
        if metric not in metrics:
            return normalized_limit

        try:
            percent_bonus = int(override.get("percent_bonus", 0))
        except (TypeError, ValueError):
            return normalized_limit
        if percent_bonus <= 0:
            return normalized_limit

        bonus = (normalized_limit * percent_bonus + 99) // 100
        return normalized_limit + bonus

    async def resolve_metric_snapshot(
        self,
        *,
        user_id: UUID,
        tier: str,
        metric: str,
    ) -> QuotaSnapshot:
        normalized_tier = self.normalize_tier(tier)
        normalized_metric = self.normalize_metric(metric)
        limits = self._limits_for_tier(normalized_tier)
        limit = max(int(limits.get(normalized_metric, 0)), 0)
        if (
            limit <= 0
            and normalized_metric == UsageMetric.CPU_TOKENS_MONTHLY_TOTAL
        ):
            limit = max(
                int(limits.get(UsageMetric.LEGACY_CPU_JOBS_MONTHLY_TOTAL, 0)),
                0,
            )
        limit = self._limit_with_user_override(
            user_id=user_id,
            metric=normalized_metric,
            limit=limit,
        )

        reset_at: datetime | None = None
        if normalized_metric in {
            UsageMetric.AI_TOKENS_MONTHLY_TOTAL,
            UsageMetric.CPU_TOKENS_MONTHLY_TOTAL,
        }:
            monthly = await self._usage.get_monthly_usage(user_id=user_id)
            reset_at = month_window_reset_at(monthly.window_month)
            if normalized_metric == UsageMetric.AI_TOKENS_MONTHLY_TOTAL:
                used = monthly.ai_total_tokens
            else:
                used = monthly.cpu_tokens_total
        elif normalized_metric in {
            UsageMetric.STRATEGIES_CURRENT_COUNT,
            UsageMetric.DEPLOYMENTS_RUNNING_COUNT,
        }:
            capacity = await self._usage.get_capacity_snapshot(user_id=user_id)
            if normalized_metric == UsageMetric.STRATEGIES_CURRENT_COUNT:
                used = capacity.strategies_current_count
            else:
                used = capacity.deployments_running_count
        else:
            used = 0

        remaining = max(limit - used, 0)
        return QuotaSnapshot(
            metric=normalized_metric,
            tier=normalized_tier,
            used=used,
            limit=limit,
            remaining=remaining,
            reset_at=reset_at,
        )

    async def assert_quota_available(
        self,
        *,
        user_id: UUID,
        tier: str,
        metric: str,
        increment: int = 1,
    ) -> QuotaSnapshot:
        snapshot = await self.resolve_metric_snapshot(
            user_id=user_id,
            tier=tier,
            metric=metric,
        )
        projected_used = snapshot.used + max(int(increment), 0)
        if projected_used > snapshot.limit:
            raise QuotaExceededError(
                metric=snapshot.metric,
                tier=snapshot.tier,
                used=snapshot.used,
                limit=snapshot.limit,
                remaining=snapshot.remaining,
                reset_at=snapshot.reset_at,
            )
        return snapshot

    async def list_usage_overview(
        self,
        *,
        user_id: UUID,
        tier: str,
    ) -> list[QuotaSnapshot]:
        metrics = [
            UsageMetric.AI_TOKENS_MONTHLY_TOTAL,
            UsageMetric.CPU_TOKENS_MONTHLY_TOTAL,
            UsageMetric.STRATEGIES_CURRENT_COUNT,
            UsageMetric.DEPLOYMENTS_RUNNING_COUNT,
        ]
        rows: list[QuotaSnapshot] = []
        for metric in metrics:
            rows.append(
                await self.resolve_metric_snapshot(
                    user_id=user_id,
                    tier=tier,
                    metric=metric,
                )
            )
        return rows
