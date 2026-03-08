"""Billing API schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

TierValue = Literal["free", "go", "plus", "pro"]


class BillingCheckoutSessionRequest(BaseModel):
    plan: Literal["go", "plus", "pro"]


class BillingCheckoutSessionResponse(BaseModel):
    session_id: str
    checkout_url: str
    publishable_key: str


class BillingPortalSessionResponse(BaseModel):
    portal_url: str


class BillingPlanResponse(BaseModel):
    tier: TierValue
    display_name: str
    price_usd_monthly: float
    currency: str = "USD"
    stripe_price_id: str | None = None
    stripe_product_id: str | None = None
    limits: dict[str, int] = Field(default_factory=dict)


class BillingPlansResponse(BaseModel):
    plans: list[BillingPlanResponse] = Field(default_factory=list)


class BillingQuotaMetricResponse(BaseModel):
    metric: str
    used: int
    limit: int
    remaining: int
    reset_at: datetime | None = None


class BillingSubscriptionResponse(BaseModel):
    status: str
    tier: TierValue
    stripe_subscription_id: str | None = None
    stripe_price_id: str | None = None
    current_period_end: datetime | None = None
    cancel_at_period_end: bool = False
    pending_tier: TierValue | None = None
    pending_price_id: str | None = None


class BillingOverviewResponse(BaseModel):
    tier: TierValue
    subscription: BillingSubscriptionResponse
    quotas: list[BillingQuotaMetricResponse] = Field(default_factory=list)
    cost_model: dict[str, float] = Field(default_factory=dict)


class BillingWebhookAckResponse(BaseModel):
    event_id: str
    event_type: str
    status: str
    duplicate: bool = False
