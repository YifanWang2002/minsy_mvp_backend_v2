"""Billing endpoints: plans, checkout, portal, overview, webhooks."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse, StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.dependencies import get_db
from apps.api.middleware.auth import get_current_user
from apps.api.schemas.billing_schemas import (
    BillingChangePlanRequest,
    BillingChangePlanResponse,
    BillingCheckoutSessionRequest,
    BillingCheckoutSessionResponse,
    BillingOverviewResponse,
    BillingPlanResponse,
    BillingPlansResponse,
    BillingPortalSessionResponse,
    BillingQuotaMetricResponse,
    BillingSubscriptionResponse,
    BillingWebhookAckResponse,
)
from apps.api.services.billing_return_page import (
    normalize_billing_status,
    render_billing_return_html,
)
from apps.api.services.billing_webhook_service import BillingWebhookService
from packages.domain.billing.quota_service import QuotaService
from packages.domain.billing.subscription_sync_service import SubscriptionSyncService
from packages.domain.billing.usage_service import UsageService
from packages.infra.db.models.billing_customer import BillingCustomer
from packages.infra.db.models.billing_subscription import BillingSubscription
from packages.infra.db.models.user import User
from packages.infra.observability.logger import logger
from packages.infra.providers.stripe.client import (
    StripeClientConfigError,
    StripeWebhookSignatureError,
    stripe_client,
)
from packages.shared_settings.schema.settings import settings

router = APIRouter(prefix="/billing", tags=["billing"])

_ACTIVE_SUBSCRIPTION_STATUSES = (
    "trialing",
    "active",
    "past_due",
    "unpaid",
    "paused",
)
_CHECKOUT_TRIAL_DAYS = 7


def _is_missing_promotion_code_error(exc: Exception) -> bool:
    message = str(exc).strip().lower()
    return "no such promotion code" in message


@router.get("/plans", response_model=BillingPlansResponse)
async def get_billing_plans() -> BillingPlansResponse:
    limits = settings.billing_tier_limits
    pricing = settings.billing_cost_model
    go_monthly = float(pricing.get("go_price_usd", 8.0))
    plus_monthly = float(pricing.get("plus_price_usd", 20.0))
    pro_monthly = float(pricing.get("pro_price_usd", 60.0))
    go_yearly = float(pricing.get("go_price_usd_yearly", go_monthly * 12))
    plus_yearly = float(pricing.get("plus_price_usd_yearly", plus_monthly * 12))
    pro_yearly = float(pricing.get("pro_price_usd_yearly", pro_monthly * 12))

    plans = [
        BillingPlanResponse(
            tier="free",
            display_name="Free",
            price_usd_monthly=0,
            price_usd_yearly=0,
            stripe_price_id=None,
            stripe_price_id_monthly=None,
            stripe_price_id_yearly=None,
            stripe_product_id=None,
            limits=limits.get("free", {}),
        ),
        BillingPlanResponse(
            tier="go",
            display_name="Go",
            price_usd_monthly=go_monthly,
            price_usd_yearly=go_yearly,
            stripe_price_id=settings.stripe_price_go_monthly.strip() or None,
            stripe_price_id_monthly=settings.stripe_price_go_monthly.strip() or None,
            stripe_price_id_yearly=settings.stripe_price_go_yearly.strip() or None,
            stripe_product_id=settings.stripe_product_go.strip() or None,
            limits=limits.get("go", {}),
        ),
        BillingPlanResponse(
            tier="plus",
            display_name="Plus",
            price_usd_monthly=plus_monthly,
            price_usd_yearly=plus_yearly,
            stripe_price_id=settings.stripe_price_plus_monthly.strip() or None,
            stripe_price_id_monthly=settings.stripe_price_plus_monthly.strip() or None,
            stripe_price_id_yearly=settings.stripe_price_plus_yearly.strip() or None,
            stripe_product_id=None,
            limits=limits.get("plus", {}),
        ),
        BillingPlanResponse(
            tier="pro",
            display_name="Pro",
            price_usd_monthly=pro_monthly,
            price_usd_yearly=pro_yearly,
            stripe_price_id=settings.stripe_price_pro_monthly.strip() or None,
            stripe_price_id_monthly=settings.stripe_price_pro_monthly.strip() or None,
            stripe_price_id_yearly=settings.stripe_price_pro_yearly.strip() or None,
            stripe_product_id=None,
            limits=limits.get("pro", {}),
        ),
    ]
    return BillingPlansResponse(plans=plans)


@router.post("/checkout-session", response_model=BillingCheckoutSessionResponse)
async def create_checkout_session(
    payload: BillingCheckoutSessionRequest,
    request: Request,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> BillingCheckoutSessionResponse:
    _require_stripe_ready()

    target_tier = _normalize_tier(payload.plan)
    target_interval = _normalize_billing_interval(payload.interval)
    price_id = _price_id_for_tier(tier=target_tier, interval=target_interval)
    if not price_id:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "BILLING_PRICE_NOT_CONFIGURED",
                "message": (
                    f"Stripe price id is missing for plan '{target_tier}' "
                    f"with interval '{target_interval}'."
                ),
            },
        )

    active_subscription = await _latest_active_subscription_for_user(
        db=db,
        user_id=user.id,
    )
    if active_subscription is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "BILLING_ACTIVE_SUBSCRIPTION_EXISTS",
                "message": "Active subscription exists. Use /billing/change-plan for plan switching.",
            },
        )

    customer = await _resolve_or_create_customer(db=db, user=user)
    remote_active_subscription = await _latest_remote_active_subscription_for_customer(
        customer_id=customer.stripe_customer_id,
    )
    if remote_active_subscription is not None:
        sync_service = SubscriptionSyncService(db, stripe_client=stripe_client)
        await sync_service.sync_subscription_payload(
            subscription_payload=remote_active_subscription,
            fallback_user_id=user.id,
        )
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "BILLING_ACTIVE_SUBSCRIPTION_EXISTS",
                "message": "Active subscription exists. Use /billing/change-plan for plan switching.",
            },
        )

    has_prior_paid_subscription = await _user_has_paid_subscription_history(
        db=db,
        user_id=user.id,
    )

    session = await _create_checkout_session_for_tier(
        customer_id=customer.stripe_customer_id,
        price_id=price_id,
        success_url=_resolve_checkout_return_url(request=request, billing="success"),
        cancel_url=_resolve_checkout_return_url(request=request, billing="cancel"),
        client_reference_id=str(user.id),
        metadata={
            "user_id": str(user.id),
            "target_tier": target_tier,
            "target_interval": target_interval,
        },
        trial_days=0 if has_prior_paid_subscription else _CHECKOUT_TRIAL_DAYS,
        target_tier=target_tier,
    )
    await db.commit()

    checkout_url = str(session.get("url") or "").strip()
    session_id = str(session.get("id") or "").strip()
    if not checkout_url or not session_id:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "code": "STRIPE_CHECKOUT_CREATE_FAILED",
                "message": "Stripe checkout session did not return url/id.",
            },
        )

    return BillingCheckoutSessionResponse(
        session_id=session_id,
        checkout_url=checkout_url,
        publishable_key=settings.stripe_publishable_key,
    )


@router.post("/portal-session", response_model=BillingPortalSessionResponse)
async def create_portal_session(
    request: Request,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> BillingPortalSessionResponse:
    _require_stripe_ready()
    customer = await _resolve_or_create_customer(db=db, user=user)

    session = await stripe_client.create_billing_portal_session(
        customer_id=customer.stripe_customer_id,
        return_url=_resolve_portal_return_url(request=request),
    )
    await db.commit()

    portal_url = str(session.get("url") or "").strip()
    if not portal_url:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "code": "STRIPE_PORTAL_CREATE_FAILED",
                "message": "Stripe billing portal session did not return url.",
            },
        )

    return BillingPortalSessionResponse(portal_url=portal_url)


@router.post("/change-plan", response_model=BillingChangePlanResponse)
async def change_billing_plan(
    payload: BillingChangePlanRequest,
    request: Request,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> BillingChangePlanResponse:
    _require_stripe_ready()

    target_tier = _normalize_tier(payload.plan)
    target_interval = _normalize_billing_interval(payload.interval)
    target_price_id = _price_id_for_tier(tier=target_tier, interval=target_interval)
    if not target_price_id:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "BILLING_PRICE_NOT_CONFIGURED",
                "message": (
                    f"Stripe price id is missing for plan '{target_tier}' "
                    f"with interval '{target_interval}'."
                ),
            },
        )

    active_subscription = await _latest_active_subscription_for_user(
        db=db,
        user_id=user.id,
    )
    subscription_payload: dict | None = None
    fallback_current_tier = _normalize_tier(user.current_tier)

    if active_subscription is None:
        customer = await _resolve_or_create_customer(db=db, user=user)
        remote_active_subscription = (
            await _latest_remote_active_subscription_for_customer(
                customer_id=customer.stripe_customer_id,
            )
        )
        if remote_active_subscription is not None:
            subscription_payload = remote_active_subscription
        else:
            has_prior_paid_subscription = await _user_has_paid_subscription_history(
                db=db,
                user_id=user.id,
            )
            checkout_session = await _create_checkout_session_for_tier(
                customer_id=customer.stripe_customer_id,
                price_id=target_price_id,
                success_url=_resolve_checkout_return_url(
                    request=request, billing="success"
                ),
                cancel_url=_resolve_checkout_return_url(
                    request=request, billing="cancel"
                ),
                client_reference_id=str(user.id),
                metadata={
                    "user_id": str(user.id),
                    "target_tier": target_tier,
                    "target_interval": target_interval,
                },
                trial_days=0 if has_prior_paid_subscription else _CHECKOUT_TRIAL_DAYS,
                target_tier=target_tier,
            )
            await db.commit()
            checkout_url = str(checkout_session.get("url") or "").strip()
            if not checkout_url:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail={
                        "code": "STRIPE_CHECKOUT_CREATE_FAILED",
                        "message": "Stripe checkout session did not return url.",
                    },
                )
            return BillingChangePlanResponse(
                action="checkout_redirect",
                current_tier=_normalize_tier(user.current_tier),
                target_tier=target_tier,
                redirect_url=checkout_url,
                publishable_key=settings.stripe_publishable_key,
            )

    if active_subscription is not None:
        fallback_current_tier = _normalize_tier(active_subscription.tier)
        subscription_id = active_subscription.stripe_subscription_id.strip()
    else:
        subscription_id = (
            str(subscription_payload.get("id") or "").strip()
            if isinstance(subscription_payload, dict)
            else ""
        )
    if not subscription_id:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "BILLING_SUBSCRIPTION_ID_MISSING",
                "message": "Active subscription id missing. Please open billing portal.",
            },
        )

    if not isinstance(subscription_payload, dict):
        subscription_payload = await stripe_client.retrieve_subscription(
            subscription_id
        )
    subscription_item_id, current_price_id = _resolve_primary_subscription_item(
        subscription_payload
    )
    if not subscription_item_id:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "code": "STRIPE_SUBSCRIPTION_ITEM_NOT_FOUND",
                "message": "Unable to resolve current subscription item.",
            },
        )

    current_tier = _tier_from_price_or_fallback(
        price_id=current_price_id,
        fallback_tier=fallback_current_tier,
    )
    if (
        current_tier == target_tier
        and (current_price_id or "").strip() == target_price_id
    ):
        sync_service = SubscriptionSyncService(db, stripe_client=stripe_client)
        await sync_service.sync_subscription_payload(
            subscription_payload=subscription_payload,
            fallback_user_id=user.id,
        )
        await db.commit()
        return BillingChangePlanResponse(
            action="noop",
            current_tier=current_tier,
            target_tier=target_tier,
        )

    current_rank = _tier_rank(current_tier)
    target_rank = _tier_rank(target_tier)
    metadata: dict[str, str] | None = None
    proration_behavior = "always_invoice"
    payment_behavior = "pending_if_incomplete"
    if target_rank < current_rank:
        proration_behavior = "none"
        payment_behavior = "allow_incomplete"
        current_period_end_raw = subscription_payload.get("current_period_end")
        if _latest_invoice_is_paid(subscription_payload) and isinstance(
            current_period_end_raw, (int, float)
        ):
            metadata = {
                "entitlements_hold_until": str(int(current_period_end_raw)),
                "pending_tier": target_tier,
            }

    updated_subscription_payload = await stripe_client.update_subscription_price(
        subscription_id,
        subscription_item_id=subscription_item_id,
        price_id=target_price_id,
        proration_behavior=proration_behavior,
        payment_behavior=payment_behavior,
        metadata=metadata,
    )
    sync_service = SubscriptionSyncService(db, stripe_client=stripe_client)
    await sync_service.sync_subscription_payload(
        subscription_payload=updated_subscription_payload,
        fallback_user_id=user.id,
    )
    await db.commit()

    return BillingChangePlanResponse(
        action="updated",
        current_tier=current_tier,
        target_tier=target_tier,
    )


@router.get("/overview", response_model=BillingOverviewResponse)
async def get_billing_overview(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> BillingOverviewResponse:
    return await _build_billing_overview_response(
        db=db,
        user_id=user.id,
        current_tier=user.current_tier,
    )


@router.get("/overview/stream")
async def stream_billing_overview(
    request: Request,
    poll_seconds: float = Query(default=3.0, ge=0.5, le=30.0),
    heartbeat_seconds: float = Query(default=90.0, ge=15.0, le=300.0),
    max_events: int | None = Query(default=None, ge=1, le=2000),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """Push billing overview snapshots whenever quota/tier state changes."""

    async def _event_stream() -> AsyncIterator[str]:
        emitted = 0
        event_seq = 0
        last_signature: str | None = None
        last_heartbeat_at = time.monotonic()

        while True:
            if await request.is_disconnected():
                break

            db.expire_all()
            current_tier = await db.scalar(
                select(User.current_tier).where(User.id == user.id)
            )
            if current_tier is None:
                break

            overview = await _build_billing_overview_response(
                db=db,
                user_id=user.id,
                current_tier=str(current_tier),
            )
            signature = _billing_overview_signature(overview)
            now_iso = datetime.now(UTC).isoformat()

            if signature != last_signature:
                event_seq += 1
                event_name = "snapshot" if last_signature is None else "quota_changed"
                payload = {
                    "server_time": now_iso,
                    "overview": overview.model_dump(mode="json"),
                }
                yield (
                    f"id: {event_seq}\n"
                    f"event: {event_name}\n"
                    f"data: {json.dumps(payload, ensure_ascii=True)}\n\n"
                )
                last_signature = signature
                last_heartbeat_at = time.monotonic()
                emitted += 1
                if max_events is not None and emitted >= max_events:
                    break
            else:
                now_monotonic = time.monotonic()
                if now_monotonic - last_heartbeat_at >= heartbeat_seconds:
                    heartbeat = {"server_time": now_iso}
                    yield f"event: heartbeat\ndata: {json.dumps(heartbeat, ensure_ascii=True)}\n\n"
                    last_heartbeat_at = now_monotonic
                    emitted += 1
                    if max_events is not None and emitted >= max_events:
                        break

            # Do not keep idle-in-transaction sessions during SSE sleeps.
            await db.rollback()
            await asyncio.sleep(poll_seconds)

        if max_events is not None:
            yield f"event: stream_end\ndata: {json.dumps({'events': emitted}, ensure_ascii=True)}\n\n"

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post(
    "/webhooks/stripe",
    response_model=BillingWebhookAckResponse,
    include_in_schema=False,
)
async def stripe_webhook(
    request: Request,
    stripe_signature: str | None = Header(default=None, alias="Stripe-Signature"),
    db: AsyncSession = Depends(get_db),
) -> BillingWebhookAckResponse:
    payload = await request.body()
    service = BillingWebhookService(db, stripe_client=stripe_client)

    try:
        result = await service.process_stripe_webhook(
            payload=payload,
            signature_header=stripe_signature,
        )
        await db.commit()
    except StripeWebhookSignatureError as exc:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "STRIPE_WEBHOOK_SIGNATURE_INVALID",
                "message": str(exc),
            },
        ) from exc
    except StripeClientConfigError as exc:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "BILLING_WEBHOOK_NOT_CONFIGURED",
                "message": str(exc),
            },
        ) from exc
    except Exception as exc:  # noqa: BLE001
        # Persist webhook failure metadata when available; fallback to rollback if commit fails.
        try:
            await db.commit()
        except Exception:  # noqa: BLE001
            await db.rollback()
        logger.exception(
            "[billing] stripe webhook processing failed event_signature_present=%s error=%s",
            bool(stripe_signature),
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "STRIPE_WEBHOOK_PROCESSING_ERROR",
                "message": "Stripe webhook processing failed. Please retry delivery.",
            },
        ) from exc

    return BillingWebhookAckResponse(
        event_id=str(result.get("event_id") or ""),
        event_type=str(result.get("event_type") or "unknown"),
        status=str(result.get("status") or "processed"),
        duplicate=bool(result.get("duplicate")),
    )


@router.get("/return", include_in_schema=False, response_class=HTMLResponse)
async def billing_return_page(request: Request) -> HTMLResponse:
    billing = normalize_billing_status(request.query_params.get("billing", ""))
    language = _resolve_request_language(request)
    app_return_raw = request.query_params.get("app_return_url")
    app_return_url = _sanitize_app_return_url(raw_url=app_return_raw, billing=billing)
    html = render_billing_return_html(
        billing=billing,
        language=language,
        app_return_url=app_return_url,
    )
    return HTMLResponse(content=html, status_code=status.HTTP_200_OK)


async def _build_billing_overview_response(
    *,
    db: AsyncSession,
    user_id,
    current_tier: str | None,
) -> BillingOverviewResponse:
    normalized_tier = _normalize_tier(current_tier)

    subscription = await db.scalar(
        select(BillingSubscription)
        .where(BillingSubscription.user_id == user_id)
        .order_by(
            BillingSubscription.updated_at.desc(),
            BillingSubscription.created_at.desc(),
        )
        .limit(1)
    )

    if subscription is None:
        subscription_response = BillingSubscriptionResponse(
            status="inactive",
            tier=normalized_tier,
            stripe_subscription_id=None,
            stripe_price_id=None,
            current_period_end=None,
            cancel_at_period_end=False,
            pending_tier=None,
            pending_price_id=None,
        )
    else:
        subscription_response = BillingSubscriptionResponse(
            status=subscription.status,
            tier=_normalize_tier(subscription.tier),
            stripe_subscription_id=subscription.stripe_subscription_id,
            stripe_price_id=subscription.stripe_price_id,
            current_period_end=subscription.current_period_end,
            cancel_at_period_end=subscription.cancel_at_period_end,
            pending_tier=_normalize_tier(subscription.pending_tier)
            if subscription.pending_tier
            else None,
            pending_price_id=subscription.pending_price_id,
        )

    quota_service = QuotaService(UsageService(db))
    snapshots = await quota_service.list_usage_overview(
        user_id=user_id,
        tier=normalized_tier,
    )
    quota_rows = [
        BillingQuotaMetricResponse(
            metric=item.metric,
            used=item.used,
            limit=item.limit,
            remaining=item.remaining,
            reset_at=item.reset_at,
        )
        for item in snapshots
    ]

    return BillingOverviewResponse(
        tier=normalized_tier,
        subscription=subscription_response,
        quotas=quota_rows,
        cost_model=settings.billing_cost_model,
    )


def _billing_overview_signature(overview: BillingOverviewResponse) -> str:
    payload = {
        "tier": overview.tier,
        "subscription": {
            "status": overview.subscription.status,
            "tier": overview.subscription.tier,
            "pending_tier": overview.subscription.pending_tier,
            "cancel_at_period_end": overview.subscription.cancel_at_period_end,
        },
        "quotas": [
            {
                "metric": row.metric,
                "used": int(row.used),
                "limit": int(row.limit),
                "remaining": int(row.remaining),
                "reset_at": row.reset_at.isoformat() if row.reset_at else None,
            }
            for row in overview.quotas
        ],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _resolve_request_language(request: Request) -> str:
    query_lang = str(request.query_params.get("lang") or "").strip().lower()
    if query_lang.startswith("zh"):
        return "zh"
    if query_lang.startswith("en"):
        return "en"

    header = str(request.headers.get("accept-language") or "").strip().lower()
    if header.startswith("zh") or ",zh" in header:
        return "zh"
    return "en"


def _require_stripe_ready() -> None:
    if not stripe_client.is_configured:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "BILLING_NOT_CONFIGURED",
                "message": "Stripe is not configured in this environment.",
            },
        )


async def _resolve_or_create_customer(
    *,
    db: AsyncSession,
    user: User,
) -> BillingCustomer:
    existing = await db.scalar(
        select(BillingCustomer).where(BillingCustomer.user_id == user.id)
    )
    if existing is not None:
        return existing

    created = await stripe_client.create_customer(
        email=user.email,
        metadata={"user_id": str(user.id)},
    )
    customer_id = str(created.get("id") or "").strip()
    if not customer_id:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "code": "STRIPE_CUSTOMER_CREATE_FAILED",
                "message": "Stripe customer create returned empty id.",
            },
        )

    sync_service = SubscriptionSyncService(db, stripe_client=stripe_client)
    return await sync_service.upsert_customer_mapping(
        user_id=user.id,
        stripe_customer_id=customer_id,
        email=user.email,
        metadata={"origin": "checkout_or_portal"},
    )


async def _latest_active_subscription_for_user(
    *,
    db: AsyncSession,
    user_id,
) -> BillingSubscription | None:
    return await db.scalar(
        select(BillingSubscription)
        .where(
            BillingSubscription.user_id == user_id,
            BillingSubscription.status.in_(_ACTIVE_SUBSCRIPTION_STATUSES),
        )
        .order_by(
            BillingSubscription.updated_at.desc(),
            BillingSubscription.created_at.desc(),
        )
        .limit(1)
    )


async def _latest_remote_active_subscription_for_customer(
    *,
    customer_id: str,
) -> dict | None:
    normalized_customer_id = customer_id.strip()
    if not normalized_customer_id:
        return None
    rows = await stripe_client.list_subscriptions(
        customer_id=normalized_customer_id,
        status="all",
        limit=20,
    )
    for row in rows:
        if not isinstance(row, dict):
            continue
        status = str(row.get("status") or "").strip().lower()
        if status in _ACTIVE_SUBSCRIPTION_STATUSES:
            return row
    return None


async def _user_has_paid_subscription_history(*, db: AsyncSession, user_id) -> bool:
    row = await db.scalar(
        select(BillingSubscription.id)
        .where(
            BillingSubscription.user_id == user_id,
            BillingSubscription.tier.in_(("go", "plus", "pro")),
        )
        .limit(1)
    )
    return row is not None


def _price_id_for_tier(*, tier: str, interval: str) -> str:
    normalized_tier = _normalize_tier(tier)
    normalized_interval = _normalize_billing_interval(interval)
    monthly_map = {
        "go": settings.stripe_price_go_monthly.strip(),
        "plus": settings.stripe_price_plus_monthly.strip(),
        "pro": settings.stripe_price_pro_monthly.strip(),
    }
    yearly_map = {
        "go": settings.stripe_price_go_yearly.strip(),
        "plus": settings.stripe_price_plus_yearly.strip(),
        "pro": settings.stripe_price_pro_yearly.strip(),
    }
    if normalized_interval == "yearly":
        return yearly_map.get(normalized_tier) or monthly_map.get(normalized_tier, "")
    return monthly_map.get(normalized_tier, "")


def _promotion_code_for_checkout_tier(tier: str) -> str | None:
    normalized_tier = _normalize_tier(tier)
    if normalized_tier not in {"plus", "pro"}:
        return None
    resolved = settings.stripe_promotion_code_plus_pro.strip()
    return resolved or None


async def _create_checkout_session_for_tier(
    *,
    customer_id: str,
    price_id: str,
    success_url: str,
    cancel_url: str,
    client_reference_id: str,
    metadata: dict[str, str],
    trial_days: int,
    target_tier: str,
) -> dict:
    promotion_code_id = _promotion_code_for_checkout_tier(target_tier)
    try:
        return await stripe_client.create_checkout_session(
            customer_id=customer_id,
            price_id=price_id,
            success_url=success_url,
            cancel_url=cancel_url,
            client_reference_id=client_reference_id,
            metadata=metadata,
            trial_days=trial_days,
            promotion_code_id=promotion_code_id,
        )
    except Exception as exc:  # noqa: BLE001
        if promotion_code_id and _is_missing_promotion_code_error(exc):
            logger.warning(
                "[billing] stripe promotion code missing; retry without promotion code "
                "tier=%s promotion_code_id=%s",
                target_tier,
                promotion_code_id,
            )
            return await stripe_client.create_checkout_session(
                customer_id=customer_id,
                price_id=price_id,
                success_url=success_url,
                cancel_url=cancel_url,
                client_reference_id=client_reference_id,
                metadata=metadata,
                trial_days=trial_days,
                promotion_code_id=None,
            )
        raise


def _tier_rank(tier: str) -> int:
    normalized = _normalize_tier(tier)
    return {
        "free": 0,
        "go": 1,
        "plus": 2,
        "pro": 3,
    }.get(normalized, 0)


def _tier_from_price_or_fallback(*, price_id: str | None, fallback_tier: str) -> str:
    mapping = settings.stripe_price_to_tier_map
    normalized = mapping.get((price_id or "").strip())
    if normalized in {"go", "plus", "pro"}:
        return normalized
    return _normalize_tier(fallback_tier)


def _latest_invoice_is_paid(subscription_payload: dict) -> bool:
    latest_invoice = subscription_payload.get("latest_invoice")
    if not isinstance(latest_invoice, dict):
        return False
    status = str(latest_invoice.get("status") or "").strip().lower()
    if status == "paid":
        return True
    return bool(latest_invoice.get("paid_out_of_band"))


def _resolve_primary_subscription_item(
    subscription_payload: dict,
) -> tuple[str | None, str | None]:
    items_root = subscription_payload.get("items")
    rows = items_root.get("data") if isinstance(items_root, dict) else []
    if not isinstance(rows, list):
        return None, None
    for item in rows:
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("id") or "").strip()
        price_raw = item.get("price")
        if isinstance(price_raw, dict):
            price_id = str(price_raw.get("id") or "").strip()
        else:
            price_id = str(price_raw or "").strip()
        if item_id:
            return item_id, price_id or None
    return None, None


def _normalize_tier(raw: str | None) -> str:
    value = (raw or "").strip().lower()
    if value in {"free", "go", "plus", "pro"}:
        return value
    return "free"


def _normalize_billing_interval(raw: str | None) -> str:
    value = (raw or "").strip().lower()
    if value == "yearly":
        return "yearly"
    return "monthly"


def _resolve_checkout_return_url(*, request: Request, billing: str) -> str:
    if billing == "cancel":
        base_url = settings.effective_billing_checkout_cancel_url
    else:
        base_url = settings.effective_billing_checkout_success_url
    app_return = _resolve_frontend_billing_return_url(request=request, billing=billing)
    return _append_app_return_url(base_url=base_url, app_return_url=app_return)


def _resolve_portal_return_url(*, request: Request) -> str:
    base_url = settings.effective_billing_portal_return_url
    app_return = _resolve_frontend_billing_return_url(request=request, billing="portal")
    return _append_app_return_url(base_url=base_url, app_return_url=app_return)


def _append_app_return_url(*, base_url: str, app_return_url: str | None) -> str:
    normalized = base_url.strip()
    if not normalized or not app_return_url:
        return normalized

    parsed = urlparse(normalized)
    query_map = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query_map["app_return_url"] = app_return_url
    return urlunparse(parsed._replace(query=urlencode(query_map)))


def _resolve_frontend_billing_return_url(
    *, request: Request, billing: str
) -> str | None:
    if billing not in {"success", "cancel", "portal"}:
        return None

    candidates: list[str] = []
    explicit = settings.billing_frontend_base_url.strip()
    if explicit:
        candidates.append(explicit)

    origin = str(request.headers.get("origin") or "").strip()
    if origin:
        candidates.append(origin)

    referer = str(request.headers.get("referer") or "").strip()
    referer_origin = _extract_origin(referer)
    if referer_origin:
        candidates.append(referer_origin)

    for candidate in candidates:
        normalized_origin = _normalize_http_origin(candidate)
        if not normalized_origin:
            continue
        if not _is_allowed_frontend_origin(normalized_origin):
            continue
        return f"{normalized_origin}/billing/return?{urlencode({'billing': billing})}"
    return None


def _sanitize_app_return_url(*, raw_url: str | None, billing: str) -> str | None:
    if not isinstance(raw_url, str):
        return None
    normalized = raw_url.strip()
    if not normalized:
        return None

    parsed = urlparse(normalized)
    origin = _normalize_http_origin(normalized)
    if not origin or not parsed.netloc:
        return None
    if not _is_allowed_frontend_origin(origin):
        return None

    query_map = dict(parse_qsl(parsed.query, keep_blank_values=True))
    if billing in {"success", "cancel", "portal"}:
        query_map["billing"] = billing
    sanitized_path = parsed.path if parsed.path else "/billing/return"
    return urlunparse(
        parsed._replace(path=sanitized_path, query=urlencode(query_map), fragment=""),
    )


def _is_allowed_frontend_origin(origin: str) -> bool:
    normalized = _normalize_http_origin(origin)
    if not normalized:
        return False

    allowed = {
        item.rstrip("/")
        for item in settings.effective_cors_origins
        if isinstance(item, str) and item.strip()
    }
    return normalized in allowed


def _extract_origin(raw_url: str) -> str | None:
    parsed = urlparse(raw_url.strip())
    if not parsed.scheme or not parsed.netloc:
        return None
    return f"{parsed.scheme}://{parsed.netloc}"


def _normalize_http_origin(raw_origin: str) -> str:
    parsed = urlparse(raw_origin.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    return f"{parsed.scheme}://{parsed.netloc}".rstrip("/")
