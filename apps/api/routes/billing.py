"""Billing endpoints: plans, checkout, portal, overview, webhooks."""

from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from fastapi.responses import HTMLResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.dependencies import get_db
from apps.api.middleware.auth import get_current_user
from apps.api.schemas.billing_schemas import (
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


@router.get("/plans", response_model=BillingPlansResponse)
async def get_billing_plans() -> BillingPlansResponse:
    limits = settings.billing_tier_limits
    pricing = settings.billing_cost_model

    plans = [
        BillingPlanResponse(
            tier="free",
            display_name="Free",
            price_usd_monthly=0,
            stripe_price_id=None,
            stripe_product_id=None,
            limits=limits.get("free", {}),
        ),
        BillingPlanResponse(
            tier="go",
            display_name="Go",
            price_usd_monthly=float(pricing.get("go_price_usd", 8.0)),
            stripe_price_id=settings.stripe_price_go_monthly.strip() or None,
            stripe_product_id=settings.stripe_product_go.strip() or None,
            limits=limits.get("go", {}),
        ),
        BillingPlanResponse(
            tier="plus",
            display_name="Plus",
            price_usd_monthly=float(pricing.get("plus_price_usd", 20.0)),
            stripe_price_id=settings.stripe_price_plus_monthly.strip() or None,
            stripe_product_id=None,
            limits=limits.get("plus", {}),
        ),
        BillingPlanResponse(
            tier="pro",
            display_name="Pro",
            price_usd_monthly=float(pricing.get("pro_price_usd", 60.0)),
            stripe_price_id=settings.stripe_price_pro_monthly.strip() or None,
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

    price_map = {
        "go": settings.stripe_price_go_monthly.strip(),
        "plus": settings.stripe_price_plus_monthly.strip(),
        "pro": settings.stripe_price_pro_monthly.strip(),
    }
    price_id = price_map.get(payload.plan, "")
    if not price_id:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "BILLING_PRICE_NOT_CONFIGURED",
                "message": f"Stripe price id is missing for plan '{payload.plan}'.",
            },
        )

    customer = await _resolve_or_create_customer(db=db, user=user)

    session = await stripe_client.create_checkout_session(
        customer_id=customer.stripe_customer_id,
        price_id=price_id,
        success_url=_resolve_checkout_return_url(request=request, billing="success"),
        cancel_url=_resolve_checkout_return_url(request=request, billing="cancel"),
        client_reference_id=str(user.id),
        metadata={
            "user_id": str(user.id),
            "target_tier": payload.plan,
        },
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


@router.get("/overview", response_model=BillingOverviewResponse)
async def get_billing_overview(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> BillingOverviewResponse:
    normalized_tier = _normalize_tier(user.current_tier)

    subscription = await db.scalar(
        select(BillingSubscription)
        .where(BillingSubscription.user_id == user.id)
        .order_by(BillingSubscription.updated_at.desc(), BillingSubscription.created_at.desc())
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
            pending_tier=_normalize_tier(subscription.pending_tier) if subscription.pending_tier else None,
            pending_price_id=subscription.pending_price_id,
        )

    quota_service = QuotaService(UsageService(db))
    snapshots = await quota_service.list_usage_overview(
        user_id=user.id,
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


def _normalize_tier(raw: str | None) -> str:
    value = (raw or "").strip().lower()
    if value in {"free", "go", "plus", "pro"}:
        return value
    return "free"


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


def _resolve_frontend_billing_return_url(*, request: Request, billing: str) -> str | None:
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
