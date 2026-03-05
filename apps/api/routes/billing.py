"""Billing endpoints: plans, checkout, portal, overview, webhooks."""

from __future__ import annotations

from html import escape
from urllib.parse import parse_qsl, quote, urlencode, urlparse, urlunparse

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
from apps.api.services.billing_webhook_service import BillingWebhookService
from packages.domain.billing.quota_service import QuotaService
from packages.domain.billing.subscription_sync_service import SubscriptionSyncService
from packages.domain.billing.usage_service import UsageService
from packages.infra.db.models.billing_customer import BillingCustomer
from packages.infra.db.models.billing_subscription import BillingSubscription
from packages.infra.db.models.user import User
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
            limits=limits.get("free", {}),
        ),
        BillingPlanResponse(
            tier="plus",
            display_name="Plus",
            price_usd_monthly=float(pricing.get("plus_price_usd", 20.0)),
            stripe_price_id=settings.stripe_price_plus_monthly.strip() or None,
            limits=limits.get("plus", {}),
        ),
        BillingPlanResponse(
            tier="pro",
            display_name="Pro",
            price_usd_monthly=float(pricing.get("pro_price_usd", 60.0)),
            stripe_price_id=settings.stripe_price_pro_monthly.strip() or None,
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
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "STRIPE_WEBHOOK_PROCESSING_ERROR",
                "message": f"{type(exc).__name__}: {exc}",
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
    billing_raw = request.query_params.get("billing", "")
    billing = (billing_raw or "").strip().lower()
    if billing in {"failed", "fail", "error"}:
        billing = "cancel"
    elif billing == "succeeded":
        billing = "success"

    language = _resolve_request_language(request)
    is_zh = language == "zh"

    if billing == "success":
        status_kind = "success"
        tone = "#16A34A"
    elif billing in {"cancel"}:
        status_kind = "failed"
        tone = "#DC2626"
    else:
        status_kind = "neutral"
        tone = "#2563EB"

    if is_zh:
        title_map = {
            "success": "支付已完成",
            "failed": "支付未完成",
            "neutral": "账单状态已更新",
        }
        subtitle_map = {
            "success": "你的 Stripe 支付已成功提交。",
            "failed": "本次支付未完成，你可以随时重新尝试。",
            "neutral": "你可以返回应用查看当前订阅状态。",
        }
        thanks_map = {
            "success": "感谢你的支持，我们正在后台同步你的订阅信息。",
            "failed": "感谢你体验 Minsy。支付失败通常与银行卡、网络或 Stripe 校验相关。",
            "neutral": "感谢你使用账单流程，我们正在同步最新账单状态。",
        }
        badge_map = {
            "success": "成功",
            "failed": "失败",
            "neutral": "同步中",
        }
        support_hint = "如果遇到付款问题、功能 bug，或者想优先体验最新功能，欢迎加入我们的 Telegram demo 用户群，直接与我们沟通。"
        group_label = "Telegram 演示用户群"
        join_hint = "扫码或点击下方链接即可加入："
        copy_hint = "若无法打开链接，请手动复制。"
        action_label = "打开订阅管理"
    else:
        title_map = {
            "success": "Payment completed",
            "failed": "Payment not completed",
            "neutral": "Billing status updated",
        }
        subtitle_map = {
            "success": "Your Stripe payment went through successfully.",
            "failed": "Your payment was not completed. You can retry anytime.",
            "neutral": "You can return to the app and review your subscription status.",
        }
        thanks_map = {
            "success": "Thanks for your support. We are syncing your subscription details in the background.",
            "failed": "Thanks for trying Minsy. Payment can fail due to card, network, or Stripe validation issues.",
            "neutral": "Thanks for visiting our billing flow. Your latest billing state is being synchronized.",
        }
        badge_map = {
            "success": "Success",
            "failed": "Failed",
            "neutral": "In sync",
        }
        support_hint = "If you hit payment issues, feature bugs, or want early access to the newest beta capabilities, join our Telegram demo group and talk to us directly."
        group_label = "Telegram demo user group"
        join_hint = "Scan the QR code or open the link below:"
        copy_hint = "If opening fails, copy the link manually."
        action_label = "Open subscription"

    title = title_map.get(status_kind, title_map["neutral"])
    subtitle = subtitle_map.get(status_kind, subtitle_map["neutral"])
    thanks = thanks_map.get(status_kind, thanks_map["neutral"])
    badge = badge_map.get(status_kind, badge_map["neutral"])

    app_return_raw = request.query_params.get("app_return_url")
    app_return_url = _sanitize_app_return_url(raw_url=app_return_raw, billing=billing)
    action_html = ""
    if app_return_url:
        action_html = (
            "<div class=\"actions\">"
            f"<a class=\"button primary\" href=\"{escape(app_return_url)}\" rel=\"noreferrer\">"
            f"{escape(action_label)}"
            "</a>"
            "</div>"
        )

    ring_html = ""
    if status_kind in {"success", "failed"}:
        ring_html = f"<div class=\"ring-orbit ring-{status_kind}\" aria-hidden=\"true\"></div>"

    telegram_link = "https://t.me/+vbxwOIwswK43Yzc1"
    qr_src = (
        "https://api.qrserver.com/v1/create-qr-code/?size=220x220&data="
        + quote(telegram_link, safe="")
    )

    html = f"""<!doctype html>
<html lang="{escape(language)}">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    :root {{
      --bg: #ffffff;
      --surface-subtle: #fafafa;
      --border: #e5e5e5;
      --text-primary: #0d0d0d;
      --text-secondary: #474747;
      --text-muted: #6b6b6b;
      --accent: #3b82f6;
      --card-shadow-1: 0 1px 3px rgba(0,0,0,0.06);
      --card-shadow-2: 0 1px 2px rgba(0,0,0,0.04);
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      min-height: 100vh;
      font-family: Inter, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "Noto Sans CJK SC", "Source Han Sans SC", sans-serif;
      color: var(--text-primary);
      background:
        radial-gradient(circle at 15% 10%, rgba(59,130,246,0.08), transparent 34%),
        radial-gradient(circle at 85% 80%, rgba(24,24,27,0.05), transparent 30%),
        var(--bg);
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 20px;
    }}
    .card {{
      width: min(760px, 100%);
      border-radius: 12px;
      border: 1px solid var(--border);
      background: var(--bg);
      box-shadow: var(--card-shadow-1), var(--card-shadow-2);
      padding: 22px 24px 20px;
      position: relative;
      z-index: 1;
    }}
    .card-shell {{
      width: min(760px, 100%);
      position: relative;
      border-radius: 12px;
    }}
    .ring-orbit {{
      position: absolute;
      inset: -3px;
      border-radius: 14px;
      pointer-events: none;
      overflow: hidden;
      z-index: 0;
      filter: saturate(1.08);
    }}
    .ring-orbit::before {{
      content: "";
      position: absolute;
      left: -50%;
      top: -50%;
      width: 200%;
      height: 200%;
      background: conic-gradient(
        from 0deg,
        transparent 0deg,
        transparent 300deg,
        var(--ring-color) 328deg,
        transparent 360deg
      );
      animation: ring-spin 3s linear infinite;
      filter: blur(10px);
      opacity: 0.9;
    }}
    .ring-success {{
      --ring-color: rgba(22, 163, 74, 0.88);
    }}
    .ring-failed {{
      --ring-color: rgba(220, 38, 38, 0.88);
    }}
    @keyframes ring-spin {{
      0% {{
        transform: rotate(0deg);
      }}
      100% {{
        transform: rotate(360deg);
      }}
    }}
    .header {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 12px;
    }}
    .title {{
      margin: 0;
      font-size: 24px;
      line-height: 1.3;
      font-weight: 600;
      color: var(--text-primary);
    }}
    .subtitle {{
      margin: 6px 0 0;
      font-size: 13px;
      line-height: 1.5;
      color: var(--text-secondary);
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      border-radius: 999px;
      border: 1px solid {tone};
      background: rgba(255, 255, 255, 0.9);
      color: {tone};
      padding: 4px 8px;
      font-size: 11px;
      line-height: 1.15;
      font-weight: 700;
      white-space: nowrap;
      margin-top: 2px;
    }}
    .thanks {{
      margin: 18px 0 0;
      font-size: 13px;
      line-height: 1.55;
      color: var(--text-primary);
    }}
    .support {{
      margin: 8px 0 0;
      font-size: 13px;
      line-height: 1.55;
      color: var(--text-secondary);
    }}
    .telegram {{
      margin-top: 16px;
      width: 100%;
      padding: 12px;
      border-radius: 10px;
      border: 1px solid var(--border);
      background: var(--surface-subtle);
      display: grid;
      grid-template-columns: 110px 1fr;
      gap: 12px;
      align-items: start;
    }}
    .qr-wrap {{
      width: 110px;
      height: 110px;
      border-radius: 8px;
      border: 1px solid var(--border);
      overflow: hidden;
      background: var(--surface-subtle);
    }}
    .qr-wrap img {{
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
    }}
    .group-label {{
      margin: 6px 0 0;
      font-size: 12px;
      line-height: 1.25;
      font-weight: 600;
      color: var(--text-secondary);
    }}
    .join-hint {{
      margin: 10px 0 0;
      font-size: 12px;
      line-height: 1.35;
      color: var(--text-primary);
    }}
    .tg-link {{
      margin-top: 10px;
      display: inline-block;
      font-size: 12px;
      line-height: 1.35;
      font-weight: 600;
      color: var(--accent);
      text-decoration: underline;
      text-decoration-color: var(--accent);
      word-break: break-all;
    }}
    .copy-hint {{
      margin: 6px 0 0;
      font-size: 11px;
      line-height: 1.35;
      color: var(--text-secondary);
    }}
    .actions {{
      margin-top: 16px;
      display: flex;
      gap: 10px;
    }}
    .button {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 34px;
      border-radius: 8px;
      padding: 8px 16px;
      text-decoration: none;
      font-size: 12px;
      line-height: 1.1;
      font-weight: 600;
      border: 1px solid var(--border);
      color: var(--text-primary);
      background: #fff;
    }}
    .button.primary {{
      background: #18181b;
      border-color: #18181b;
      color: #fafafa;
    }}
    @media (max-width: 560px) {{
      .header {{
        flex-direction: column;
        align-items: flex-start;
      }}
      .telegram {{
        grid-template-columns: 1fr;
      }}
      .group-label {{
        margin-top: 0;
      }}
    }}
  </style>
</head>
<body>
  <div class="card-shell status-{escape(status_kind)}">
    {ring_html}
    <main class="card">
      <div class="header">
        <div>
          <h1 class="title">{escape(title)}</h1>
          <p class="subtitle">{escape(subtitle)}</p>
        </div>
        <span class="badge">{escape(badge)}</span>
      </div>
      <p class="thanks">{escape(thanks)}</p>
      <p class="support">{escape(support_hint)}</p>
      <section class="telegram">
        <div class="qr-wrap">
          <img src="{escape(qr_src)}" alt="Telegram QR code">
        </div>
        <div>
          <p class="group-label">{escape(group_label)}</p>
          <p class="join-hint">{escape(join_hint)}</p>
          <a class="tg-link" href="{escape(telegram_link)}" rel="noreferrer noopener">{escape(telegram_link)}</a>
          <p class="copy-hint">{escape(copy_hint)}</p>
        </div>
      </section>
      {action_html}
    </main>
  </div>
</body>
</html>"""
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
    if value in {"free", "plus", "pro"}:
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


def _candidate_frontend_billing_return_urls(*, billing: str) -> list[tuple[str, str]]:
    if billing not in {"success", "cancel", "portal"}:
        return []

    unique_origins: list[str] = []
    seen: set[str] = set()

    explicit = _normalize_http_origin(settings.billing_frontend_base_url)
    if explicit and explicit not in seen and _is_allowed_frontend_origin(explicit):
        seen.add(explicit)
        unique_origins.append(explicit)

    for candidate in settings.effective_cors_origins:
        origin = _normalize_http_origin(candidate)
        if not origin or origin in seen:
            continue
        parsed = urlparse(origin)
        hostname = (parsed.hostname or "").strip().lower()
        if hostname not in {"localhost", "127.0.0.1"}:
            continue
        seen.add(origin)
        unique_origins.append(origin)

    rows: list[tuple[str, str]] = []
    for origin in unique_origins[:3]:
        parsed = urlparse(origin)
        host_port = parsed.netloc
        label = f"Open {host_port}"
        url = f"{origin}/billing/return?{urlencode({'billing': billing})}"
        rows.append((label, url))
    return rows


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
