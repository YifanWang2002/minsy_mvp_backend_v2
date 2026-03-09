"""Sync Stripe subscription payloads into local billing tables."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from packages.infra.db.models.billing_customer import BillingCustomer
from packages.infra.db.models.billing_subscription import BillingSubscription
from packages.infra.db.models.user import User
from packages.infra.observability.logger import logger
from packages.infra.providers.stripe.client import StripeClient
from packages.shared_settings.schema.settings import settings

_ACTIVE_TIER_STATUSES: frozenset[str] = frozenset({"trialing", "active", "past_due", "unpaid"})


class SubscriptionSyncService:
    """Persistence service for webhook/checkout-driven billing sync."""

    def __init__(self, db: AsyncSession, *, stripe_client: StripeClient) -> None:
        self._db = db
        self._stripe = stripe_client

    async def upsert_customer_mapping(
        self,
        *,
        user_id: UUID,
        stripe_customer_id: str,
        email: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BillingCustomer:
        normalized_customer_id = stripe_customer_id.strip()
        existing = await self._db.scalar(
            select(BillingCustomer).where(BillingCustomer.user_id == user_id)
        )
        if existing is None:
            existing = await self._db.scalar(
                select(BillingCustomer).where(
                    BillingCustomer.stripe_customer_id == normalized_customer_id,
                )
            )

        if existing is None:
            existing = BillingCustomer(
                user_id=user_id,
                stripe_customer_id=normalized_customer_id,
            )
            self._db.add(existing)

        existing.user_id = user_id
        existing.stripe_customer_id = normalized_customer_id
        existing.email = email
        existing.metadata_ = dict(metadata or {})
        existing.synced_at = datetime.now(UTC)
        await self._db.flush()
        return existing

    async def sync_from_checkout_session(
        self,
        *,
        checkout_session: dict[str, Any],
        latest_event_id: str | None = None,
    ) -> BillingSubscription | None:
        customer_id = str(checkout_session.get("customer") or "").strip()
        if not customer_id:
            return None

        metadata_raw = checkout_session.get("metadata")
        metadata = dict(metadata_raw) if isinstance(metadata_raw, dict) else {}
        user_id_raw = str(
            metadata.get("user_id")
            or checkout_session.get("client_reference_id")
            or ""
        ).strip()
        user_id: UUID | None = None
        if user_id_raw:
            try:
                user_id = UUID(user_id_raw)
            except ValueError:
                user_id = None

        customer_details = checkout_session.get("customer_details")
        customer_email_raw = customer_details.get("email") if isinstance(customer_details, dict) else None
        customer_email = customer_email_raw.strip() if isinstance(customer_email_raw, str) else None
        resolved_user_id: UUID | None = None
        if isinstance(user_id, UUID):
            user = await self._db.scalar(select(User).where(User.id == user_id))
            if user is None:
                logger.warning(
                    "[billing] checkout session references missing user; skip mapping user_id=%s stripe_customer_id=%s",
                    user_id,
                    customer_id,
                )
            else:
                resolved_user_id = user.id
        if resolved_user_id is not None:
            await self.upsert_customer_mapping(
                user_id=resolved_user_id,
                stripe_customer_id=customer_id,
                email=customer_email or None,
                metadata=metadata,
            )

        subscription_id = str(checkout_session.get("subscription") or "").strip()
        if not subscription_id:
            return None

        subscription_payload = await self._stripe.retrieve_subscription(subscription_id)
        return await self.sync_subscription_payload(
            subscription_payload=subscription_payload,
            fallback_user_id=resolved_user_id,
            latest_event_id=latest_event_id,
        )

    async def sync_subscription_payload(
        self,
        *,
        subscription_payload: dict[str, Any],
        fallback_user_id: UUID | None = None,
        latest_event_id: str | None = None,
    ) -> BillingSubscription | None:
        stripe_subscription_id = str(subscription_payload.get("id") or "").strip()
        if not stripe_subscription_id:
            return None

        stripe_customer_id = str(subscription_payload.get("customer") or "").strip()
        user_id = await self._resolve_user_id(
            stripe_customer_id=stripe_customer_id,
            fallback_user_id=fallback_user_id,
        )
        if user_id is None:
            logger.warning(
                "[billing] unable to map subscription to user stripe_subscription_id=%s stripe_customer_id=%s",
                stripe_subscription_id,
                stripe_customer_id,
            )
            return None

        customer = await self._db.scalar(
            select(BillingCustomer).where(BillingCustomer.user_id == user_id)
        )

        resolved_price_id, resolved_tier = self._resolve_current_price_and_tier(subscription_payload)
        pending_price_id, pending_tier = self._resolve_pending_price_and_tier(subscription_payload)
        status = str(subscription_payload.get("status") or "inactive").strip().lower() or "inactive"
        effective_tier = resolved_tier if status in _ACTIVE_TIER_STATUSES else "free"
        metadata_raw = subscription_payload.get("metadata")
        metadata = dict(metadata_raw) if isinstance(metadata_raw, dict) else {}
        hold_entitlements_until = _resolve_hold_entitlements_until(
            metadata.get("entitlements_hold_until")
        )
        metadata_pending_tier = _normalize_tier_value(metadata.get("pending_tier"))

        record = await self._db.scalar(
            select(BillingSubscription).where(
                BillingSubscription.stripe_subscription_id == stripe_subscription_id,
            )
        )
        if record is None:
            record = BillingSubscription(
                user_id=user_id,
                customer_id=customer.id if customer is not None else None,
                stripe_customer_id=stripe_customer_id,
                stripe_subscription_id=stripe_subscription_id,
            )
            self._db.add(record)

        if (
            status in _ACTIVE_TIER_STATUSES
            and hold_entitlements_until is not None
            and datetime.now(UTC) < hold_entitlements_until
        ):
            current_record_tier = _normalize_tier_value(record.tier)
            if current_record_tier and _tier_rank(resolved_tier) < _tier_rank(current_record_tier):
                effective_tier = current_record_tier
                pending_tier = metadata_pending_tier or resolved_tier
                pending_price_id = pending_price_id or resolved_price_id

        record.user_id = user_id
        record.customer_id = customer.id if customer is not None else None
        record.stripe_customer_id = stripe_customer_id
        record.stripe_price_id = resolved_price_id
        record.tier = effective_tier
        record.status = status
        record.cancel_at_period_end = bool(subscription_payload.get("cancel_at_period_end"))
        record.current_period_start = _from_stripe_timestamp(subscription_payload.get("current_period_start"))
        record.current_period_end = _from_stripe_timestamp(subscription_payload.get("current_period_end"))
        record.trial_start = _from_stripe_timestamp(subscription_payload.get("trial_start"))
        record.trial_end = _from_stripe_timestamp(subscription_payload.get("trial_end"))
        record.canceled_at = _from_stripe_timestamp(subscription_payload.get("canceled_at"))
        record.ended_at = _from_stripe_timestamp(subscription_payload.get("ended_at"))
        record.pending_price_id = pending_price_id
        record.pending_tier = pending_tier
        record.latest_invoice_id = _resolve_latest_invoice_id(subscription_payload)
        record.latest_event_id = latest_event_id
        record.raw_payload = dict(subscription_payload)
        record.synced_at = datetime.now(UTC)
        await self._db.flush()

        await self._sync_user_tier(user_id=user_id, target_tier=effective_tier)
        return record

    async def apply_invoice_event(
        self,
        *,
        invoice_payload: dict[str, Any],
        latest_event_id: str | None = None,
    ) -> BillingSubscription | None:
        subscription_id = str(invoice_payload.get("subscription") or "").strip()
        if not subscription_id:
            return None

        subscription_payload = await self._stripe.retrieve_subscription(subscription_id)
        return await self.sync_subscription_payload(
            subscription_payload=subscription_payload,
            latest_event_id=latest_event_id,
        )

    async def mark_subscription_ended(
        self,
        *,
        stripe_subscription_id: str,
        ended_at: datetime | None = None,
        latest_event_id: str | None = None,
    ) -> BillingSubscription | None:
        record = await self._db.scalar(
            select(BillingSubscription).where(
                BillingSubscription.stripe_subscription_id == stripe_subscription_id,
            )
        )
        if record is None:
            return None

        record.status = "canceled"
        record.tier = "free"
        record.pending_tier = None
        record.pending_price_id = None
        record.ended_at = ended_at or datetime.now(UTC)
        record.latest_event_id = latest_event_id
        record.synced_at = datetime.now(UTC)
        await self._db.flush()
        await self._sync_user_tier(user_id=record.user_id, target_tier="free")
        return record

    async def _resolve_user_id(
        self,
        *,
        stripe_customer_id: str,
        fallback_user_id: UUID | None,
    ) -> UUID | None:
        if stripe_customer_id:
            mapping = await self._db.scalar(
                select(BillingCustomer).where(BillingCustomer.stripe_customer_id == stripe_customer_id)
            )
            if mapping is not None:
                return mapping.user_id

        if fallback_user_id is not None:
            user = await self._db.scalar(select(User).where(User.id == fallback_user_id))
            if user is not None:
                if stripe_customer_id:
                    await self.upsert_customer_mapping(
                        user_id=fallback_user_id,
                        stripe_customer_id=stripe_customer_id,
                    )
                return fallback_user_id
            logger.warning(
                "[billing] subscription fallback user missing; skip sync fallback_user_id=%s stripe_customer_id=%s",
                fallback_user_id,
                stripe_customer_id,
            )
        return None

    async def _sync_user_tier(self, *, user_id: UUID, target_tier: str) -> None:
        user = await self._db.scalar(select(User).where(User.id == user_id))
        if user is None:
            return
        user.current_tier = (
            target_tier if target_tier in {"free", "go", "plus", "pro"} else "free"
        )
        await self._db.flush()

    @staticmethod
    def _resolve_current_price_and_tier(subscription_payload: dict[str, Any]) -> tuple[str | None, str]:
        price_to_tier = settings.stripe_price_to_tier_map
        product_to_tier = settings.stripe_product_to_tier_map
        items_root = subscription_payload.get("items")
        items = items_root.get("data") if isinstance(items_root, dict) else []
        if not isinstance(items, list):
            items = []

        fallback_price: str | None = None
        for item in items:
            if not isinstance(item, dict):
                continue
            price_raw = item.get("price")
            price = dict(price_raw) if isinstance(price_raw, dict) else {}
            price_id = str(price.get("id") or item.get("price") or "").strip()
            if not price_id:
                continue
            fallback_price = fallback_price or price_id
            product_id = str(price.get("product") or "").strip()
            tier = price_to_tier.get(price_id) or product_to_tier.get(product_id)
            if tier in {"go", "plus", "pro"}:
                return price_id, tier

        if fallback_price is not None:
            return fallback_price, "free"
        return None, "free"

    @staticmethod
    def _resolve_pending_price_and_tier(subscription_payload: dict[str, Any]) -> tuple[str | None, str | None]:
        price_to_tier = settings.stripe_price_to_tier_map
        product_to_tier = settings.stripe_product_to_tier_map
        pending_raw = subscription_payload.get("pending_update")
        pending = dict(pending_raw) if isinstance(pending_raw, dict) else {}
        subscription_items = pending.get("subscription_items")
        if not isinstance(subscription_items, list):
            return None, None

        for item in subscription_items:
            if not isinstance(item, dict):
                continue
            price_raw = item.get("price")
            price_id = ""
            if isinstance(price_raw, dict):
                price_id = str(price_raw.get("id") or "").strip()
            elif isinstance(price_raw, str):
                price_id = price_raw.strip()
            if not price_id:
                continue
            product_id = (
                str(price_raw.get("product") or "").strip()
                if isinstance(price_raw, dict)
                else ""
            )
            return price_id, price_to_tier.get(price_id) or product_to_tier.get(product_id)
        return None, None


def _from_stripe_timestamp(raw: Any) -> datetime | None:
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw.astimezone(UTC)
    if isinstance(raw, (int, float)):
        try:
            return datetime.fromtimestamp(float(raw), tz=UTC)
        except Exception:  # noqa: BLE001
            return None
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        normalized = text.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    return None


def _resolve_latest_invoice_id(subscription_payload: dict[str, Any]) -> str | None:
    raw = subscription_payload.get("latest_invoice")
    if isinstance(raw, str):
        text = raw.strip()
        return text or None
    if isinstance(raw, dict):
        text = str(raw.get("id") or "").strip()
        return text or None
    return None


def _normalize_tier_value(raw: Any) -> str | None:
    value = str(raw or "").strip().lower()
    if value in {"free", "go", "plus", "pro"}:
        return value
    return None


def _tier_rank(raw: str) -> int:
    return {
        "free": 0,
        "go": 1,
        "plus": 2,
        "pro": 3,
    }.get(_normalize_tier_value(raw) or "free", 0)


def _resolve_hold_entitlements_until(raw: Any) -> datetime | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        try:
            return datetime.fromtimestamp(float(raw), tz=UTC)
        except Exception:  # noqa: BLE001
            return None
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        try:
            as_int = int(text)
            return datetime.fromtimestamp(as_int, tz=UTC)
        except ValueError:
            pass
        normalized = text.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    return None
