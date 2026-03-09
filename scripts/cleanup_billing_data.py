"""Cleanup billing data for Stripe environment migration.

Default behavior is dry-run. Use --apply to execute deletions.

Examples:
  # Dry-run: auto-detect users with test-mode billing footprints.
  uv run python scripts/cleanup_billing_data.py

  # Apply cleanup for specific users.
  uv run python scripts/cleanup_billing_data.py \
    --emails alice@example.com bob@example.com \
    --apply

  # Apply cleanup for all targeted users and reset usage ledgers too.
  uv run python scripts/cleanup_billing_data.py \
    --all-users \
    --mode test_only \
    --delete-usage \
    --apply
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from sqlalchemy import Boolean, cast, delete, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

# Allow running as `python scripts/cleanup_billing_data.py` from backend root.
BACKEND_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from packages.infra.db.models.billing_customer import BillingCustomer
from packages.infra.db.models.billing_subscription import BillingSubscription
from packages.infra.db.models.billing_usage_event import BillingUsageEvent
from packages.infra.db.models.billing_usage_monthly import BillingUsageMonthly
from packages.infra.db.models.billing_webhook_event import BillingWebhookEvent
from packages.infra.db.models.user import User
from packages.infra.db.session import close_postgres, get_db_session, init_postgres

ACTIVE_SUBSCRIPTION_STATUSES: tuple[str, ...] = (
    "trialing",
    "active",
    "past_due",
    "unpaid",
    "paused",
)


@dataclass(slots=True)
class CleanupPlan:
    target_user_ids: set[UUID]
    subscription_ids_to_delete: list[UUID]
    webhook_ids_to_delete: list[UUID]
    customer_ids_to_delete: list[UUID]
    usage_event_ids_to_delete: list[UUID]
    usage_monthly_ids_to_delete: list[UUID]
    user_tier_updates: dict[UUID, str]
    skipped_users_without_account: set[UUID]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cleanup billing tables safely (dry-run by default).",
    )
    parser.add_argument(
        "--emails",
        nargs="*",
        default=[],
        help="Target users by email.",
    )
    parser.add_argument(
        "--user-ids",
        nargs="*",
        default=[],
        help="Target users by UUID.",
    )
    parser.add_argument(
        "--all-users",
        action="store_true",
        help="Target all users (dangerous, still dry-run unless --apply).",
    )
    parser.add_argument(
        "--mode",
        choices=("test_only", "live_only", "all"),
        default="test_only",
        help=(
            "Which subscription/webhook records to clean: "
            "test_only (default), live_only, or all."
        ),
    )
    parser.add_argument(
        "--include-missing-livemode-as-test",
        action="store_true",
        help=(
            "When --mode=test_only, treat subscription rows without raw_payload.livemode "
            "as test rows."
        ),
    )
    parser.add_argument(
        "--delete-usage",
        action="store_true",
        help="Also delete billing_usage_events and billing_usage_monthly for target users.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Execute changes. Without this flag, script only prints the plan.",
    )
    return parser.parse_args()


def _normalize_emails(raw_values: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in raw_values:
        email = value.strip().lower()
        if not email or email in seen:
            continue
        seen.add(email)
        normalized.append(email)
    return normalized


def _parse_user_ids(raw_values: Iterable[str]) -> list[UUID]:
    parsed: list[UUID] = []
    seen: set[UUID] = set()
    for value in raw_values:
        text = value.strip()
        if not text:
            continue
        try:
            item = UUID(text)
        except ValueError as exc:
            raise ValueError(f"Invalid UUID in --user-ids: {text}") from exc
        if item in seen:
            continue
        seen.add(item)
        parsed.append(item)
    return parsed


def _subscription_livemode_filter(
    *,
    mode: str,
    include_missing_as_test: bool,
):
    if mode == "all":
        return None

    livemode_expr = cast(BillingSubscription.raw_payload["livemode"].astext, Boolean)
    if mode == "live_only":
        return livemode_expr.is_(True)
    if mode == "test_only":
        if include_missing_as_test:
            return or_(livemode_expr.is_(False), livemode_expr.is_(None))
        return livemode_expr.is_(False)
    raise ValueError(f"Unsupported mode: {mode}")


def _webhook_livemode_filter(*, mode: str):
    if mode == "all":
        return None
    if mode == "live_only":
        return BillingWebhookEvent.livemode.is_(True)
    if mode == "test_only":
        return BillingWebhookEvent.livemode.is_(False)
    raise ValueError(f"Unsupported mode: {mode}")


async def _resolve_target_user_ids(
    db: AsyncSession,
    *,
    emails: list[str],
    user_ids: list[UUID],
    all_users: bool,
    mode: str,
    include_missing_as_test: bool,
) -> set[UUID]:
    target_ids: set[UUID] = set()

    if emails:
        rows = await db.execute(select(User.id).where(User.email.in_(emails)))
        target_ids.update(rows.scalars().all())

    if user_ids:
        rows = await db.execute(select(User.id).where(User.id.in_(user_ids)))
        target_ids.update(rows.scalars().all())

    if all_users:
        rows = await db.execute(select(User.id))
        target_ids.update(rows.scalars().all())
        return target_ids

    if target_ids:
        return target_ids

    # Auto-discover users from billing footprints if caller didn't provide explicit filters.
    sub_filter = _subscription_livemode_filter(
        mode=mode,
        include_missing_as_test=include_missing_as_test,
    )
    sub_stmt = select(BillingSubscription.user_id).distinct()
    if sub_filter is not None:
        sub_stmt = sub_stmt.where(sub_filter)
    rows = await db.execute(sub_stmt)
    target_ids.update(rows.scalars().all())

    webhook_filter = _webhook_livemode_filter(mode=mode)
    webhook_stmt = (
        select(BillingWebhookEvent.user_id)
        .where(BillingWebhookEvent.user_id.is_not(None))
        .distinct()
    )
    if webhook_filter is not None:
        webhook_stmt = webhook_stmt.where(webhook_filter)
    rows = await db.execute(webhook_stmt)
    target_ids.update(item for item in rows.scalars().all() if item is not None)

    if mode == "all":
        customer_rows = await db.execute(select(BillingCustomer.user_id).distinct())
        target_ids.update(customer_rows.scalars().all())

    return target_ids


async def _compute_remaining_effective_tier(
    db: AsyncSession,
    *,
    user_id: UUID,
    deleted_subscription_ids: set[UUID],
) -> str:
    stmt = (
        select(BillingSubscription)
        .where(
            BillingSubscription.user_id == user_id,
            BillingSubscription.status.in_(ACTIVE_SUBSCRIPTION_STATUSES),
        )
        .order_by(BillingSubscription.updated_at.desc(), BillingSubscription.created_at.desc())
    )
    rows = (await db.execute(stmt)).scalars().all()
    for row in rows:
        if row.id in deleted_subscription_ids:
            continue
        tier = str(row.tier or "").strip().lower()
        if tier in {"go", "plus", "pro"}:
            return tier
        return "free"
    return "free"


async def _build_cleanup_plan(
    db: AsyncSession,
    *,
    emails: list[str],
    user_ids: list[UUID],
    all_users: bool,
    mode: str,
    include_missing_as_test: bool,
    delete_usage: bool,
) -> CleanupPlan:
    target_user_ids = await _resolve_target_user_ids(
        db,
        emails=emails,
        user_ids=user_ids,
        all_users=all_users,
        mode=mode,
        include_missing_as_test=include_missing_as_test,
    )
    if not target_user_ids:
        return CleanupPlan(
            target_user_ids=set(),
            subscription_ids_to_delete=[],
            webhook_ids_to_delete=[],
            customer_ids_to_delete=[],
            usage_event_ids_to_delete=[],
            usage_monthly_ids_to_delete=[],
            user_tier_updates={},
            skipped_users_without_account=set(),
        )

    sub_stmt = select(BillingSubscription.id, BillingSubscription.user_id).where(
        BillingSubscription.user_id.in_(target_user_ids),
    )
    sub_filter = _subscription_livemode_filter(
        mode=mode,
        include_missing_as_test=include_missing_as_test,
    )
    if sub_filter is not None:
        sub_stmt = sub_stmt.where(sub_filter)
    sub_rows = (await db.execute(sub_stmt)).all()
    subscription_ids_to_delete = [row[0] for row in sub_rows]
    deleted_subscription_ids = set(subscription_ids_to_delete)

    webhook_stmt = select(BillingWebhookEvent.id, BillingWebhookEvent.user_id).where(
        BillingWebhookEvent.user_id.in_(target_user_ids),
    )
    webhook_filter = _webhook_livemode_filter(mode=mode)
    if webhook_filter is not None:
        webhook_stmt = webhook_stmt.where(webhook_filter)
    webhook_rows = (await db.execute(webhook_stmt)).all()
    webhook_ids_to_delete = [row[0] for row in webhook_rows]

    # Delete customer mappings only if user has no subscription rows remaining after cleanup.
    customer_rows = (
        await db.execute(
            select(BillingCustomer.id, BillingCustomer.user_id).where(
                BillingCustomer.user_id.in_(target_user_ids),
            )
        )
    ).all()
    customer_ids_to_delete: list[UUID] = []
    for customer_id, customer_user_id in customer_rows:
        remaining_for_user = await db.execute(
            select(BillingSubscription.id).where(
                BillingSubscription.user_id == customer_user_id,
            ),
        )
        remaining_ids = set(remaining_for_user.scalars().all())
        if remaining_ids.issubset(deleted_subscription_ids):
            customer_ids_to_delete.append(customer_id)

    usage_event_ids_to_delete: list[UUID] = []
    usage_monthly_ids_to_delete: list[UUID] = []
    if delete_usage:
        usage_event_rows = (
            await db.execute(
                select(BillingUsageEvent.id).where(
                    BillingUsageEvent.user_id.in_(target_user_ids),
                )
            )
        ).all()
        usage_event_ids_to_delete = [row[0] for row in usage_event_rows]

        usage_monthly_rows = (
            await db.execute(
                select(BillingUsageMonthly.id).where(
                    BillingUsageMonthly.user_id.in_(target_user_ids),
                )
            )
        ).all()
        usage_monthly_ids_to_delete = [row[0] for row in usage_monthly_rows]

    existing_user_rows = (
        await db.execute(
            select(User.id, User.current_tier).where(User.id.in_(target_user_ids))
        )
    ).all()
    existing_tier_by_user: dict[UUID, str] = {
        user_id: str(current_tier or "").strip().lower() or "free"
        for user_id, current_tier in existing_user_rows
    }
    existing_user_ids = set(existing_tier_by_user.keys())
    skipped_users_without_account = target_user_ids - existing_user_ids

    user_tier_updates: dict[UUID, str] = {}
    for user_id in existing_user_ids:
        target_tier = await _compute_remaining_effective_tier(
            db,
            user_id=user_id,
            deleted_subscription_ids=deleted_subscription_ids,
        )
        current_tier = existing_tier_by_user.get(user_id, "free")
        if current_tier != target_tier:
            user_tier_updates[user_id] = target_tier

    return CleanupPlan(
        target_user_ids=target_user_ids,
        subscription_ids_to_delete=subscription_ids_to_delete,
        webhook_ids_to_delete=webhook_ids_to_delete,
        customer_ids_to_delete=customer_ids_to_delete,
        usage_event_ids_to_delete=usage_event_ids_to_delete,
        usage_monthly_ids_to_delete=usage_monthly_ids_to_delete,
        user_tier_updates=user_tier_updates,
        skipped_users_without_account=skipped_users_without_account,
    )


async def _apply_cleanup_plan(db: AsyncSession, plan: CleanupPlan) -> None:
    if plan.subscription_ids_to_delete:
        await db.execute(
            delete(BillingSubscription).where(
                BillingSubscription.id.in_(plan.subscription_ids_to_delete),
            )
        )

    if plan.webhook_ids_to_delete:
        await db.execute(
            delete(BillingWebhookEvent).where(
                BillingWebhookEvent.id.in_(plan.webhook_ids_to_delete),
            )
        )

    if plan.usage_event_ids_to_delete:
        await db.execute(
            delete(BillingUsageEvent).where(
                BillingUsageEvent.id.in_(plan.usage_event_ids_to_delete),
            )
        )

    if plan.usage_monthly_ids_to_delete:
        await db.execute(
            delete(BillingUsageMonthly).where(
                BillingUsageMonthly.id.in_(plan.usage_monthly_ids_to_delete),
            )
        )

    if plan.customer_ids_to_delete:
        await db.execute(
            delete(BillingCustomer).where(
                BillingCustomer.id.in_(plan.customer_ids_to_delete),
            )
        )

    for user_id, target_tier in plan.user_tier_updates.items():
        await db.execute(
            update(User)
            .where(User.id == user_id)
            .values(current_tier=target_tier),
        )

    await db.commit()


async def _render_user_preview(db: AsyncSession, user_ids: set[UUID], limit: int = 12) -> list[str]:
    if not user_ids:
        return []
    rows = (
        await db.execute(
            select(User.id, User.email, User.current_tier)
            .where(User.id.in_(user_ids))
            .order_by(User.email.asc())
            .limit(limit)
        )
    ).all()
    rendered: list[str] = []
    for user_id, email, tier in rows:
        rendered.append(f"{email} ({user_id}) tier={tier}")
    return rendered


async def _run() -> int:
    args = _parse_args()
    emails = _normalize_emails(args.emails)
    try:
        user_ids = _parse_user_ids(args.user_ids)
    except ValueError as exc:
        print(f"[error] {exc}")
        return 2

    await init_postgres(ensure_schema=False)
    try:
        async for db in get_db_session():
            plan = await _build_cleanup_plan(
                db,
                emails=emails,
                user_ids=user_ids,
                all_users=bool(args.all_users),
                mode=str(args.mode),
                include_missing_as_test=bool(args.include_missing_livemode_as_test),
                delete_usage=bool(args.delete_usage),
            )

            print("=== Billing Cleanup Plan ===")
            print(f"mode: {args.mode}")
            print(f"target_users: {len(plan.target_user_ids)}")
            print(f"subscriptions_to_delete: {len(plan.subscription_ids_to_delete)}")
            print(f"webhooks_to_delete: {len(plan.webhook_ids_to_delete)}")
            print(f"customers_to_delete: {len(plan.customer_ids_to_delete)}")
            print(f"usage_events_to_delete: {len(plan.usage_event_ids_to_delete)}")
            print(f"usage_monthly_to_delete: {len(plan.usage_monthly_ids_to_delete)}")
            print(f"user_tier_updates: {len(plan.user_tier_updates)}")

            preview = await _render_user_preview(db, plan.target_user_ids, limit=12)
            if preview:
                print("sample_target_users:")
                for line in preview:
                    print(f"  - {line}")

            if plan.skipped_users_without_account:
                print(
                    f"skipped_missing_users: {len(plan.skipped_users_without_account)} "
                    "(not found in users table)"
                )

            if not args.apply:
                print("dry_run: true (no data was changed)")
                return 0

            await _apply_cleanup_plan(db, plan)
            print("dry_run: false")
            print("status: cleanup applied successfully")
            return 0
    finally:
        await close_postgres()

    return 0


def main() -> None:
    exit_code = asyncio.run(_run())
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
