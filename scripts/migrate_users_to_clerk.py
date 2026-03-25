"""Migrate local development users into Clerk using bcrypt password digests.

Default behavior is dry-run. Use --apply to create/update Clerk users and
backfill local `users.clerk_user_id`.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

BACKEND_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND_DIR))

os.environ.setdefault("APP_ENV", "dev")
os.environ.setdefault("MINSY_SERVICE", "api")
os.environ.setdefault(
    "MINSY_ENV_FILES",
    ",".join(
        [
            "env/.env.secrets",
            "env/.env.common",
            "env/.env.dev",
            "env/.env.dev.api",
            "env/.env.dev.localtest",
        ]
    ),
)

from packages.infra.db.models.user import User
from packages.infra.db.session import close_postgres, get_db_session, init_postgres
from packages.infra.providers.clerk.client import ClerkClient


@dataclass(frozen=True, slots=True)
class LocalUserSnapshot:
    user_id: str
    email: str
    name: str
    password_hash: str | None
    clerk_user_id: str | None
    current_tier: str
    auth_provider: str


@dataclass(slots=True)
class MigrationResult:
    user_id: str
    email: str
    local_clerk_user_id: str | None
    remote_clerk_user_id: str | None
    action: str
    reason: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import local users into Clerk using bcrypt password digests.",
    )
    parser.add_argument("--emails", nargs="*", default=[], help="Filter by email.")
    parser.add_argument("--user-ids", nargs="*", default=[], help="Filter by UUID.")
    parser.add_argument(
        "--exclude-domain",
        nargs="*",
        default=[],
        help="Skip emails whose domain matches any provided value.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit migrated users.")
    parser.add_argument(
        "--csv-report",
        default="runtime/reports/clerk_dev_migration.csv",
        help="Path to write migration report CSV.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Persist to Clerk and backfill local users. Default is dry-run.",
    )
    return parser.parse_args()


def _normalize_emails(values: list[str]) -> list[str]:
    seen: set[str] = set()
    normalized: list[str] = []
    for value in values:
        email = value.strip().lower()
        if not email or email in seen:
            continue
        normalized.append(email)
        seen.add(email)
    return normalized


def _normalize_domains(values: list[str]) -> list[str]:
    seen: set[str] = set()
    normalized: list[str] = []
    for value in values:
        domain = value.strip().lower().lstrip("@")
        if not domain or domain in seen:
            continue
        normalized.append(domain)
        seen.add(domain)
    return normalized


def _parse_user_ids(values: list[str]) -> list[UUID]:
    parsed: list[UUID] = []
    seen: set[UUID] = set()
    for value in values:
        text = value.strip()
        if not text:
            continue
        item = UUID(text)
        if item in seen:
            continue
        parsed.append(item)
        seen.add(item)
    return parsed


def _snapshot_from_user(user: User) -> LocalUserSnapshot:
    return LocalUserSnapshot(
        user_id=str(user.id),
        email=user.email.strip().lower(),
        name=user.name.strip(),
        password_hash=user.password_hash,
        clerk_user_id=(user.clerk_user_id or "").strip() or None,
        current_tier=str(user.current_tier or "").strip().lower() or "free",
        auth_provider=str(user.auth_provider or "").strip().lower() or "legacy_password",
    )


def _split_display_name(name: str) -> tuple[str | None, str | None]:
    normalized = name.strip()
    if not normalized:
        return None, None
    parts = [item for item in normalized.split() if item]
    if not parts:
        return None, None
    first_name = parts[0]
    last_name = " ".join(parts[1:]) or None
    return first_name, last_name


def build_clerk_create_payload(snapshot: LocalUserSnapshot) -> dict[str, Any]:
    first_name, last_name = _split_display_name(snapshot.name)
    payload: dict[str, Any] = {
        "email_address": [snapshot.email],
        "external_id": snapshot.user_id,
        "skip_password_checks": True,
        "skip_password_requirement": True,
        "private_metadata": {
            "local_user_id": snapshot.user_id,
            "local_auth_provider": snapshot.auth_provider,
        },
        "public_metadata": {
            "tier": snapshot.current_tier,
        },
    }
    if first_name:
        payload["first_name"] = first_name
    if last_name:
        payload["last_name"] = last_name
    if snapshot.password_hash and snapshot.password_hash.strip():
        payload["password_digest"] = snapshot.password_hash.strip()
        payload["password_hasher"] = "bcrypt"
    return payload


async def _resolve_remote_user(
    snapshot: LocalUserSnapshot,
    client: ClerkClient,
) -> dict[str, Any] | None:
    if snapshot.clerk_user_id:
        matched = await client.get_user(snapshot.clerk_user_id)
        if matched is not None:
            return matched
    matched = await client.find_user_by_external_id(snapshot.user_id)
    if matched is not None:
        return matched
    return await client.find_user_by_email(snapshot.email)


async def _plan_migration(
    snapshot: LocalUserSnapshot,
    client: ClerkClient,
    *,
    apply: bool,
    assume_empty_remote: bool = False,
) -> tuple[MigrationResult, dict[str, Any] | None]:
    if not snapshot.email:
        return (
            MigrationResult(
                user_id=snapshot.user_id,
                email="",
                local_clerk_user_id=snapshot.clerk_user_id,
                remote_clerk_user_id=None,
                action="skipped",
                reason="missing_email",
            ),
            None,
        )

    if snapshot.password_hash is None or not snapshot.password_hash.strip():
        return (
            MigrationResult(
                user_id=snapshot.user_id,
                email=snapshot.email,
                local_clerk_user_id=snapshot.clerk_user_id,
                remote_clerk_user_id=None,
                action="skipped",
                reason="missing_password_hash",
            ),
            None,
        )

    existing = None
    if not assume_empty_remote or snapshot.clerk_user_id:
        existing = await _resolve_remote_user(snapshot, client)
    if existing is not None:
        clerk_user_id = str(existing.get("id") or "").strip() or None
        if apply and clerk_user_id is not None:
            await client.update_user(
                clerk_user_id,
                {
                    "external_id": snapshot.user_id,
                    "private_metadata": {
                        "local_user_id": snapshot.user_id,
                        "local_auth_provider": snapshot.auth_provider,
                    },
                },
            )
            await client.update_user_metadata(
                clerk_user_id,
                public_metadata={"tier": snapshot.current_tier},
                private_metadata={
                    "local_user_id": snapshot.user_id,
                    "local_auth_provider": snapshot.auth_provider,
                },
            )
        return (
            MigrationResult(
                user_id=snapshot.user_id,
                email=snapshot.email,
                local_clerk_user_id=snapshot.clerk_user_id,
                remote_clerk_user_id=clerk_user_id,
                action="matched_existing" if apply else "would_match_existing",
                reason="existing_remote_user",
            ),
            existing,
        )

    payload = build_clerk_create_payload(snapshot)
    if not apply:
        return (
            MigrationResult(
                user_id=snapshot.user_id,
                email=snapshot.email,
                local_clerk_user_id=snapshot.clerk_user_id,
                remote_clerk_user_id=None,
                action="would_create",
                reason="ready_for_import",
            ),
            payload,
        )

    created = await client.create_user(payload)
    clerk_user_id = str(created.get("id") or "").strip() or None
    return (
        MigrationResult(
            user_id=snapshot.user_id,
            email=snapshot.email,
            local_clerk_user_id=snapshot.clerk_user_id,
            remote_clerk_user_id=clerk_user_id,
            action="created",
            reason="imported_with_password_digest",
        ),
        created,
    )


async def _load_target_users(
    db: AsyncSession,
    *,
    emails: list[str],
    user_ids: list[UUID],
    excluded_domains: list[str],
    limit: int,
) -> list[User]:
    stmt = select(User).order_by(User.created_at.asc())
    if emails:
        stmt = stmt.where(User.email.in_(emails))
    if user_ids:
        stmt = stmt.where(User.id.in_(user_ids))
    for domain in excluded_domains:
        normalized = domain.strip().lower()
        if normalized:
            stmt = stmt.where(~User.email.like(f"%@{normalized}"))
    if limit > 0:
        stmt = stmt.limit(limit)
    rows = await db.execute(stmt)
    return list(rows.scalars().all())


async def _backfill_local_user(
    db: AsyncSession,
    user: User,
    *,
    clerk_user_id: str | None,
) -> None:
    if clerk_user_id is None or not clerk_user_id.strip():
        return
    user.clerk_user_id = clerk_user_id.strip()
    user.auth_provider = "clerk"
    await db.commit()


def _write_csv_report(path: Path, rows: list[MigrationResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "user_id",
                "email",
                "local_clerk_user_id",
                "remote_clerk_user_id",
                "action",
                "reason",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


async def _run(args: argparse.Namespace) -> int:
    client = ClerkClient()
    await init_postgres()
    report_rows: list[MigrationResult] = []
    remote_users = await client.list_users(limit=1)
    assume_empty_remote = len(remote_users) == 0

    try:
        async for db in get_db_session():
            users = await _load_target_users(
                db,
                emails=_normalize_emails(args.emails),
                user_ids=_parse_user_ids(args.user_ids),
                excluded_domains=_normalize_domains(args.exclude_domain),
                limit=max(int(args.limit or 0), 0),
            )
            print(
                f"Loaded {len(users)} local users from dev database. "
                f"remote_instance_empty={assume_empty_remote}"
            )

            for user in users:
                snapshot = _snapshot_from_user(user)
                result, _ = await _plan_migration(
                    snapshot,
                    client,
                    apply=args.apply,
                    assume_empty_remote=assume_empty_remote,
                )
                report_rows.append(result)
                print(
                    f"[{result.action}] email={result.email} "
                    f"local_clerk_user_id={result.local_clerk_user_id or '-'} "
                    f"remote_clerk_user_id={result.remote_clerk_user_id or '-'} "
                    f"reason={result.reason}"
                )
                if args.apply and result.remote_clerk_user_id:
                    await _backfill_local_user(
                        db,
                        user,
                        clerk_user_id=result.remote_clerk_user_id,
                    )

            break
    finally:
        await close_postgres()

    report_path = BACKEND_DIR / str(args.csv_report)
    _write_csv_report(report_path, report_rows)
    print(f"Wrote CSV report to {report_path}")

    created = sum(1 for row in report_rows if row.action == "created")
    matched = sum(
        1
        for row in report_rows
        if row.action in {"matched_existing", "would_match_existing"}
    )
    skipped = sum(1 for row in report_rows if row.action == "skipped")
    print(
        f"Summary: created={created} matched_existing={matched} skipped={skipped} "
        f"mode={'apply' if args.apply else 'dry-run'}"
    )
    return 0


def main() -> int:
    return asyncio.run(_run(_parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
