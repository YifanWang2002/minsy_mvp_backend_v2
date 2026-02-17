#!/usr/bin/env python3
"""Backfill session title metadata for existing sessions.

By default this script runs in dry-run mode and rolls back all DB writes.
Pass ``--apply`` to persist changes.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from dotenv import load_dotenv
from sqlalchemy import select

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings  # noqa: E402
from src.models import database as db_module  # noqa: E402
from src.models.session import Session  # noqa: E402
from src.services.session_title_service import (  # noqa: E402
    read_session_title_from_metadata,
    refresh_session_title,
)


@dataclass(slots=True)
class _ChangedRow:
    session_id: UUID
    phase: str
    before: str | None
    after: str | None


def _load_env() -> None:
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=False)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill metadata.session_title and metadata.session_title_record "
            "for existing sessions."
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Persist changes. Default is dry-run (rollback).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of sessions to process.",
    )
    parser.add_argument(
        "--user-id",
        type=UUID,
        default=None,
        help="Optional filter for a single user.",
    )
    parser.add_argument(
        "--session-id",
        dest="session_ids",
        action="append",
        type=UUID,
        default=None,
        help="Optional session id filter. Can be passed multiple times.",
    )
    parser.add_argument(
        "--show-changed",
        type=int,
        default=20,
        help="Max number of changed rows to print.",
    )
    return parser.parse_args()


async def _collect_session_ids(args: argparse.Namespace) -> list[UUID]:
    assert db_module.AsyncSessionLocal is not None
    async with db_module.AsyncSessionLocal() as db:
        stmt = select(Session.id).order_by(Session.updated_at.asc(), Session.id.asc())
        if args.user_id is not None:
            stmt = stmt.where(Session.user_id == args.user_id)
        if args.session_ids:
            stmt = stmt.where(Session.id.in_(args.session_ids))
        if isinstance(args.limit, int) and args.limit > 0:
            stmt = stmt.limit(args.limit)
        rows = await db.scalars(stmt)
        return list(rows)


async def _run() -> int:
    _load_env()
    args = _parse_args()

    print(
        (
            f"[session-title-backfill] mode={'apply' if args.apply else 'dry-run'} "
            f"db={settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"
        ),
    )

    await db_module.init_postgres(ensure_schema=False)
    assert db_module.AsyncSessionLocal is not None

    session_ids = await _collect_session_ids(args)
    if not session_ids:
        print("[session-title-backfill] no sessions matched filter.")
        await db_module.close_postgres()
        return 0

    processed = 0
    changed = 0
    unchanged = 0
    failed = 0
    changed_rows: list[_ChangedRow] = []

    try:
        async with db_module.AsyncSessionLocal() as db:
            for session_id in session_ids:
                session = await db.get(Session, session_id)
                if session is None:
                    failed += 1
                    continue

                processed += 1
                before = read_session_title_from_metadata(dict(session.metadata_ or {}))
                before_title = before.title
                before_record = before.record

                try:
                    after = await refresh_session_title(db=db, session=session)
                except Exception as exc:  # noqa: BLE001
                    failed += 1
                    print(
                        f"[session-title-backfill] failed session_id={session_id}: {exc}",
                    )
                    continue

                is_changed = (
                    before_title != after.title or before_record != after.record
                )
                if is_changed:
                    changed += 1
                    if len(changed_rows) < max(args.show_changed, 0):
                        changed_rows.append(
                            _ChangedRow(
                                session_id=session.id,
                                phase=session.current_phase,
                                before=before_title,
                                after=after.title,
                            ),
                        )
                else:
                    unchanged += 1

            if args.apply and failed == 0:
                await db.commit()
                commit_mode = "committed"
            else:
                await db.rollback()
                if args.apply and failed > 0:
                    commit_mode = "rolled_back_due_to_errors"
                else:
                    commit_mode = "dry_run_rolled_back"

    finally:
        await db_module.close_postgres()

    print("[session-title-backfill] summary")
    print(f"  - processed: {processed}")
    print(f"  - changed: {changed}")
    print(f"  - unchanged: {unchanged}")
    print(f"  - failed: {failed}")
    print(f"  - transaction: {commit_mode}")

    if changed_rows:
        print("[session-title-backfill] changed samples")
        for row in changed_rows:
            print(
                "  - "
                f"session_id={row.session_id} phase={row.phase} "
                f"before={row.before!r} after={row.after!r}",
            )

    if args.apply and failed > 0:
        return 1
    return 0


def main() -> int:
    return asyncio.run(_run())


if __name__ == "__main__":
    raise SystemExit(main())
