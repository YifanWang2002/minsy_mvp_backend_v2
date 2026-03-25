"""Verify local dev users are mapped to Clerk users after migration."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from sqlalchemy import select

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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify local Clerk user mappings against Clerk dev instance.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit verified rows.")
    return parser.parse_args()


async def _run(limit: int) -> int:
    client = ClerkClient()
    await init_postgres()
    missing = 0
    verified = 0

    try:
        remote_users = await client.list_users(limit=max(limit, 1) if limit > 0 else 500)
        remote_by_id = {
            str(user.get("id") or "").strip(): user
            for user in remote_users
            if str(user.get("id") or "").strip()
        }
        async for db in get_db_session():
            stmt = (
                select(User)
                .where(User.clerk_user_id.is_not(None))
                .order_by(User.created_at.asc())
            )
            if limit > 0:
                stmt = stmt.limit(limit)
            rows = (await db.execute(stmt)).scalars().all()
            for user in rows:
                remote = remote_by_id.get(str(user.clerk_user_id))
                if remote is None:
                    missing += 1
                    print(
                        f"[missing] email={user.email} clerk_user_id={user.clerk_user_id}"
                    )
                    continue
                verified += 1
                print(
                    f"[ok] email={user.email} clerk_user_id={user.clerk_user_id} "
                    f"remote_external_id={remote.get('external_id')}"
                )
            break
    finally:
        await close_postgres()

    print(f"Verified={verified} missing={missing}")
    return 0 if missing == 0 else 1


def main() -> int:
    args = _parse_args()
    return asyncio.run(_run(args.limit))


if __name__ == "__main__":
    raise SystemExit(main())
