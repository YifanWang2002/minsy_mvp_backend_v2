"""Cleanup script for test user data.

This script:
1. Deletes deployment strategies older than 1 day for user 2@test.com
2. Deletes sessions with no user messages for user 2@test.com
"""

import asyncio
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from packages.infra.db.models.deployment import Deployment
from packages.infra.db.models.session import Message, Session
from packages.infra.db.models.strategy import Strategy
from packages.infra.db.models.user import User
from packages.infra.db.session import get_db_session, init_postgres


async def cleanup_old_deployments(db: AsyncSession, user_id: str) -> int:
    """Delete deployments older than 1 day for the given user."""
    one_day_ago = datetime.now(UTC) - timedelta(days=1)

    # Find strategies with old deployments
    stmt = (
        select(Strategy.id)
        .join(Deployment, Strategy.id == Deployment.strategy_id)
        .where(Strategy.user_id == user_id)
        .where(Deployment.created_at < one_day_ago)
        .distinct()
    )
    result = await db.execute(stmt)
    strategy_ids = [row[0] for row in result.fetchall()]

    if not strategy_ids:
        print("No old deployment strategies found.")
        return 0

    # Delete the strategies (cascades to deployments)
    delete_stmt = delete(Strategy).where(Strategy.id.in_(strategy_ids))
    result = await db.execute(delete_stmt)
    await db.commit()

    deleted_count = result.rowcount
    print(f"Deleted {deleted_count} strategies with old deployments (created before {one_day_ago})")
    return deleted_count


async def cleanup_empty_sessions(db: AsyncSession, user_id: str) -> int:
    """Delete sessions with no user messages for the given user."""
    # Find sessions with no user messages
    subquery = (
        select(Message.session_id)
        .where(Message.role == "user")
        .distinct()
        .subquery()
    )

    stmt = (
        select(Session.id, Session.created_at, func.count(Message.id).label("message_count"))
        .outerjoin(Message, Session.id == Message.session_id)
        .where(Session.user_id == user_id)
        .where(~Session.id.in_(select(subquery.c.session_id)))
        .group_by(Session.id)
    )

    result = await db.execute(stmt)
    empty_sessions = result.fetchall()

    if not empty_sessions:
        print("No empty sessions found.")
        return 0

    print(f"\nFound {len(empty_sessions)} sessions with no user messages:")
    for session_id, created_at, msg_count in empty_sessions:
        print(f"  - Session {session_id}: created {created_at}, {msg_count} total messages")

    session_ids = [row[0] for row in empty_sessions]

    # Delete the sessions (cascades to messages)
    delete_stmt = delete(Session).where(Session.id.in_(session_ids))
    result = await db.execute(delete_stmt)
    await db.commit()

    deleted_count = result.rowcount
    print(f"\nDeleted {deleted_count} empty sessions")
    return deleted_count


async def main():
    """Main cleanup routine."""
    await init_postgres(ensure_schema=False)

    async for db in get_db_session():
        # Find the test user
        stmt = select(User).where(User.email == "2@test.com")
        result = await db.execute(stmt)
        user = result.scalar_one_or_none()

        if not user:
            print("User 2@test.com not found!")
            return

        print(f"Found user: {user.email} (ID: {user.id})")
        print("=" * 60)

        # Cleanup old deployments
        print("\n1. Cleaning up old deployment strategies...")
        await cleanup_old_deployments(db, user.id)

        # Cleanup empty sessions
        print("\n2. Cleaning up empty sessions...")
        await cleanup_empty_sessions(db, user.id)

        print("\n" + "=" * 60)
        print("Cleanup completed!")


if __name__ == "__main__":
    asyncio.run(main())
