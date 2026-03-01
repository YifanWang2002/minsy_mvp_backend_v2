"""Cleanup script for test user data - simplified version.

This script:
1. Deletes deployment strategies older than 1 day for user 2@test.com
2. Deletes sessions with no user messages for user 2@test.com
"""

import asyncio
from datetime import UTC, datetime, timedelta

import asyncpg


async def main():
    """Main cleanup routine."""
    # Connect directly to PostgreSQL
    conn = await asyncpg.connect(
        host="localhost",
        port=5432,
        user="postgres",
        password="123456",
        database="minsy_pgsql",
    )

    try:
        # Find the test user
        user = await conn.fetchrow(
            "SELECT id, email FROM users WHERE email = $1", "2@test.com"
        )

        if not user:
            print("User 2@test.com not found!")
            return

        user_id = user["id"]
        print(f"Found user: {user['email']} (ID: {user_id})")
        print("=" * 60)

        # 1. Cleanup old deployment strategies
        print("\n1. Cleaning up old deployment strategies...")
        one_day_ago = datetime.now(UTC) - timedelta(days=1)

        # Find strategies with old deployments
        old_strategies = await conn.fetch(
            """
            SELECT DISTINCT s.id, s.name, s.created_at
            FROM strategies s
            JOIN deployments d ON s.id = d.strategy_id
            WHERE s.user_id = $1 AND d.created_at < $2
            ORDER BY s.created_at
            """,
            user_id,
            one_day_ago,
        )

        if old_strategies:
            print(f"Found {len(old_strategies)} strategies with old deployments:")
            for strat in old_strategies:
                print(f"  - {strat['name']} (created: {strat['created_at']})")

            strategy_ids = [s["id"] for s in old_strategies]

            # Delete the strategies (cascades to deployments)
            deleted = await conn.execute(
                "DELETE FROM strategies WHERE id = ANY($1)", strategy_ids
            )
            print(f"\nDeleted {len(strategy_ids)} strategies with old deployments")
        else:
            print("No old deployment strategies found.")

        # 2. Cleanup empty sessions
        print("\n2. Cleaning up empty sessions...")

        # Find sessions with no user messages
        empty_sessions = await conn.fetch(
            """
            SELECT s.id, s.created_at,
                   COUNT(m.id) FILTER (WHERE m.role = 'user') as user_msg_count,
                   COUNT(m.id) as total_msg_count
            FROM sessions s
            LEFT JOIN messages m ON s.id = m.session_id
            WHERE s.user_id = $1
            GROUP BY s.id
            HAVING COUNT(m.id) FILTER (WHERE m.role = 'user') = 0
            ORDER BY s.created_at
            """,
            user_id,
        )

        if empty_sessions:
            print(f"Found {len(empty_sessions)} sessions with no user messages:")
            for sess in empty_sessions:
                print(
                    f"  - Session {sess['id']}: created {sess['created_at']}, "
                    f"{sess['total_msg_count']} total messages"
                )

            session_ids = [s["id"] for s in empty_sessions]

            # Delete the sessions (cascades to messages)
            deleted = await conn.execute(
                "DELETE FROM sessions WHERE id = ANY($1)", session_ids
            )
            print(f"\nDeleted {len(session_ids)} empty sessions")
        else:
            print("No empty sessions found.")

        print("\n" + "=" * 60)
        print("Cleanup completed!")

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
