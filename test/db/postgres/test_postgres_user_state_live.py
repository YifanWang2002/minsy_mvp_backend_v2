from __future__ import annotations

import asyncio
from uuid import UUID

import asyncpg


async def _fetch_user_business_state(email: str) -> dict[str, object]:
    connection = await asyncpg.connect(
        host="127.0.0.1",
        port=5432,
        user="postgres",
        password="123456",
        database="minsy_pgsql",
    )
    try:
        user_row = await connection.fetchrow(
            "SELECT id, email, name, is_active FROM users WHERE email=$1",
            email,
        )
        assert user_row is not None, f"missing user {email}"

        user_id = user_row["id"]
        assert isinstance(user_id, UUID)

        profile_count = await connection.fetchval(
            "SELECT COUNT(*) FROM user_profiles WHERE user_id=$1",
            user_id,
        )
        settings_count = await connection.fetchval(
            "SELECT COUNT(*) FROM user_settings WHERE user_id=$1",
            user_id,
        )
        session_count = await connection.fetchval(
            "SELECT COUNT(*) FROM sessions WHERE user_id=$1",
            user_id,
        )

        return {
            "user_id": str(user_id),
            "email": str(user_row["email"]),
            "name": str(user_row["name"]),
            "is_active": bool(user_row["is_active"]),
            "profile_count": int(profile_count),
            "settings_count": int(settings_count),
            "session_count": int(session_count),
        }
    finally:
        await connection.close()


async def _check_reference_paths(user_id: str) -> dict[str, int]:
    connection = await asyncpg.connect(
        host="127.0.0.1",
        port=5432,
        user="postgres",
        password="123456",
        database="minsy_pgsql",
    )
    try:
        uid = UUID(user_id)
        strategy_count = await connection.fetchval(
            "SELECT COUNT(*) FROM strategies WHERE user_id=$1",
            uid,
        )
        deployment_count = await connection.fetchval(
            "SELECT COUNT(*) FROM deployments WHERE user_id=$1",
            uid,
        )
        backtest_count = await connection.fetchval(
            "SELECT COUNT(*) FROM backtest_jobs WHERE user_id=$1",
            uid,
        )
        return {
            "strategy_count": int(strategy_count),
            "deployment_count": int(deployment_count),
            "backtest_count": int(backtest_count),
        }
    finally:
        await connection.close()


def test_000_accessibility_seed_user_business_state() -> None:
    state = asyncio.run(_fetch_user_business_state("2@test.com"))
    assert state["email"] == "2@test.com"
    assert state["is_active"] is True
    assert state["profile_count"] >= 1


def test_010_seed_user_reference_paths_queryable() -> None:
    state = asyncio.run(_fetch_user_business_state("2@test.com"))
    refs = asyncio.run(_check_reference_paths(str(state["user_id"])))
    assert refs["strategy_count"] >= 0
    assert refs["deployment_count"] >= 0
    assert refs["backtest_count"] >= 0
