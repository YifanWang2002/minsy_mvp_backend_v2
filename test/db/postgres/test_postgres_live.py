from __future__ import annotations

import asyncio

import asyncpg
import bcrypt


async def _fetch_seed_user() -> asyncpg.Record | None:
    connection = await asyncpg.connect(
        host="127.0.0.1",
        port=5432,
        user="postgres",
        password="123456",
        database="minsy_pgsql",
    )
    try:
        return await connection.fetchrow(
            "SELECT id, email, password_hash FROM users WHERE email=$1",
            "2@test.com",
        )
    finally:
        await connection.close()


def test_000_accessibility_postgres_seed_user_exists() -> None:
    row = asyncio.run(_fetch_seed_user())
    assert row is not None, "Expected seeded user 2@test.com in local PostgreSQL(5432)."


def test_010_seed_user_password_matches_plaintext() -> None:
    row = asyncio.run(_fetch_seed_user())
    assert row is not None, "Expected seeded user 2@test.com in local PostgreSQL(5432)."
    password_hash = str(row["password_hash"]).encode("utf-8")
    assert bcrypt.checkpw(b"123456", password_hash), "Seeded password does not match 123456."
