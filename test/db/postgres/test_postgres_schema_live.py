from __future__ import annotations

import asyncio

import asyncpg


async def _fetch_public_table_names() -> set[str]:
    connection = await asyncpg.connect(
        host="127.0.0.1",
        port=5432,
        user="postgres",
        password="123456",
        database="minsy_pgsql",
    )
    try:
        rows = await connection.fetch(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='public'"
        )
        return {str(row["table_name"]) for row in rows}
    finally:
        await connection.close()


async def _fetch_key_table_counts() -> dict[str, int]:
    tables = (
        "users",
        "sessions",
        "messages",
        "strategies",
        "strategy_revisions",
        "deployments",
        "deployment_runs",
        "orders",
        "positions",
        "backtest_jobs",
    )
    connection = await asyncpg.connect(
        host="127.0.0.1",
        port=5432,
        user="postgres",
        password="123456",
        database="minsy_pgsql",
    )
    try:
        counts: dict[str, int] = {}
        for table in tables:
            row = await connection.fetchrow(f"SELECT COUNT(*) AS c FROM {table}")
            counts[table] = int(row["c"])
        return counts
    finally:
        await connection.close()


def test_000_accessibility_postgres_key_tables_exist() -> None:
    names = asyncio.run(_fetch_public_table_names())
    required = {
        "users",
        "sessions",
        "messages",
        "strategies",
        "strategy_revisions",
        "deployments",
        "deployment_runs",
        "orders",
        "positions",
        "backtest_jobs",
        "broker_accounts",
        "trade_approval_requests",
    }
    missing = required - names
    assert not missing, f"Missing tables: {sorted(missing)}"


def test_010_postgres_key_table_count_queries_executable() -> None:
    counts = asyncio.run(_fetch_key_table_counts())
    assert all(value >= 0 for value in counts.values())
    assert counts["users"] >= 1
