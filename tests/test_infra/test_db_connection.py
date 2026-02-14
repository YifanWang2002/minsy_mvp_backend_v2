import asyncpg
import pytest

from src.config import settings


@pytest.mark.asyncio
async def test_db_connection_select_1() -> None:
    conn = await asyncpg.connect(
        host=settings.postgres_host,
        port=settings.postgres_port,
        user=settings.postgres_user,
        password=settings.postgres_password,
        database=settings.postgres_db,
    )
    try:
        value = await conn.fetchval("SELECT 1")
    finally:
        await conn.close()

    assert value == 1
