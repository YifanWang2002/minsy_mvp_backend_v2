from __future__ import annotations

import sys
from pathlib import Path

import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings  # noqa: E402
from src.models import database as db_module  # noqa: E402


def _force_test_database_name() -> None:
    """Protect local primary DB by forcing pytest into a *_test database."""
    current = settings.postgres_db.strip()
    if current.endswith("_test"):
        return
    settings.postgres_db = f"{current}_test"


@pytest_asyncio.fixture
async def db_session() -> AsyncSession:
    _force_test_database_name()
    await db_module.close_postgres()
    await db_module.init_db(drop_existing=True)
    assert db_module.AsyncSessionLocal is not None
    async with db_module.AsyncSessionLocal() as session:
        yield session
    await db_module.close_postgres()
