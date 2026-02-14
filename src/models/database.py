"""Async PostgreSQL engine/session helpers."""

from __future__ import annotations

from collections.abc import AsyncIterator
from urllib.parse import quote_plus

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.config import settings
from src.models.base import Base
from src.util.logger import logger

engine: AsyncEngine | None = None
AsyncSessionLocal: async_sessionmaker[AsyncSession] | None = None
session_factory: async_sessionmaker[AsyncSession] | None = None


def _admin_database_url() -> str:
    """Build a connection URL to a maintenance DB for CREATE DATABASE checks."""
    user = quote_plus(settings.postgres_user)
    password = quote_plus(settings.postgres_password)
    return (
        f"postgresql+asyncpg://{user}:{password}@"
        f"{settings.postgres_host}:{settings.postgres_port}/postgres"
    )


def _quote_identifier(identifier: str) -> str:
    """Safely quote SQL identifiers like database names."""
    escaped = identifier.replace('"', '""')
    return f'"{escaped}"'


def _import_all_models() -> None:
    """Import models so SQLAlchemy metadata knows all tables."""
    import src.models.backtest  # noqa: F401
    import src.models.deployment  # noqa: F401
    import src.models.phase_transition  # noqa: F401
    import src.models.session  # noqa: F401
    import src.models.strategy  # noqa: F401
    import src.models.strategy_revision  # noqa: F401
    import src.models.user  # noqa: F401


async def _ensure_sessions_phase_constraint() -> None:
    """Ensure sessions phase check-constraint includes newly added phases."""
    assert engine is not None
    phase_constraint_sql = (
        "current_phase IN "
        "('kyc', 'pre_strategy', 'strategy', 'stress_test', 'deployment', 'completed', 'error')"
    )
    async with engine.begin() as connection:
        await connection.execute(
            text("UPDATE sessions SET current_phase = 'stress_test' WHERE current_phase = 'backtest'"),
        )
        await connection.execute(
            text("UPDATE sessions SET current_phase = 'deployment' WHERE current_phase = 'deploy'"),
        )
        await connection.execute(
            text("ALTER TABLE sessions DROP CONSTRAINT IF EXISTS ck_sessions_current_phase"),
        )
        await connection.execute(
            text(
                "ALTER TABLE sessions "
                f"ADD CONSTRAINT ck_sessions_current_phase CHECK ({phase_constraint_sql})",
            ),
        )


async def _ensure_sessions_archival_columns() -> None:
    """Ensure session archival column/index exist for older deployments."""
    assert engine is not None
    async with engine.begin() as connection:
        await connection.execute(
            text("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS archived_at TIMESTAMPTZ NULL"),
        )
        await connection.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_sessions_user_archived_updated "
                "ON sessions (user_id, archived_at, updated_at)",
            ),
        )


def _ensure_engine() -> None:
    global engine, AsyncSessionLocal, session_factory
    if engine is None:
        engine = create_async_engine(
            settings.database_url,
            pool_size=settings.postgres_pool_size,
            max_overflow=settings.postgres_max_overflow,
            pool_pre_ping=True,
            echo=settings.sqlalchemy_echo,
        )
    if AsyncSessionLocal is None:
        AsyncSessionLocal = async_sessionmaker(
            bind=engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
        session_factory = AsyncSessionLocal


async def _ensure_database_exists() -> None:
    """Create target PostgreSQL database when it is missing."""
    admin_engine = create_async_engine(
        _admin_database_url(),
        pool_pre_ping=True,
        echo=settings.sqlalchemy_echo,
        isolation_level="AUTOCOMMIT",
    )

    try:
        async with admin_engine.connect() as connection:
            result = await connection.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :database_name"),
                {"database_name": settings.postgres_db},
            )
            if result.scalar() != 1:
                database_name = _quote_identifier(settings.postgres_db)
                await connection.execute(text(f"CREATE DATABASE {database_name}"))
                logger.info('PostgreSQL database "%s" created.', settings.postgres_db)
    finally:
        await admin_engine.dispose()


async def init_postgres(*, ensure_schema: bool = True) -> None:
    """Initialize database and shared engine/session factory.

    Parameters
    ----------
    ensure_schema:
        When True, ensure all tables and phase constraints exist.
    """
    await _ensure_database_exists()
    _import_all_models()
    _ensure_engine()
    assert engine is not None

    if ensure_schema:
        async with engine.begin() as connection:
            await connection.execute(text("SELECT 1"))
            await connection.run_sync(Base.metadata.create_all)
        await _ensure_sessions_phase_constraint()
        await _ensure_sessions_archival_columns()
        logger.info("PostgreSQL pool initialized and schema ensured.")
        return

    async with engine.connect() as connection:
        await connection.execute(text("SELECT 1"))
    logger.info("PostgreSQL pool initialized.")


async def close_postgres() -> None:
    """Dispose engine and reset global references."""
    global engine, AsyncSessionLocal, session_factory

    if engine is not None:
        await engine.dispose()
        logger.info("PostgreSQL pool closed.")

    engine = None
    AsyncSessionLocal = None
    session_factory = None


async def init_db(drop_existing: bool = False) -> None:
    """Create tables (optionally dropping existing tables first)."""
    await init_postgres(ensure_schema=False)
    assert engine is not None

    async with engine.begin() as connection:
        if drop_existing:
            await connection.run_sync(Base.metadata.drop_all)
        await connection.run_sync(Base.metadata.create_all)
    await _ensure_sessions_phase_constraint()
    await _ensure_sessions_archival_columns()


async def get_db_session() -> AsyncIterator[AsyncSession]:
    """Yield async session from shared factory."""
    if AsyncSessionLocal is None:
        await init_postgres()
    assert AsyncSessionLocal is not None

    async with AsyncSessionLocal() as session:
        yield session


async def postgres_healthcheck() -> bool:
    """Check PostgreSQL liveness with SELECT 1."""
    if engine is None:
        return False
    try:
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
        return True
    except Exception:
        logger.exception("PostgreSQL health check failed.")
        return False
