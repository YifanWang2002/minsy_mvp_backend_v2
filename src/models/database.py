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
    import src.models.broker_account  # noqa: F401
    import src.models.broker_account_audit_log  # noqa: F401
    import src.models.deployment  # noqa: F401
    import src.models.deployment_run  # noqa: F401
    import src.models.fill  # noqa: F401
    import src.models.manual_trade_action  # noqa: F401
    import src.models.market_data_error_event  # noqa: F401
    import src.models.market_data_sync_chunk  # noqa: F401
    import src.models.market_data_sync_job  # noqa: F401
    import src.models.notification_delivery_attempt  # noqa: F401
    import src.models.notification_outbox  # noqa: F401
    import src.models.optimization_trial  # noqa: F401
    import src.models.order  # noqa: F401
    import src.models.order_state_transition  # noqa: F401
    import src.models.phase_transition  # noqa: F401
    import src.models.pnl_snapshot  # noqa: F401
    import src.models.position  # noqa: F401
    import src.models.session  # noqa: F401
    import src.models.signal_event  # noqa: F401
    import src.models.social_connector  # noqa: F401
    import src.models.strategy  # noqa: F401
    import src.models.strategy_revision  # noqa: F401
    import src.models.stress_job  # noqa: F401
    import src.models.stress_job_item  # noqa: F401
    import src.models.trade_approval_request  # noqa: F401
    import src.models.trading_preference  # noqa: F401
    import src.models.trading_event_outbox  # noqa: F401
    import src.models.user  # noqa: F401
    import src.models.user_notification_preference  # noqa: F401
    import src.models.user_settings  # noqa: F401


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


async def _ensure_trading_runtime_columns() -> None:
    """Ensure incremental trading columns/constraints exist without migrations."""
    assert engine is not None
    order_status_constraint_sql = (
        "status IN ('new', 'accepted', 'pending_new', 'partially_filled', "
        "'filled', 'canceled', 'rejected', 'expired')"
    )
    async with engine.begin() as connection:
        await connection.execute(
            text("ALTER TABLE orders ADD COLUMN IF NOT EXISTS reject_reason TEXT NULL"),
        )
        await connection.execute(
            text("ALTER TABLE orders ADD COLUMN IF NOT EXISTS provider_updated_at TIMESTAMPTZ NULL"),
        )
        await connection.execute(
            text("ALTER TABLE orders ADD COLUMN IF NOT EXISTS last_sync_at TIMESTAMPTZ NULL"),
        )
        await connection.execute(
            text("ALTER TABLE orders DROP CONSTRAINT IF EXISTS ck_orders_status"),
        )
        await connection.execute(
            text(
                "ALTER TABLE orders "
                f"ADD CONSTRAINT ck_orders_status CHECK ({order_status_constraint_sql})",
            ),
        )
        await connection.execute(
            text("ALTER TABLE fills ADD COLUMN IF NOT EXISTS provider_fill_id VARCHAR(120) NULL"),
        )
        await connection.execute(
            text("CREATE INDEX IF NOT EXISTS ix_fills_provider_fill_id ON fills (provider_fill_id)"),
        )


async def _ensure_trading_event_outbox_constraint() -> None:
    """Ensure trading_event_outbox event-type constraint includes approval events."""
    assert engine is not None
    event_type_constraint_sql = (
        "event_type IN ('deployment_status', 'order_update', 'fill_update', "
        "'position_update', 'pnl_update', 'trade_approval_update', 'heartbeat')"
    )
    async with engine.begin() as connection:
        await connection.execute(
            text("ALTER TABLE trading_event_outbox DROP CONSTRAINT IF EXISTS ck_trading_event_outbox_event_type"),
        )
        await connection.execute(
            text(
                "ALTER TABLE trading_event_outbox "
                f"ADD CONSTRAINT ck_trading_event_outbox_event_type CHECK ({event_type_constraint_sql})",
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
        await _ensure_trading_runtime_columns()
        await _ensure_trading_event_outbox_constraint()
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
    await _ensure_trading_runtime_columns()
    await _ensure_trading_event_outbox_constraint()


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
