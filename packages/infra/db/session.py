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

from packages.infra.db.models.base import Base
from packages.infra.observability.logger import logger
from packages.shared_settings.schema.settings import settings

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
    import packages.infra.db.models.backtest  # noqa: F401
    import packages.infra.db.models.billing_customer  # noqa: F401
    import packages.infra.db.models.billing_subscription  # noqa: F401
    import packages.infra.db.models.billing_usage_event  # noqa: F401
    import packages.infra.db.models.billing_usage_monthly  # noqa: F401
    import packages.infra.db.models.billing_webhook_event  # noqa: F401
    import packages.infra.db.models.broker_account  # noqa: F401
    import packages.infra.db.models.broker_account_audit_log  # noqa: F401
    import packages.infra.db.models.deployment  # noqa: F401
    import packages.infra.db.models.deployment_run  # noqa: F401
    import packages.infra.db.models.fill  # noqa: F401
    import packages.infra.db.models.manual_trade_action  # noqa: F401
    import packages.infra.db.models.market_data_catalog  # noqa: F401
    import packages.infra.db.models.market_data_error_event  # noqa: F401
    import packages.infra.db.models.market_data_sync_chunk  # noqa: F401
    import packages.infra.db.models.market_data_sync_job  # noqa: F401
    import packages.infra.db.models.notification_delivery_attempt  # noqa: F401
    import packages.infra.db.models.notification_outbox  # noqa: F401
    import packages.infra.db.models.optimization_trial  # noqa: F401
    import packages.infra.db.models.order  # noqa: F401
    import packages.infra.db.models.order_state_transition  # noqa: F401
    import packages.infra.db.models.phase_transition  # noqa: F401
    import packages.infra.db.models.pnl_snapshot  # noqa: F401
    import packages.infra.db.models.position  # noqa: F401
    import packages.infra.db.models.sandbox_ledger_entry  # noqa: F401
    import packages.infra.db.models.session  # noqa: F401
    import packages.infra.db.models.signal_event  # noqa: F401
    import packages.infra.db.models.social_connector  # noqa: F401
    import packages.infra.db.models.strategy  # noqa: F401
    import packages.infra.db.models.strategy_revision  # noqa: F401
    import packages.infra.db.models.stress_job  # noqa: F401
    import packages.infra.db.models.stress_job_item  # noqa: F401
    import packages.infra.db.models.trade_approval_request  # noqa: F401
    import packages.infra.db.models.trading_event_outbox  # noqa: F401
    import packages.infra.db.models.trading_preference  # noqa: F401
    import packages.infra.db.models.user  # noqa: F401
    import packages.infra.db.models.user_notification_preference  # noqa: F401
    import packages.infra.db.models.user_settings  # noqa: F401


async def _ensure_sessions_phase_constraint() -> None:
    """Ensure sessions phase check-constraint includes newly added phases."""
    assert engine is not None
    phase_constraint_sql = (
        "current_phase IN "
        "('kyc', 'pre_strategy', 'strategy', 'stress_test', 'deployment', 'completed', 'error')"
    )
    async with engine.begin() as connection:
        await connection.execute(
            text(
                "UPDATE sessions SET current_phase = 'stress_test' WHERE current_phase = 'backtest'"
            ),
        )
        await connection.execute(
            text(
                "UPDATE sessions SET current_phase = 'deployment' WHERE current_phase = 'deploy'"
            ),
        )
        await connection.execute(
            text(
                "ALTER TABLE sessions DROP CONSTRAINT IF EXISTS ck_sessions_current_phase"
            ),
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
            text(
                "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS archived_at TIMESTAMPTZ NULL"
            ),
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
            text(
                "ALTER TABLE orders ADD COLUMN IF NOT EXISTS provider_updated_at TIMESTAMPTZ NULL"
            ),
        )
        await connection.execute(
            text(
                "ALTER TABLE orders ADD COLUMN IF NOT EXISTS last_sync_at TIMESTAMPTZ NULL"
            ),
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
            text(
                "ALTER TABLE fills ADD COLUMN IF NOT EXISTS provider_fill_id VARCHAR(120) NULL"
            ),
        )
        await connection.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_fills_provider_fill_id ON fills (provider_fill_id)"
            ),
        )


async def _ensure_trading_event_outbox_constraint() -> None:
    """Ensure trading_event_outbox event-type constraint includes approval events."""
    assert engine is not None
    event_type_constraint_sql = (
        "event_type IN ('deployment_status', 'order_update', 'fill_update', "
        "'position_update', 'pnl_update', 'manual_action_update', 'trade_approval_update', 'heartbeat')"
    )
    async with engine.begin() as connection:
        await connection.execute(
            text(
                "ALTER TABLE trading_event_outbox DROP CONSTRAINT IF EXISTS ck_trading_event_outbox_event_type"
            ),
        )
        await connection.execute(
            text(
                "ALTER TABLE trading_event_outbox "
                f"ADD CONSTRAINT ck_trading_event_outbox_event_type CHECK ({event_type_constraint_sql})",
            ),
        )


async def _ensure_broker_account_schema() -> None:
    """Normalize broker_accounts columns/constraints for multi-broker runtime."""
    assert engine is not None
    provider_constraint_sql = "provider IN ('alpaca', 'ccxt', 'sandbox')"
    async with engine.begin() as connection:
        await connection.execute(
            text(
                "ALTER TABLE broker_accounts ADD COLUMN IF NOT EXISTS exchange_id VARCHAR(64)"
            ),
        )
        await connection.execute(
            text(
                "ALTER TABLE broker_accounts ADD COLUMN IF NOT EXISTS account_uid VARCHAR(128)"
            ),
        )
        await connection.execute(
            text(
                "ALTER TABLE broker_accounts ADD COLUMN IF NOT EXISTS is_default BOOLEAN"
            ),
        )
        await connection.execute(
            text(
                "ALTER TABLE broker_accounts ADD COLUMN IF NOT EXISTS is_sandbox BOOLEAN"
            ),
        )
        await connection.execute(
            text(
                "ALTER TABLE broker_accounts "
                "ADD COLUMN IF NOT EXISTS last_validation_error_code VARCHAR(64)",
            ),
        )
        await connection.execute(
            text(
                "ALTER TABLE broker_accounts "
                "ADD COLUMN IF NOT EXISTS capabilities JSONB DEFAULT '{}'::jsonb",
            ),
        )

        # Backfill normalized defaults.
        await connection.execute(
            text("UPDATE broker_accounts SET exchange_id = COALESCE(exchange_id, '')"),
        )
        await connection.execute(
            text("UPDATE broker_accounts SET account_uid = COALESCE(account_uid, '')"),
        )
        await connection.execute(
            text("UPDATE broker_accounts SET is_default = COALESCE(is_default, false)"),
        )
        await connection.execute(
            text("UPDATE broker_accounts SET is_sandbox = COALESCE(is_sandbox, false)"),
        )
        await connection.execute(
            text(
                "UPDATE broker_accounts SET capabilities = COALESCE(capabilities, '{}'::jsonb)"
            ),
        )

        # Provider-specific structured field backfills.
        await connection.execute(
            text(
                "UPDATE broker_accounts "
                "SET exchange_id = 'alpaca' "
                "WHERE provider = 'alpaca' AND exchange_id = ''",
            ),
        )
        await connection.execute(
            text(
                "UPDATE broker_accounts "
                "SET exchange_id = 'sandbox' "
                "WHERE provider = 'sandbox' AND exchange_id = ''",
            ),
        )
        await connection.execute(
            text(
                "UPDATE broker_accounts "
                "SET exchange_id = lower(COALESCE(NULLIF(metadata->>'exchange_id',''), NULLIF(validation_metadata->>'exchange_id',''), exchange_id, '')) "
                "WHERE provider = 'ccxt' "
                "AND exchange_id = ''",
            ),
        )
        await connection.execute(
            text(
                "UPDATE broker_accounts "
                "SET account_uid = COALESCE(NULLIF(validation_metadata->>'paper_account_id',''), NULLIF(key_fingerprint,''), id::text) "
                "WHERE provider = 'alpaca' "
                "AND account_uid = ''",
            ),
        )
        await connection.execute(
            text(
                "UPDATE broker_accounts "
                "SET account_uid = COALESCE(NULLIF(metadata->>'account_uid',''), NULLIF(key_fingerprint,''), id::text) "
                "WHERE provider = 'ccxt' "
                "AND account_uid = ''",
            ),
        )
        await connection.execute(
            text(
                "UPDATE broker_accounts "
                "SET account_uid = COALESCE(NULLIF(metadata->>'account_uid',''), id::text) "
                "WHERE provider = 'sandbox' "
                "AND account_uid = ''",
            ),
        )
        await connection.execute(
            text(
                "UPDATE broker_accounts "
                "SET is_sandbox = true "
                "WHERE provider = 'sandbox'",
            ),
        )
        await connection.execute(
            text(
                "UPDATE broker_accounts "
                "SET is_sandbox = true "
                "WHERE provider = 'ccxt' "
                "AND lower(COALESCE(exchange_id, '')) = 'okx' "
                "AND COALESCE(metadata->>'sandbox', validation_metadata->>'sandbox', 'true') IN ('1', 'true', 'yes', 'on')",
            ),
        )

        # Keep only one active account per normalized identity tuple.
        await connection.execute(
            text(
                "WITH ranked AS ( "
                "    SELECT id, "
                "           row_number() OVER ( "
                "               PARTITION BY user_id, provider, exchange_id, account_uid "
                "               ORDER BY last_validated_at DESC NULLS LAST, created_at DESC, id DESC "
                "           ) AS rn "
                "    FROM broker_accounts "
                "    WHERE status = 'active' "
                ") "
                "UPDATE broker_accounts AS b "
                "SET status = 'inactive', "
                "    is_default = false, "
                "    updated_source = 'system', "
                "    last_error = COALESCE(NULLIF(b.last_error, ''), 'auto_inactivated_duplicate_identity') "
                "FROM ranked "
                "WHERE b.id = ranked.id AND ranked.rn > 1",
            ),
        )

        await connection.execute(
            text(
                "ALTER TABLE broker_accounts ALTER COLUMN exchange_id SET DEFAULT ''",
            ),
        )
        await connection.execute(
            text(
                "ALTER TABLE broker_accounts ALTER COLUMN account_uid SET DEFAULT ''",
            ),
        )
        await connection.execute(
            text(
                "ALTER TABLE broker_accounts ALTER COLUMN is_default SET DEFAULT false",
            ),
        )
        await connection.execute(
            text(
                "ALTER TABLE broker_accounts ALTER COLUMN is_sandbox SET DEFAULT false",
            ),
        )
        await connection.execute(
            text(
                "ALTER TABLE broker_accounts ALTER COLUMN capabilities SET DEFAULT '{}'::jsonb",
            ),
        )
        await connection.execute(
            text("ALTER TABLE broker_accounts ALTER COLUMN exchange_id SET NOT NULL"),
        )
        await connection.execute(
            text("ALTER TABLE broker_accounts ALTER COLUMN account_uid SET NOT NULL"),
        )
        await connection.execute(
            text("ALTER TABLE broker_accounts ALTER COLUMN is_default SET NOT NULL"),
        )
        await connection.execute(
            text("ALTER TABLE broker_accounts ALTER COLUMN is_sandbox SET NOT NULL"),
        )
        await connection.execute(
            text("ALTER TABLE broker_accounts ALTER COLUMN capabilities SET NOT NULL"),
        )

        # Keep a single active default per user/mode.
        await connection.execute(
            text(
                "WITH ranked AS ( "
                "    SELECT id, "
                "           row_number() OVER ( "
                "               PARTITION BY user_id, mode "
                "               ORDER BY is_default DESC, last_validated_at DESC NULLS LAST, created_at DESC, id DESC "
                "           ) AS rn "
                "    FROM broker_accounts "
                "    WHERE status = 'active' "
                ") "
                "UPDATE broker_accounts AS b "
                "SET is_default = (ranked.rn = 1) "
                "FROM ranked "
                "WHERE b.id = ranked.id",
            ),
        )
        await connection.execute(
            text(
                "UPDATE broker_accounts SET is_default = false WHERE status <> 'active'"
            ),
        )

        # Sync provider check-constraint with model.
        await connection.execute(
            text(
                "ALTER TABLE broker_accounts DROP CONSTRAINT IF EXISTS ck_broker_accounts_provider"
            ),
        )
        await connection.execute(
            text(
                "ALTER TABLE broker_accounts "
                f"ADD CONSTRAINT ck_broker_accounts_provider CHECK ({provider_constraint_sql})",
            ),
        )

        # Multi-broker identity and default constraints.
        await connection.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS "
                "uq_broker_accounts_user_provider_exchange_account_uid_active "
                "ON broker_accounts (user_id, provider, exchange_id, account_uid) "
                "WHERE status = 'active'",
            ),
        )
        await connection.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS "
                "uq_broker_accounts_user_mode_default_active "
                "ON broker_accounts (user_id, mode) "
                "WHERE is_default = true AND status = 'active'",
            ),
        )


async def _ensure_broker_account_audit_log_constraint() -> None:
    """Ensure audit action constraint covers default-selection events."""
    assert engine is not None
    action_constraint_sql = (
        "action IN ('create', 'update', 'validate', 'deactivate', 'set_default')"
    )
    async with engine.begin() as connection:
        await connection.execute(
            text(
                "ALTER TABLE broker_account_audit_logs "
                "DROP CONSTRAINT IF EXISTS ck_broker_account_audit_logs_action",
            ),
        )
        await connection.execute(
            text(
                "ALTER TABLE broker_account_audit_logs "
                f"ADD CONSTRAINT ck_broker_account_audit_logs_action CHECK ({action_constraint_sql})",
            ),
        )


async def _ensure_user_billing_columns() -> None:
    """Ensure users table has tier column and matching check constraint."""
    assert engine is not None
    tier_constraint_sql = "current_tier IN ('free', 'go', 'plus', 'pro')"
    async with engine.begin() as connection:
        await connection.execute(
            text("ALTER TABLE users ADD COLUMN IF NOT EXISTS current_tier VARCHAR(20)"),
        )
        await connection.execute(
            text(
                "UPDATE users SET current_tier = COALESCE(NULLIF(current_tier, ''), 'free')"
            ),
        )
        await connection.execute(
            text("ALTER TABLE users ALTER COLUMN current_tier SET DEFAULT 'free'"),
        )
        await connection.execute(
            text("ALTER TABLE users ALTER COLUMN current_tier SET NOT NULL"),
        )
        await connection.execute(
            text("ALTER TABLE users DROP CONSTRAINT IF EXISTS ck_users_current_tier"),
        )
        await connection.execute(
            text(
                "ALTER TABLE users "
                f"ADD CONSTRAINT ck_users_current_tier CHECK ({tier_constraint_sql})",
            ),
        )


async def _ensure_user_settings_schema() -> None:
    """Ensure user_settings has onboarding status JSON payload."""
    assert engine is not None
    async with engine.begin() as connection:
        await connection.execute(
            text(
                "ALTER TABLE user_settings "
                "ADD COLUMN IF NOT EXISTS onboarding_status JSONB",
            ),
        )
        await connection.execute(
            text(
                "UPDATE user_settings "
                "SET onboarding_status = COALESCE(onboarding_status, '{}'::jsonb)",
            ),
        )
        await connection.execute(
            text(
                "ALTER TABLE user_settings "
                "ALTER COLUMN onboarding_status SET DEFAULT '{}'::jsonb",
            ),
        )
        await connection.execute(
            text(
                "ALTER TABLE user_settings ALTER COLUMN onboarding_status SET NOT NULL",
            ),
        )


async def _ensure_billing_schema() -> None:
    """Ensure billing tables include replay-safe webhook and usage idempotency constraints."""
    assert engine is not None
    async with engine.begin() as connection:
        await connection.execute(
            text(
                "ALTER TABLE billing_subscriptions "
                "DROP CONSTRAINT IF EXISTS ck_billing_subscriptions_tier",
            ),
        )
        await connection.execute(
            text(
                "ALTER TABLE billing_subscriptions "
                "ADD CONSTRAINT ck_billing_subscriptions_tier "
                "CHECK (tier IN ('free', 'go', 'plus', 'pro'))",
            ),
        )
        await connection.execute(
            text(
                "ALTER TABLE billing_subscriptions "
                "DROP CONSTRAINT IF EXISTS ck_billing_subscriptions_pending_tier",
            ),
        )
        await connection.execute(
            text(
                "ALTER TABLE billing_subscriptions "
                "ADD CONSTRAINT ck_billing_subscriptions_pending_tier "
                "CHECK (pending_tier IS NULL OR pending_tier IN ('free', 'go', 'plus', 'pro'))",
            ),
        )

        await connection.execute(
            text(
                "ALTER TABLE billing_webhook_events "
                "ADD COLUMN IF NOT EXISTS failed_at TIMESTAMPTZ",
            ),
        )

        # Keep one usage event per idempotent reference before enforcing uniqueness.
        await connection.execute(
            text(
                "DELETE FROM billing_usage_events e "
                "USING ("
                "  SELECT id FROM ("
                "    SELECT id, "
                "           ROW_NUMBER() OVER ("
                "             PARTITION BY user_id, metric_code, reference_type, reference_id "
                "             ORDER BY occurred_at ASC, id ASC"
                "           ) AS rn "
                "    FROM billing_usage_events "
                "    WHERE reference_type IS NOT NULL "
                "      AND reference_id IS NOT NULL"
                "  ) ranked "
                "  WHERE ranked.rn > 1"
                ") dupes "
                "WHERE e.id = dupes.id",
            ),
        )
        await connection.execute(
            text(
                "DO $$ "
                "BEGIN "
                "  IF NOT EXISTS ("
                "    SELECT 1 FROM pg_constraint "
                "    WHERE conname = 'uq_billing_usage_events_user_metric_reference'"
                "  ) THEN "
                "    ALTER TABLE billing_usage_events "
                "    ADD CONSTRAINT uq_billing_usage_events_user_metric_reference "
                "    UNIQUE (user_id, metric_code, reference_type, reference_id); "
                "  END IF; "
                "END $$;",
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
        await _ensure_broker_account_schema()
        await _ensure_broker_account_audit_log_constraint()
        await _ensure_user_billing_columns()
        await _ensure_user_settings_schema()
        await _ensure_billing_schema()
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
    await _ensure_broker_account_schema()
    await _ensure_broker_account_audit_log_constraint()
    await _ensure_user_billing_columns()
    await _ensure_billing_schema()
    await _ensure_sessions_phase_constraint()
    await _ensure_sessions_archival_columns()
    await _ensure_trading_runtime_columns()
    await _ensure_trading_event_outbox_constraint()


async def get_db_session() -> AsyncIterator[AsyncSession]:
    """Yield async session from shared factory."""
    if AsyncSessionLocal is None:
        # API startup is the schema owner; request-path lazy init only opens pool.
        await init_postgres(ensure_schema=False)
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
