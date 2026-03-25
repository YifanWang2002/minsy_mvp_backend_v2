from __future__ import annotations

import asyncio
import importlib
import os
import sys

import asyncpg


BACKEND_DIR = "/Users/yifanwang/minsy_mvp_remastered/backend"
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)


def _configure_local_env() -> None:
    os.environ["MINSY_ENV_FILES"] = ",".join(
        [
            "env/.env.secrets",
            "env/.env.common",
            "env/.env.dev",
            "env/.env.dev.api",
            "env/.env.dev.localtest",
        ]
    )
    os.environ["MINSY_SERVICE"] = "api"


async def _ensure_auth_schema() -> None:
    _configure_local_env()
    from packages.shared_settings.loader import service_loader
    from packages.shared_settings.schema import settings as settings_module

    settings_module.get_settings.cache_clear()
    service_loader._load_legacy_settings.cache_clear()
    service_loader.get_common_settings.cache_clear()
    service_loader.get_api_settings.cache_clear()
    service_loader.get_mcp_settings.cache_clear()
    service_loader.get_worker_cpu_settings.cache_clear()
    service_loader.get_worker_io_settings.cache_clear()
    service_loader.get_beat_settings.cache_clear()
    settings_module.settings = settings_module.get_settings(service="api")

    for module_name in ("packages.infra.db.session",):
        module = sys.modules.get(module_name)
        if module is not None:
            importlib.reload(module)

    from packages.infra.db.session import close_postgres, init_postgres

    try:
        await init_postgres()
    finally:
        await close_postgres()


async def _fetch_users_auth_columns() -> dict[str, dict[str, object]]:
    connection = await asyncpg.connect(
        host="127.0.0.1",
        port=5432,
        user="postgres",
        password="123456",
        database="minsy_pgsql",
    )
    try:
        rows = await connection.fetch(
            """
            SELECT column_name, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = 'users'
              AND column_name IN ('password_hash', 'clerk_user_id', 'auth_provider')
            ORDER BY column_name
            """
        )
        return {
            str(row["column_name"]): {
                "is_nullable": row["is_nullable"],
                "column_default": row["column_default"],
            }
            for row in rows
        }
    finally:
        await connection.close()


async def _fetch_users_auth_constraints() -> set[str]:
    connection = await asyncpg.connect(
        host="127.0.0.1",
        port=5432,
        user="postgres",
        password="123456",
        database="minsy_pgsql",
    )
    try:
        rows = await connection.fetch(
            """
            SELECT conname
            FROM pg_constraint
            WHERE conname IN ('ck_users_auth_provider')
            UNION
            SELECT indexname AS conname
            FROM pg_indexes
            WHERE schemaname = 'public'
              AND tablename = 'users'
              AND indexname = 'uq_users_clerk_user_id'
            """
        )
        return {str(row["conname"]) for row in rows}
    finally:
        await connection.close()


def test_000_postgres_users_auth_columns_and_constraints_exist() -> None:
    asyncio.run(_ensure_auth_schema())
    columns = asyncio.run(_fetch_users_auth_columns())
    constraints = asyncio.run(_fetch_users_auth_constraints())

    assert columns["password_hash"]["is_nullable"] == "YES"
    assert columns["clerk_user_id"]["is_nullable"] == "YES"
    assert columns["auth_provider"]["is_nullable"] == "NO"
    assert "legacy_password" in str(columns["auth_provider"]["column_default"])
    assert constraints == {"ck_users_auth_provider", "uq_users_clerk_user_id"}
