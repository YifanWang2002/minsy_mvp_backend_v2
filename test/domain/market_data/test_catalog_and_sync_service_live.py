from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from uuid import uuid4

import pandas as pd
from fastapi.testclient import TestClient
from sqlalchemy import select

from packages.infra.providers.trading.adapters.base import OhlcvBar


def _missing_ranges_for_window(start: datetime, end: datetime) -> list[dict[str, str]]:
    return [
        {
            "start": start.isoformat(),
            "end": end.isoformat(),
        }
    ]


def test_000_accessibility_catalog_scan_and_query(
    api_test_client: TestClient,
    tmp_path,
) -> None:
    _ = api_test_client
    symbol = f"CAT{uuid4().hex[:8]}".upper()
    market_dir = tmp_path / "us_stocks"
    market_dir.mkdir(parents=True, exist_ok=True)
    shard = market_dir / f"{symbol}_1min_rth_2026.parquet"
    frame = pd.DataFrame(
        {
            "timestamp": [
                datetime(2026, 1, 1, 14, 30, tzinfo=UTC),
                datetime(2026, 1, 1, 14, 31, tzinfo=UTC),
                datetime(2026, 1, 1, 14, 32, tzinfo=UTC),
            ],
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [10.0, 11.0, 12.0],
        }
    )
    frame.to_parquet(shard, index=False)

    async def _exercise() -> None:
        from packages.domain.market_data.catalog_service import (
            get_symbol_coverage,
            mark_accessed,
            scan_and_sync_catalog,
        )
        from packages.infra.db import session as db_session_module
        from packages.infra.db.models.market_data_catalog import MarketDataCatalog

        await db_session_module.init_postgres(ensure_schema=True)
        assert db_session_module.AsyncSessionLocal is not None
        async with db_session_module.AsyncSessionLocal() as db:
            synced = await scan_and_sync_catalog(db, tmp_path)
            await db.commit()
            assert synced >= 1

            coverage = await get_symbol_coverage(
                db,
                market="stock",
                symbol=symbol,
                timeframe="1m",
            )
            assert len(coverage) == 1
            assert coverage[0].row_count == 3
            assert coverage[0].year == 2026
            assert coverage[0].file_size_bytes > 0

            await mark_accessed(
                db,
                market="us_stocks",
                symbol=symbol,
                timeframe="1m",
                year=2026,
            )
            await db.commit()

            row = await db.scalar(
                select(MarketDataCatalog).where(
                    MarketDataCatalog.market == "us_stocks",
                    MarketDataCatalog.symbol == symbol,
                    MarketDataCatalog.timeframe == "1m",
                    MarketDataCatalog.year == 2026,
                )
            )
            assert row is not None
            assert row.last_accessed_at is not None

    assert api_test_client.portal is not None
    api_test_client.portal.call(_exercise)


def test_010_sync_job_create_deduplicates_running_or_queued(
    api_test_client: TestClient,
) -> None:
    _ = api_test_client
    symbol = f"DUP{uuid4().hex[:10]}".upper()
    requested_start = datetime.now(UTC) - timedelta(hours=1)
    requested_end = datetime.now(UTC)

    async def _exercise() -> None:
        from packages.domain.market_data.sync_service import create_market_data_sync_job
        from packages.infra.db import session as db_session_module

        await db_session_module.init_postgres(ensure_schema=True)
        assert db_session_module.AsyncSessionLocal is not None
        async with db_session_module.AsyncSessionLocal() as db:
            first = await create_market_data_sync_job(
                db,
                provider="alpaca",
                market="stock",
                symbol=symbol,
                timeframe="1m",
                requested_start=requested_start,
                requested_end=requested_end,
                missing_ranges=_missing_ranges_for_window(requested_start, requested_end),
                auto_commit=True,
            )
            second = await create_market_data_sync_job(
                db,
                provider="alpaca",
                market="stock",
                symbol=symbol,
                timeframe="1m",
                requested_start=requested_start,
                requested_end=requested_end,
                missing_ranges=_missing_ranges_for_window(requested_start, requested_end),
                auto_commit=True,
            )
            assert first.job_id == second.job_id

    assert api_test_client.portal is not None
    api_test_client.portal.call(_exercise)


def test_020_execute_sync_job_cancelled_when_lock_not_acquired(
    api_test_client: TestClient,
    monkeypatch,
) -> None:
    _ = api_test_client
    symbol = f"LOCK{uuid4().hex[:8]}".upper()
    requested_start = datetime.now(UTC) - timedelta(hours=2)
    requested_end = datetime.now(UTC) - timedelta(hours=1)

    class _FakeRedisLockMiss:
        async def set(self, *_args: Any, **_kwargs: Any) -> bool:
            return False

        async def eval(self, *_args: Any, **_kwargs: Any) -> int:
            return 0

        async def delete(self, *_args: Any, **_kwargs: Any) -> int:
            return 0

    import packages.domain.market_data.sync_service as sync_service

    async def _fake_init_redis() -> None:
        return None

    fake_redis = _FakeRedisLockMiss()
    monkeypatch.setattr(sync_service, "init_redis", _fake_init_redis)
    monkeypatch.setattr(sync_service, "get_redis_client", lambda: fake_redis)

    async def _exercise() -> None:
        from packages.infra.db import session as db_session_module
        from packages.infra.db.models.market_data_sync_job import MarketDataSyncJob

        await db_session_module.init_postgres(ensure_schema=True)
        assert db_session_module.AsyncSessionLocal is not None
        async with db_session_module.AsyncSessionLocal() as db:
            receipt = await sync_service.create_market_data_sync_job(
                db,
                provider="alpaca",
                market="stock",
                symbol=symbol,
                timeframe="1m",
                requested_start=requested_start,
                requested_end=requested_end,
                missing_ranges=_missing_ranges_for_window(requested_start, requested_end),
                auto_commit=True,
            )
            view = await sync_service.execute_market_data_sync_job(
                db,
                job_id=receipt.job_id,
                auto_commit=True,
            )
            assert view.status == "failed"
            assert view.current_step == "cancelled_duplicate"
            assert view.errors
            assert "Another worker is syncing" in view.errors[0]["message"]

            row = await db.scalar(
                select(MarketDataSyncJob).where(MarketDataSyncJob.id == receipt.job_id)
            )
            assert row is not None
            assert row.status == "cancelled"

    assert api_test_client.portal is not None
    api_test_client.portal.call(_exercise)


def test_030_execute_sync_job_updates_catalog_after_write(
    api_test_client: TestClient,
    monkeypatch,
    tmp_path,
) -> None:
    _ = api_test_client
    symbol = f"CATSYNC{uuid4().hex[:6]}".upper()
    requested_start = datetime(2026, 1, 10, 12, 0, tzinfo=UTC)
    requested_end = datetime(2026, 1, 10, 12, 1, tzinfo=UTC)

    class _FakeRedisLockHit:
        async def set(self, *_args: Any, **_kwargs: Any) -> bool:
            return True

        async def eval(self, *_args: Any, **_kwargs: Any) -> int:
            return 1

        async def delete(self, *_args: Any, **_kwargs: Any) -> int:
            return 1

    class _DummyProvider:
        async def aclose(self) -> None:
            return None

    import packages.domain.market_data.sync_service as sync_service
    from packages.domain.market_data.data import DataLoader as RealDataLoader

    class _PatchedDataLoader(RealDataLoader):
        def __init__(self) -> None:
            super().__init__(data_dir=tmp_path)

    async def _fake_init_redis() -> None:
        return None

    async def _fake_build_provider_client(_provider_name: str):
        provider = _DummyProvider()

        async def _close() -> None:
            await provider.aclose()

        return provider, _close

    async def _fake_fetch_provider_range(
        *,
        provider_name: str,
        provider_client: Any,
        market: str,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> list[OhlcvBar]:
        _ = (provider_name, provider_client, market, symbol, timeframe, start, end)
        return [
            OhlcvBar(
                timestamp=datetime(2026, 1, 10, 12, 0, tzinfo=UTC),
                open=Decimal("200"),
                high=Decimal("201"),
                low=Decimal("199"),
                close=Decimal("200.5"),
                volume=Decimal("8"),
            )
        ]

    monkeypatch.setattr(sync_service, "init_redis", _fake_init_redis)
    monkeypatch.setattr(sync_service, "get_redis_client", lambda: _FakeRedisLockHit())
    monkeypatch.setattr(sync_service, "_build_provider_client", _fake_build_provider_client)
    monkeypatch.setattr(sync_service, "_fetch_provider_range", _fake_fetch_provider_range)
    monkeypatch.setattr(sync_service, "DataLoader", _PatchedDataLoader)

    async def _exercise() -> None:
        from packages.domain.market_data.catalog_service import get_symbol_coverage
        from packages.infra.db import session as db_session_module

        await db_session_module.init_postgres(ensure_schema=True)
        assert db_session_module.AsyncSessionLocal is not None
        async with db_session_module.AsyncSessionLocal() as db:
            receipt = await sync_service.create_market_data_sync_job(
                db,
                provider="alpaca",
                market="stock",
                symbol=symbol,
                timeframe="1m",
                requested_start=requested_start,
                requested_end=requested_end,
                missing_ranges=_missing_ranges_for_window(requested_start, requested_end),
                auto_commit=True,
            )
            view = await sync_service.execute_market_data_sync_job(
                db,
                job_id=receipt.job_id,
                auto_commit=True,
            )
            assert view.status == "done"
            assert view.rows_written >= 1

            coverage = await get_symbol_coverage(
                db,
                market="stock",
                symbol=symbol,
                timeframe="1m",
            )
            assert len(coverage) == 1
            assert coverage[0].year == 2026
            assert coverage[0].row_count >= 1
            assert coverage[0].file_size_bytes > 0

    assert api_test_client.portal is not None
    api_test_client.portal.call(_exercise)
