from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from src.engine.data import DataLoader
from src.engine.execution.adapters.base import OhlcvBar
from src.engine.market_data import sync_service
from src.engine.market_data.sync_service import (
    create_market_data_sync_job,
    execute_market_data_sync_job,
)


class _FixedDataLoader(DataLoader):
    def __init__(self, root: Path) -> None:
        super().__init__(data_dir=root)


@pytest.mark.asyncio
async def test_market_data_sync_job_execute_writes_local_parquet(
    db_session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _PatchedDataLoader(_FixedDataLoader):
        def __init__(self) -> None:
            super().__init__(tmp_path)

    monkeypatch.setattr(sync_service, "DataLoader", _PatchedDataLoader)

    bars = [
        OhlcvBar(
            timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
            open=Decimal("100"),
            high=Decimal("101"),
            low=Decimal("99"),
            close=Decimal("100.5"),
            volume=Decimal("10"),
        ),
        OhlcvBar(
            timestamp=datetime(2024, 1, 1, 0, 1, tzinfo=UTC),
            open=Decimal("101"),
            high=Decimal("102"),
            low=Decimal("100"),
            close=Decimal("101.5"),
            volume=Decimal("11"),
        ),
        OhlcvBar(
            timestamp=datetime(2024, 1, 1, 0, 2, tzinfo=UTC),
            open=Decimal("102"),
            high=Decimal("103"),
            low=Decimal("101"),
            close=Decimal("102.5"),
            volume=Decimal("12"),
        ),
    ]

    class _FakeProvider:
        def __init__(self) -> None:
            self.calls = 0

        async def fetch_ohlcv(self, **_: object):
            self.calls += 1
            if self.calls > 1:
                return []
            return bars

    async def _fake_build_provider_client(_provider_name: str):
        provider = _FakeProvider()

        async def _close() -> None:
            return None

        return provider, _close

    monkeypatch.setattr(sync_service, "_build_provider_client", _fake_build_provider_client)

    receipt = await create_market_data_sync_job(
        db_session,
        provider="alpaca",
        market="crypto",
        symbol="BTCUSD",
        timeframe="1m",
        requested_start=datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
        requested_end=datetime(2024, 1, 1, 0, 2, tzinfo=UTC),
        missing_ranges=[
            {
                "start": "2024-01-01T00:00:00+00:00",
                "end": "2024-01-01T00:02:00+00:00",
                "bars": 3,
            }
        ],
        auto_commit=True,
    )

    view = await execute_market_data_sync_job(
        db_session,
        job_id=receipt.job_id,
        auto_commit=True,
    )

    assert view.status == "done"
    assert view.rows_written == 3
    assert view.range_filled == 1
    assert view.total_ranges == 1
    assert not view.errors

    out_file = tmp_path / "crypto" / "BTCUSD_1min_eth_2024.parquet"
    assert out_file.exists()
    frame = pd.read_parquet(out_file)
    assert len(frame) == 3
