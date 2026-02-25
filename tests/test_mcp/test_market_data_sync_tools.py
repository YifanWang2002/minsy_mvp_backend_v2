from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from mcp.server.fastmcp import FastMCP
from sqlalchemy.ext.asyncio import AsyncSession

from src.engine.data import DataLoader
from src.engine.execution.adapters.base import OhlcvBar
from src.engine.market_data import sync_service
from src.mcp.market_data import tools as market_tools


class _SessionContext:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def __aenter__(self) -> AsyncSession:
        return self._session

    async def __aexit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001
        return False


def _extract_payload(call_result: object) -> dict[str, Any]:
    if isinstance(call_result, tuple) and len(call_result) == 2:
        maybe_result = call_result[1]
        if isinstance(maybe_result, dict):
            raw = maybe_result.get("result")
            if isinstance(raw, str):
                return json.loads(raw)
    raise AssertionError(f"Unexpected call result: {call_result!r}")


def _make_local_loader(root: Path) -> DataLoader:
    return DataLoader(data_dir=root)


@pytest.mark.asyncio
async def test_market_data_sync_tools_detect_fetch_and_get(
    db_session: AsyncSession,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = _make_local_loader(tmp_path)
    (tmp_path / "crypto").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(market_tools, "_get_data_loader", lambda: loader)

    async def _fake_new_db_session() -> _SessionContext:
        return _SessionContext(db_session)

    monkeypatch.setattr(market_tools, "_new_db_session", _fake_new_db_session)

    class _PatchedDataLoader(DataLoader):
        def __init__(self) -> None:
            super().__init__(data_dir=tmp_path)

    monkeypatch.setattr(sync_service, "DataLoader", _PatchedDataLoader)

    class _FakeProvider:
        async def fetch_ohlcv(self, **_: object):
            return [
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
            ]

    async def _fake_build_provider_client(_provider_name: str):
        provider = _FakeProvider()

        async def _close() -> None:
            return None

        return provider, _close

    monkeypatch.setattr(sync_service, "_build_provider_client", _fake_build_provider_client)

    mcp = FastMCP("test-market-data-sync-tools")
    market_tools.register_market_data_tools(mcp)

    detected = _extract_payload(
        await mcp.call_tool(
            "market_data_detect_missing_ranges",
            {
                "market": "crypto",
                "symbol": "BTCUSD",
                "timeframe": "1m",
                "start_date": "2024-01-01T00:00:00Z",
                "end_date": "2024-01-01T00:01:00Z",
            },
        )
    )
    assert detected["ok"] is True
    assert detected["missing_bars"] == 2
    assert len(detected["missing_ranges"]) == 1

    fetched = _extract_payload(
        await mcp.call_tool(
            "market_data_fetch_missing_ranges",
            {
                "provider": "alpaca",
                "market": "crypto",
                "symbol": "BTCUSD",
                "timeframe": "1m",
                "start_date": "2024-01-01T00:00:00Z",
                "end_date": "2024-01-01T00:01:00Z",
                "run_async": False,
            },
        )
    )
    assert fetched["ok"] is True
    assert fetched["status"] == "done"
    assert fetched["rows_written"] >= 2
    sync_job_id = fetched["sync_job_id"]

    fetched_again = _extract_payload(
        await mcp.call_tool(
            "market_data_get_sync_job",
            {"sync_job_id": sync_job_id},
        )
    )
    assert fetched_again["ok"] is True
    assert fetched_again["sync_job_id"] == sync_job_id

    out_file = tmp_path / "crypto" / "BTCUSD_1min_eth_2024.parquet"
    assert out_file.exists()
    frame = pd.read_parquet(out_file)
    assert not frame.empty
