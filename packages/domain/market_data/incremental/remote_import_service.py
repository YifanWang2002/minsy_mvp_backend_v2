"""Remote VM incremental import service."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from io import BytesIO

import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from packages.domain.market_data.catalog_service import scan_and_sync_catalog
from packages.domain.market_data.data import DataLoader
from packages.domain.market_data.data.parquet_writer import append_ohlcv_rows
from packages.infra.storage.gcs_client import GcsClient


@dataclass(frozen=True, slots=True)
class IncrementalImportSummary:
    files_total: int
    files_processed: int
    rows_written: int
    symbols_touched: int


async def import_incremental_manifest(
    db: AsyncSession,
    *,
    bucket: str,
    manifest_object: str,
) -> IncrementalImportSummary:
    """Download one manifest from GCS and merge all incremental parquet shards."""

    gcs_client = GcsClient()
    loader = DataLoader()
    manifest_bytes = await asyncio.to_thread(
        gcs_client.download_bytes,
        bucket_name=bucket,
        object_name=manifest_object,
    )
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    files = manifest.get("files")
    if not isinstance(files, list):
        return IncrementalImportSummary(
            files_total=0,
            files_processed=0,
            rows_written=0,
            symbols_touched=0,
        )

    rows_written = 0
    processed = 0
    touched_symbols: set[tuple[str, str]] = set()

    for item in files:
        if not isinstance(item, dict):
            continue
        market = str(item.get("market") or "").strip().lower()
        symbol = str(item.get("symbol") or "").strip().upper()
        session = str(item.get("session") or "").strip().lower()
        timeframe = str(item.get("timeframe") or "").strip().lower()
        object_name = str(item.get("gcs_object") or "").strip()
        if not market or not symbol or not timeframe or not object_name:
            continue
        if timeframe not in {"1m", "5m"}:
            continue

        payload = await asyncio.to_thread(
            gcs_client.download_bytes,
            bucket_name=bucket,
            object_name=object_name,
        )
        frame = pd.read_parquet(BytesIO(payload))
        write_result = append_ohlcv_rows(
            loader=loader,
            market=market,
            symbol=symbol,
            timeframe=timeframe,
            session=session,
            rows=frame,
        )
        rows_written += int(write_result.rows_written)
        processed += 1
        touched_symbols.add((market, symbol, session))

    # Re-sync catalog once after all appends.
    await scan_and_sync_catalog(db, loader.data_dir)

    return IncrementalImportSummary(
        files_total=len(files),
        files_processed=processed,
        rows_written=rows_written,
        symbols_touched=len(touched_symbols),
    )
