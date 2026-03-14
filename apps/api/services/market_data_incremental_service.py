"""API-facing service helpers for market-data incremental sync."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from packages.infra.db.models.market_data_catalog import MarketDataCatalog
from packages.infra.db.models.market_data_incremental_import_job import (
    MarketDataIncrementalImportJob,
)


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


async def build_incremental_inventory(db: AsyncSession) -> dict[str, object]:
    """Build inventory grouped by market and symbol from catalog rows."""

    rows = (
        await db.scalars(
            select(MarketDataCatalog).order_by(
                MarketDataCatalog.market.asc(),
                MarketDataCatalog.symbol.asc(),
                MarketDataCatalog.session.asc(),
                MarketDataCatalog.timeframe.asc(),
                MarketDataCatalog.year.asc(),
            )
        )
    ).all()
    grouped: dict[tuple[str, str, str], dict[str, object]] = {}

    for row in rows:
        key = (row.market, row.symbol, row.session)
        item = grouped.get(key)
        if item is None:
            item = {
                "market": row.market,
                "symbol": row.symbol,
                "session": row.session,
                "timeframe_coverage": {},
                "available_raw_timeframes": set(),
                "coverage_start": _ensure_utc(row.start_date),
                "coverage_end": _ensure_utc(row.end_date),
            }
            grouped[key] = item

        tf_coverage = item["timeframe_coverage"]
        assert isinstance(tf_coverage, dict)
        current = tf_coverage.get(row.timeframe)
        row_start = _ensure_utc(row.start_date)
        row_end = _ensure_utc(row.end_date)
        if not isinstance(current, dict):
            tf_coverage[row.timeframe] = {"start": row_start, "end": row_end}
        else:
            tf_coverage[row.timeframe] = {
                "start": min(_ensure_utc(current["start"]), row_start),
                "end": max(_ensure_utc(current["end"]), row_end),
            }

        available = item["available_raw_timeframes"]
        assert isinstance(available, set)
        available.add(row.timeframe)
        item["coverage_start"] = min(_ensure_utc(item["coverage_start"]), row_start)
        item["coverage_end"] = max(_ensure_utc(item["coverage_end"]), row_end)

    markets_map: dict[str, list[dict[str, object]]] = {}
    for (market, symbol, session), item in grouped.items():
        coverage_start = _ensure_utc(item["coverage_start"])
        coverage_end = _ensure_utc(item["coverage_end"])
        duration_days = max(0.0, (coverage_end - coverage_start).total_seconds() / 86_400.0)
        tf_coverage_raw = item["timeframe_coverage"]
        assert isinstance(tf_coverage_raw, dict)
        tf_coverage_out: dict[str, dict[str, str]] = {}
        for timeframe, bounds in tf_coverage_raw.items():
            if not isinstance(bounds, dict):
                continue
            start = bounds.get("start")
            end = bounds.get("end")
            if not isinstance(start, datetime) or not isinstance(end, datetime):
                continue
            tf_coverage_out[timeframe] = {
                "start": _ensure_utc(start).isoformat(),
                "end": _ensure_utc(end).isoformat(),
            }
        raw_timeframes = sorted(
            tf
            for tf in item["available_raw_timeframes"]
            if isinstance(tf, str) and tf.strip()
        )
        symbol_payload = {
            "symbol": symbol,
            "session": session,
            "coverage": {
                "start": coverage_start.isoformat(),
                "end": coverage_end.isoformat(),
                "duration_days": round(duration_days, 6),
            },
            "available_raw_timeframes": raw_timeframes,
            "timeframe_coverage": tf_coverage_out,
        }
        markets_map.setdefault(market, []).append(symbol_payload)

    markets_payload = [
        {
            "market": market,
            "symbols": sorted(items, key=lambda row: (str(row["symbol"]), str(row["session"]))),
        }
        for market, items in sorted(markets_map.items(), key=lambda row: row[0])
    ]
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "markets": markets_payload,
    }


async def create_or_get_incremental_import_job(
    db: AsyncSession,
    *,
    run_id: str,
    bucket: str,
    prefix: str,
    manifest_object: str,
    file_count: int,
) -> tuple[MarketDataIncrementalImportJob, bool]:
    existing = await db.scalar(
        select(MarketDataIncrementalImportJob).where(
            MarketDataIncrementalImportJob.run_id == run_id
        )
    )
    if existing is not None:
        return existing, True

    job = MarketDataIncrementalImportJob(
        run_id=run_id,
        bucket=bucket,
        prefix=prefix,
        manifest_object=manifest_object,
        status="queued",
        file_count=max(int(file_count), 0),
        processed_files=0,
        rows_written=0,
    )
    db.add(job)
    await db.flush()
    return job, False


async def get_incremental_import_job(
    db: AsyncSession,
    *,
    job_id: UUID,
) -> MarketDataIncrementalImportJob | None:
    return await db.scalar(
        select(MarketDataIncrementalImportJob).where(
            MarketDataIncrementalImportJob.id == job_id
        )
    )
