"""Catalog CRUD for local market-data parquet shards."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from packages.domain.market_data.data import DataLoader
from packages.infra.db.models.market_data_catalog import MarketDataCatalog
from packages.infra.observability.logger import logger


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    market: str
    symbol: str
    timeframe: str
    session: str
    year: int
    start_date: datetime
    end_date: datetime
    row_count: int
    file_path: str
    file_size_bytes: int
    last_accessed_at: datetime | None


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _normalize_symbol(symbol: str) -> str:
    normalized = str(symbol).strip().upper()
    if not normalized:
        raise ValueError("symbol cannot be empty")
    return normalized


def _normalize_timeframe(timeframe: str) -> str:
    normalized = str(timeframe).strip().lower()
    if normalized not in DataLoader.TIMEFRAME_MINUTES:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    return normalized


def _normalize_session(session: str) -> str:
    normalized = str(session).strip().lower()
    if normalized not in {"rth", "eth"}:
        raise ValueError("session must be one of: rth, eth")
    return normalized


def _to_entry(model: MarketDataCatalog) -> CatalogEntry:
    return CatalogEntry(
        market=model.market,
        symbol=model.symbol,
        timeframe=model.timeframe,
        session=model.session,
        year=int(model.year),
        start_date=_ensure_utc(model.start_date),
        end_date=_ensure_utc(model.end_date),
        row_count=int(model.row_count),
        file_path=str(model.file_path),
        file_size_bytes=int(model.file_size_bytes),
        last_accessed_at=_ensure_utc(model.last_accessed_at) if model.last_accessed_at else None,
    )


def _parse_catalog_filename(
    *,
    file_path: Path,
    loader: DataLoader,
) -> tuple[str, str, str, int] | None:
    parts = file_path.stem.split("_")
    if len(parts) < 4:
        return None
    symbol = "_".join(parts[:-3]).upper()
    timeframe_file = parts[-3].strip().lower()
    session = parts[-2].strip().lower()
    try:
        year = int(parts[-1])
    except ValueError:
        return None
    timeframe = loader.FILE_TIMEFRAME_MAP_REVERSE.get(timeframe_file)
    if timeframe is None:
        return None
    return symbol, timeframe, session, year


def _read_parquet_stats(file_path: Path) -> tuple[datetime, datetime, int, int]:
    try:
        frame = pd.read_parquet(file_path, columns=["timestamp"])
    except Exception:  # noqa: BLE001
        frame = pd.read_parquet(file_path)

    if "timestamp" in frame.columns:
        raw_timestamps = frame["timestamp"]
        if pd.api.types.is_datetime64_any_dtype(raw_timestamps):
            index = pd.DatetimeIndex(raw_timestamps)
            if index.tz is None:
                index = index.tz_localize("UTC")
            else:
                index = index.tz_convert("UTC")
            timestamps = pd.Series(index, dtype="datetime64[ns, UTC]").dropna()
        else:
            timestamps = pd.to_datetime(
                raw_timestamps,
                utc=True,
                errors="coerce",
                format="mixed",
            ).dropna()
    elif isinstance(frame.index, pd.DatetimeIndex):
        index = frame.index
        if index.tz is None:
            index = index.tz_localize("UTC")
        else:
            index = index.tz_convert("UTC")
        timestamps = pd.Series(index, dtype="datetime64[ns, UTC]")
    else:
        raise ValueError(f"Timestamp column missing in file: {file_path}")

    if timestamps.empty:
        raise ValueError(f"No timestamp data available in file: {file_path}")

    start = timestamps.min().to_pydatetime()
    end = timestamps.max().to_pydatetime()
    row_count = int(len(timestamps))
    file_size = int(file_path.stat().st_size)
    return _ensure_utc(start), _ensure_utc(end), row_count, file_size


async def upsert_catalog_entry(
    db: AsyncSession,
    *,
    market: str,
    symbol: str,
    timeframe: str,
    session: str,
    year: int,
    start_date: datetime,
    end_date: datetime,
    row_count: int,
    file_path: str,
    file_size_bytes: int,
) -> CatalogEntry:
    loader = DataLoader()
    market_key = loader.normalize_market(market)
    symbol_key = _normalize_symbol(symbol)
    timeframe_key = _normalize_timeframe(timeframe)
    session_key = _normalize_session(session)
    year_key = int(year)
    if year_key < 1970:
        raise ValueError(f"Invalid catalog year: {year}")

    existing = await db.scalar(
        select(MarketDataCatalog).where(
            MarketDataCatalog.market == market_key,
            MarketDataCatalog.symbol == symbol_key,
            MarketDataCatalog.timeframe == timeframe_key,
            MarketDataCatalog.session == session_key,
            MarketDataCatalog.year == year_key,
        )
    )

    if existing is None:
        existing = MarketDataCatalog(
            market=market_key,
            symbol=symbol_key,
            timeframe=timeframe_key,
            session=session_key,
            year=year_key,
            start_date=_ensure_utc(start_date),
            end_date=_ensure_utc(end_date),
            row_count=max(int(row_count), 0),
            file_path=str(file_path),
            file_size_bytes=max(int(file_size_bytes), 0),
        )
        db.add(existing)
        await db.flush()
        return _to_entry(existing)

    existing.start_date = _ensure_utc(start_date)
    existing.end_date = _ensure_utc(end_date)
    existing.row_count = max(int(row_count), 0)
    existing.file_path = str(file_path)
    existing.file_size_bytes = max(int(file_size_bytes), 0)
    await db.flush()
    return _to_entry(existing)


async def upsert_catalog_entry_from_parquet(
    db: AsyncSession,
    *,
    market: str,
    symbol: str,
    timeframe: str,
    session: str,
    year: int,
    file_path: str | Path,
) -> CatalogEntry:
    resolved_file = Path(file_path).expanduser().resolve()
    start_date, end_date, row_count, file_size_bytes = await asyncio.to_thread(
        _read_parquet_stats,
        resolved_file,
    )
    return await upsert_catalog_entry(
        db,
        market=market,
        symbol=symbol,
        timeframe=timeframe,
        session=session,
        year=year,
        start_date=start_date,
        end_date=end_date,
        row_count=row_count,
        file_path=str(resolved_file),
        file_size_bytes=file_size_bytes,
    )


async def get_symbol_coverage(
    db: AsyncSession,
    *,
    market: str,
    symbol: str,
    timeframe: str,
) -> list[CatalogEntry]:
    loader = DataLoader()
    market_key = loader.normalize_market(market)
    symbol_key = _normalize_symbol(symbol)
    timeframe_key = _normalize_timeframe(timeframe)
    rows = await db.scalars(
        select(MarketDataCatalog)
        .where(
            MarketDataCatalog.market == market_key,
            MarketDataCatalog.symbol == symbol_key,
            MarketDataCatalog.timeframe == timeframe_key,
        )
        .order_by(MarketDataCatalog.year.asc())
    )
    return [_to_entry(item) for item in rows.all()]


async def mark_accessed(
    db: AsyncSession,
    *,
    market: str,
    symbol: str,
    timeframe: str,
    year: int,
) -> None:
    loader = DataLoader()
    market_key = loader.normalize_market(market)
    symbol_key = _normalize_symbol(symbol)
    timeframe_key = _normalize_timeframe(timeframe)
    target = await db.scalar(
        select(MarketDataCatalog).where(
            MarketDataCatalog.market == market_key,
            MarketDataCatalog.symbol == symbol_key,
            MarketDataCatalog.timeframe == timeframe_key,
            MarketDataCatalog.year == int(year),
        )
    )
    if target is None:
        return
    target.last_accessed_at = datetime.now(UTC)
    await db.flush()


async def scan_and_sync_catalog(
    db: AsyncSession,
    data_dir: Path,
) -> int:
    root = Path(data_dir).expanduser().resolve()
    if not root.exists():
        return 0

    loader = DataLoader(data_dir=root)
    synced = 0

    for market_dir in sorted(root.iterdir()):
        if not market_dir.is_dir():
            continue
        market_raw = market_dir.name
        try:
            market_key = loader.normalize_market(market_raw)
        except ValueError:
            logger.debug(
                "[market-data-catalog] skip unknown market dir=%s",
                market_dir,
            )
            continue

        for parquet_file in sorted(market_dir.glob("*.parquet")):
            parsed = _parse_catalog_filename(file_path=parquet_file, loader=loader)
            if parsed is None:
                continue
            symbol, timeframe, session, year = parsed
            try:
                await upsert_catalog_entry_from_parquet(
                    db,
                    market=market_key,
                    symbol=symbol,
                    timeframe=timeframe,
                    session=session,
                    year=year,
                    file_path=parquet_file,
                )
                synced += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[market-data-catalog] skip file=%s error=%s:%s",
                    parquet_file,
                    type(exc).__name__,
                    exc,
                )

    return synced
