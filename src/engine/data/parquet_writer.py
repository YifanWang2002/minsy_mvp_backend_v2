"""Append OHLCV rows into local parquet shards with timestamp de-duplication."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.engine.data import DataLoader


@dataclass(frozen=True, slots=True)
class ParquetAppendResult:
    """Write summary for one append operation."""

    rows_input: int
    rows_written: int
    files_touched: int


def append_ohlcv_rows(
    *,
    loader: DataLoader,
    market: str,
    symbol: str,
    timeframe: str,
    rows: pd.DataFrame,
) -> ParquetAppendResult:
    """Append rows to yearly parquet shards and drop duplicate timestamps."""

    market_key = loader.normalize_market(market)
    symbol_key = symbol.strip().upper()
    if not symbol_key:
        raise ValueError("symbol cannot be empty")

    timeframe_key = str(timeframe).strip().lower()
    if timeframe_key not in loader.FILE_TIMEFRAME_MAP:
        raise ValueError(
            f"Unsupported local write timeframe: {timeframe}. "
            f"Use one of {sorted(loader.FILE_TIMEFRAME_MAP)}."
        )

    prepared = _prepare_rows(rows)
    if prepared.empty:
        return ParquetAppendResult(rows_input=0, rows_written=0, files_touched=0)

    session = loader.MARKET_SESSION_MAP.get(market_key, "eth")
    market_dir = loader.data_dir / market_key
    market_dir.mkdir(parents=True, exist_ok=True)
    file_timeframe = loader.FILE_TIMEFRAME_MAP[timeframe_key]

    rows_input = len(prepared)
    rows_written = 0
    files_touched = 0

    for year, year_rows in prepared.groupby(prepared["timestamp"].dt.year):
        file_path = market_dir / f"{symbol_key}_{file_timeframe}_{session}_{int(year)}.parquet"
        merged, added = _merge_with_existing(file_path=file_path, rows=year_rows)
        merged.to_parquet(file_path, index=False)
        rows_written += added
        files_touched += 1

    return ParquetAppendResult(
        rows_input=rows_input,
        rows_written=rows_written,
        files_touched=files_touched,
    )


def _prepare_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"rows missing columns: {missing}")

    normalized = rows.loc[:, ["timestamp", "open", "high", "low", "close", "volume"]].copy()
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], utc=True, errors="coerce")
    normalized = normalized.dropna(subset=["timestamp"])
    normalized = normalized.sort_values("timestamp")
    for column in ("open", "high", "low", "close", "volume"):
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    normalized = normalized.dropna(subset=["open", "high", "low", "close", "volume"])
    normalized = normalized.drop_duplicates(subset=["timestamp"], keep="last")
    return normalized.reset_index(drop=True)


def _merge_with_existing(*, file_path: Path, rows: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if file_path.exists():
        existing = pd.read_parquet(file_path)
        existing = _prepare_rows(existing)
    else:
        existing = pd.DataFrame(columns=rows.columns)

    before = len(existing)
    if before == 0:
        merged = rows.copy()
    else:
        merged = pd.concat([existing, rows], ignore_index=True)
    merged = merged.drop_duplicates(subset=["timestamp"], keep="last")
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    added = max(0, len(merged) - before)
    return merged, added
