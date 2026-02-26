"""Load historical OHLCV market data from local parquet files."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True, slots=True)
class _ParquetFileInfo:
    path: Path
    symbol: str
    timeframe: str
    session: str
    year: int


class DataLoader:
    """Historical data loader with timeframe resampling and metadata APIs.

    File naming convention:
    ``{SYMBOL}_{timeframe}_{session}_{year}.parquet``.

    Examples:
    - ``BTCUSD_1min_eth_2024.parquet``
    - ``SPY_5min_rth_2024.parquet``
    """

    # Raw file timeframes currently present in data files.
    FILE_TIMEFRAME_MAP: dict[str, str] = {
        "1m": "1min",
        "5m": "5min",
    }
    FILE_TIMEFRAME_MAP_REVERSE: dict[str, str] = {
        value: key for key, value in FILE_TIMEFRAME_MAP.items()
    }

    # Resample targets. Add new items here when higher cycles are needed.
    RESAMPLE_RULES: dict[str, str] = {
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "1h",
        "2h": "2h",
        "4h": "4h",
        "1d": "1D",
    }

    TIMEFRAME_MINUTES: dict[str, int] = {
        "1m": 1,
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1h": 60,
        "2h": 120,
        "4h": 240,
        "1d": 1440,
    }

    MARKET_ALIASES: dict[str, str] = {
        "crypto": "crypto",
        "forex": "forex",
        "futures": "futures",
        "stock": "us_stocks",
        "stocks": "us_stocks",
        "us_stock": "us_stocks",
        "us_stocks": "us_stocks",
    }

    MARKET_SESSION_MAP: dict[str, str] = {
        "crypto": "eth",
        "forex": "eth",
        "futures": "eth",
        "us_stocks": "rth",
    }

    REQUIRED_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")

    def __init__(self, data_dir: str | Path | None = None) -> None:
        self.data_dir = Path(data_dir or "data").expanduser().resolve()
        self._symbol_metadata_cache: dict[tuple[str, str], dict[str, Any]] = {}
        self._time_bounds_cache: dict[Path, tuple[datetime, datetime]] = {}
        self._symbol_files_cache: dict[
            tuple[str, str, str],
            tuple[_ParquetFileInfo, ...],
        ] = {}

    @property
    def supported_timeframes(self) -> list[str]:
        return sorted(self.TIMEFRAME_MINUTES, key=self._timeframe_sort_key)

    def normalize_market(self, market: str) -> str:
        market_key = market.strip().lower()
        if market_key in self.MARKET_ALIASES:
            return self.MARKET_ALIASES[market_key]
        custom_market_dir = self.data_dir / market_key
        if custom_market_dir.exists():
            return market_key
        raise ValueError(f"Unsupported market: {market}")

    def get_available_markets(self) -> list[str]:
        """Return market directory names that currently have parquet data."""
        if not self.data_dir.exists():
            return []

        markets: list[str] = []
        for child in sorted(self.data_dir.iterdir()):
            if not child.is_dir():
                continue
            if any(child.glob("*.parquet")):
                markets.append(child.name)
        return markets

    def load(
        self,
        market: str,
        symbol: str,
        timeframe: str,
        start_date: datetime | str,
        end_date: datetime | str,
    ) -> pd.DataFrame:
        """Load OHLCV data for the requested range and timeframe."""
        market_key = self.normalize_market(market)
        symbol_key = self._normalize_symbol(symbol)
        timeframe_key = self._normalize_timeframe(timeframe)
        start_utc = self._ensure_utc(start_date)
        end_utc = self._ensure_utc(end_date)
        if end_utc < start_utc:
            raise ValueError("end_date must be greater than or equal to start_date")

        market_dir = self.data_dir / market_key
        if not market_dir.exists():
            raise FileNotFoundError(f"Market directory not found: {market_dir}")

        session = self.MARKET_SESSION_MAP.get(market_key, "eth")
        years = range(start_utc.year, end_utc.year + 1)
        source_timeframes = self._source_timeframe_candidates(timeframe_key)

        loaded_frames: list[pd.DataFrame] = []
        loaded_source_timeframes: set[str] = set()

        for year in years:
            selected_path: Path | None = None
            selected_source_tf: str | None = None

            for source_tf in source_timeframes:
                file_tf = self.FILE_TIMEFRAME_MAP[source_tf]
                candidate = market_dir / (
                    f"{symbol_key}_{file_tf}_{session}_{year}.parquet"
                )
                if candidate.exists():
                    selected_path = candidate
                    selected_source_tf = source_tf
                    break

            if selected_path is None or selected_source_tf is None:
                continue

            loaded_frames.append(self._read_parquet(selected_path))
            loaded_source_timeframes.add(selected_source_tf)

        if not loaded_frames:
            raise FileNotFoundError(
                f"No data files found for {symbol_key} in {market_key} "
                f"for years {start_utc.year}-{end_utc.year}"
            )

        merged = pd.concat(loaded_frames, ignore_index=True)
        prepared = self._prepare_dataframe(merged)

        if self._should_resample(
            target_timeframe=timeframe_key,
            loaded_source_timeframes=loaded_source_timeframes,
        ):
            prepared = self._resample(prepared, timeframe_key)

        filtered = prepared.loc[start_utc:end_utc]
        if filtered.empty:
            raise ValueError(
                f"No data found for {symbol_key} in [{start_utc}, {end_utc}]"
            )

        return filtered[list(self.REQUIRED_COLUMNS)]

    def get_available_symbols(self, market: str) -> list[str]:
        """Return all symbols that have local parquet data for the market."""
        market_key = self.normalize_market(market)
        market_dir = self.data_dir / market_key
        if not market_dir.exists():
            return []

        session = self.MARKET_SESSION_MAP.get(market_key, "eth")
        symbols: set[str] = set()
        for parquet_file in market_dir.glob("*.parquet"):
            info = self._parse_filename(parquet_file)
            if info is None or info.session != session:
                continue
            symbols.add(info.symbol)
        return sorted(symbols)

    def get_available_timeframes(
        self,
        market: str,
        symbol: str,
        *,
        include_resampled: bool = True,
    ) -> list[str]:
        """Return direct (and optionally resampled) timeframes for a symbol."""
        file_timeframes = self._get_symbol_file_timeframes(market=market, symbol=symbol)
        if not include_resampled:
            return sorted(file_timeframes, key=self._timeframe_sort_key)
        return self._expand_timeframes(file_timeframes)

    def get_symbol_metadata(self, market: str, symbol: str) -> dict[str, Any]:
        """Return symbol metadata including timerange and supported timeframes."""
        market_key = self.normalize_market(market)
        symbol_key = self._normalize_symbol(symbol)
        cache_key = (market_key, symbol_key)
        cached = self._symbol_metadata_cache.get(cache_key)
        if cached is not None:
            return deepcopy(cached)

        market_dir = self.data_dir / market_key
        if not market_dir.exists():
            raise FileNotFoundError(f"Market directory not found: {market_dir}")

        session = self.MARKET_SESSION_MAP.get(market_key, "eth")
        matching_files = self._list_symbol_files(
            market_dir=market_dir,
            symbol=symbol_key,
            session=session,
        )
        if not matching_files:
            raise FileNotFoundError(
                f"No data files found for symbol={symbol_key} in market={market_key}"
            )

        available_timerange = self._calculate_available_timerange(matching_files)
        file_timeframes = sorted(
            {item.timeframe for item in matching_files},
            key=self._timeframe_sort_key,
        )
        available_timeframes = self._expand_timeframes(file_timeframes)

        years_by_timeframe: dict[str, list[int]] = {}
        for timeframe_key in file_timeframes:
            years = sorted(
                {
                    item.year
                    for item in matching_files
                    if item.timeframe == timeframe_key
                }
            )
            years_by_timeframe[timeframe_key] = years

        metadata = {
            "market": market_key,
            "symbol": symbol_key,
            "session": session,
            "available_timerange": {
                "start": available_timerange[0].isoformat(),
                "end": available_timerange[1].isoformat(),
            },
            "available_timeframes": available_timeframes,
            "file_timeframes": file_timeframes,
            "years_by_timeframe": years_by_timeframe,
            "file_count": len(matching_files),
        }
        self._symbol_metadata_cache[cache_key] = metadata
        return deepcopy(metadata)

    def _get_symbol_file_timeframes(self, market: str, symbol: str) -> list[str]:
        market_key = self.normalize_market(market)
        symbol_key = self._normalize_symbol(symbol)
        market_dir = self.data_dir / market_key
        if not market_dir.exists():
            return []

        session = self.MARKET_SESSION_MAP.get(market_key, "eth")
        items = self._list_symbol_files(
            market_dir=market_dir,
            symbol=symbol_key,
            session=session,
        )
        return sorted({item.timeframe for item in items}, key=self._timeframe_sort_key)

    def _list_symbol_files(
        self,
        *,
        market_dir: Path,
        symbol: str,
        session: str,
    ) -> list[_ParquetFileInfo]:
        cache_key = (str(market_dir), symbol, session)
        cached = self._symbol_files_cache.get(cache_key)
        if cached is not None:
            return list(cached)

        results: list[_ParquetFileInfo] = []
        for parquet_file in market_dir.glob(f"{symbol}_*.parquet"):
            info = self._parse_filename(parquet_file)
            if info is None:
                continue
            if info.symbol != symbol or info.session != session:
                continue
            results.append(info)
        cached_results = tuple(results)
        self._symbol_files_cache[cache_key] = cached_results
        return list(cached_results)

    def _calculate_available_timerange(
        self,
        files: list[_ParquetFileInfo],
    ) -> tuple[datetime, datetime]:
        by_timeframe: dict[str, list[_ParquetFileInfo]] = {}
        for info in files:
            by_timeframe.setdefault(info.timeframe, []).append(info)

        global_start: datetime | None = None
        global_end: datetime | None = None

        for timeframe_files in by_timeframe.values():
            ordered = sorted(timeframe_files, key=lambda item: item.year)
            first_start, _ = self._read_time_bounds(ordered[0].path)
            _, last_end = self._read_time_bounds(ordered[-1].path)
            if global_start is None or first_start < global_start:
                global_start = first_start
            if global_end is None or last_end > global_end:
                global_end = last_end

        if global_start is None or global_end is None:
            raise ValueError("Unable to determine available_timerange")
        return global_start, global_end

    def _read_time_bounds(self, file_path: Path) -> tuple[datetime, datetime]:
        cache_key = file_path.resolve()
        cached = self._time_bounds_cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            frame = self._read_parquet(file_path, columns=["timestamp"])
        except Exception:  # noqa: BLE001
            frame = self._read_parquet(file_path)
        if "timestamp" in frame.columns:
            timestamp_series = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
            timestamp_series = timestamp_series.dropna()
        elif isinstance(frame.index, pd.DatetimeIndex):
            index = frame.index
            if index.tz is None:
                index = index.tz_localize("UTC")
            else:
                index = index.tz_convert("UTC")
            timestamp_series = pd.Series(index, dtype="datetime64[ns, UTC]")
        else:
            raise ValueError(f"Timestamp column missing in file: {file_path}")

        if timestamp_series.empty:
            raise ValueError(f"No timestamp data available in file: {file_path}")

        start = timestamp_series.min().to_pydatetime()
        end = timestamp_series.max().to_pydatetime()
        bounds = (start, end)
        self._time_bounds_cache[cache_key] = bounds
        return bounds

    def _prepare_dataframe(self, frame: pd.DataFrame) -> pd.DataFrame:
        if "timestamp" in frame.columns:
            timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
            valid_mask = timestamps.notna()
            frame = frame.loc[valid_mask].copy()
            frame.index = timestamps.loc[valid_mask]
        elif isinstance(frame.index, pd.DatetimeIndex):
            frame = frame.copy()
            if frame.index.tz is None:
                frame.index = frame.index.tz_localize("UTC")
            else:
                frame.index = frame.index.tz_convert("UTC")
        else:
            raise ValueError("Data must contain a timestamp column or DatetimeIndex")

        normalized_columns: dict[str, str] = {
            column.lower(): column for column in frame.columns
        }
        for required in self.REQUIRED_COLUMNS:
            source_column = normalized_columns.get(required)
            if source_column is None:
                raise ValueError(f"Missing required column: {required}")
            if source_column != required:
                frame[required] = frame[source_column]

        for column in self.REQUIRED_COLUMNS:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        cleaned = frame[list(self.REQUIRED_COLUMNS)].copy()
        cleaned = cleaned.dropna(subset=["open", "high", "low", "close"])
        cleaned["volume"] = cleaned["volume"].fillna(0.0)

        cleaned = cleaned.sort_index()
        if cleaned.index.has_duplicates:
            cleaned = cleaned[~cleaned.index.duplicated(keep="last")]
        return cleaned

    def _resample(self, frame: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        rule = self.RESAMPLE_RULES.get(timeframe)
        if rule is None:
            raise ValueError(f"Unsupported timeframe for resample: {timeframe}")

        resampled = frame.resample(rule).agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        resampled = resampled.dropna(subset=["open", "high", "low", "close"])
        resampled["volume"] = resampled["volume"].fillna(0.0)
        return resampled

    def _should_resample(
        self,
        *,
        target_timeframe: str,
        loaded_source_timeframes: set[str],
    ) -> bool:
        if target_timeframe not in self.FILE_TIMEFRAME_MAP:
            return True
        return any(source != target_timeframe for source in loaded_source_timeframes)

    def _expand_timeframes(self, file_timeframes: list[str]) -> list[str]:
        if not file_timeframes:
            return []

        available: set[str] = set()
        for timeframe in self.supported_timeframes:
            if self._can_derive_timeframe(target_timeframe=timeframe, file_timeframes=file_timeframes):
                available.add(timeframe)
        return sorted(available, key=self._timeframe_sort_key)

    def _can_derive_timeframe(
        self,
        *,
        target_timeframe: str,
        file_timeframes: list[str],
    ) -> bool:
        target_minutes = self.TIMEFRAME_MINUTES[target_timeframe]
        for source_tf in file_timeframes:
            source_minutes = self.TIMEFRAME_MINUTES.get(source_tf)
            if source_minutes is None:
                continue
            if source_minutes <= target_minutes:
                return True
        return False

    def _source_timeframe_candidates(self, target_timeframe: str) -> list[str]:
        target_minutes = self.TIMEFRAME_MINUTES[target_timeframe]
        candidates = [
            timeframe
            for timeframe in self.FILE_TIMEFRAME_MAP
            if self.TIMEFRAME_MINUTES[timeframe] <= target_minutes
        ]
        if not candidates:
            raise ValueError(
                f"Cannot find source timeframe for target timeframe: {target_timeframe}"
            )
        return sorted(candidates, key=self._timeframe_sort_key, reverse=True)

    def _parse_filename(self, file_path: Path) -> _ParquetFileInfo | None:
        parts = file_path.stem.split("_")
        if len(parts) < 4:
            return None

        symbol = "_".join(parts[:-3]).upper()
        timeframe_file = parts[-3].lower()
        session = parts[-2].lower()

        try:
            year = int(parts[-1])
        except ValueError:
            return None

        timeframe = self.FILE_TIMEFRAME_MAP_REVERSE.get(timeframe_file)
        if timeframe is None:
            return None

        return _ParquetFileInfo(
            path=file_path,
            symbol=symbol,
            timeframe=timeframe,
            session=session,
            year=year,
        )

    def _read_parquet(
        self,
        file_path: Path,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        try:
            if columns is None:
                return pd.read_parquet(file_path)
            return pd.read_parquet(file_path, columns=columns)
        except ImportError as exc:
            raise RuntimeError(
                "Parquet support is unavailable. Install 'pyarrow' or 'fastparquet'."
            ) from exc

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        normalized = symbol.strip().upper()
        if not normalized:
            raise ValueError("symbol cannot be empty")
        return normalized

    def _normalize_timeframe(self, timeframe: str) -> str:
        normalized = timeframe.strip().lower()
        if normalized not in self.TIMEFRAME_MINUTES:
            supported = ", ".join(self.supported_timeframes)
            raise ValueError(
                f"Unsupported timeframe '{timeframe}'. Supported: {supported}"
            )
        return normalized

    @staticmethod
    def _ensure_utc(value: datetime | str) -> datetime:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")
        return timestamp.to_pydatetime().astimezone(UTC)

    def _timeframe_sort_key(self, timeframe: str) -> int:
        return self.TIMEFRAME_MINUTES[timeframe]
