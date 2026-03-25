"""TradingView Charting Library datafeed endpoints."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from math import ceil
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.dependencies import get_db
from apps.api.middleware.auth import get_current_user
from apps.api.routes import market_data as market_data_route
from apps.api.schemas.charting import (
    ChartingBarResponse,
    ChartingDatafeedConfigResponse,
    ChartingExchangeResponse,
    ChartingHistoryMetadataResponse,
    ChartingHistoryResponse,
    ChartingResolveSymbolResponse,
    ChartingSearchSymbolResponse,
    ChartingSymbolTypeResponse,
)
from apps.api.services.trading_queue_service import enqueue_market_data_refresh
from packages.domain.market_data.data import DataLoader
from packages.domain.market_data.refresh_dedupe import reserve_market_data_refresh_slot
from packages.domain.market_data.runtime import RuntimeBar, market_data_runtime
from packages.infra.db.models.market_data_catalog import MarketDataCatalog
from packages.infra.db.models.user import User
from packages.infra.providers.market_data.alpaca_rest import AlpacaRestProvider
from packages.infra.providers.trading.adapters.base import OhlcvBar
from packages.shared_settings.schema.settings import settings

router = APIRouter(prefix="/chart-datafeed", tags=["chart-datafeed"])
logger = logging.getLogger(__name__)

_MARKET_ORDER = {"crypto": 0, "stocks": 1, "forex": 2, "futures": 3, "commodities": 4}
_MARKET_TYPE = {
    "crypto": "crypto",
    "stocks": "stock",
    "forex": "forex",
    "futures": "futures",
    "commodities": "commodity",
}
_MARKET_EXCHANGE = {
    "crypto": "CRYPTO",
    "stocks": "NASDAQ",
    "forex": "FX",
    "futures": "CME",
    "commodities": "COMMODITIES",
}
_MARKET_SESSION = {
    "crypto": "24x7",
    "stocks": "0930-1600",
    "forex": "24x7",
    "futures": "24x7",
    "commodities": "24x7",
}
_MARKET_TIMEZONE = {
    "crypto": "Etc/UTC",
    "stocks": "America/New_York",
    "forex": "Etc/UTC",
    "futures": "America/Chicago",
    "commodities": "Etc/UTC",
}
_MARKET_PRICESCALE = {
    "crypto": 100,
    "stocks": 100,
    "forex": 100000,
    "futures": 100,
    "commodities": 100,
}
_LOCAL_DATA_LOADER = DataLoader(
    data_dir=Path(__file__).resolve().parents[3] / "data",
)


@dataclass(frozen=True, slots=True)
class _SymbolDescriptor:
    market: str
    symbol: str
    description: str
    exchange: str
    listed_exchange: str
    symbol_type: str
    session: str
    timezone: str
    pricescale: int
    volume_precision: int = 2


_FEATURED_SYMBOLS: tuple[_SymbolDescriptor, ...] = (
    _SymbolDescriptor(
        market="crypto",
        symbol="BTCUSD",
        description="Bitcoin / US Dollar",
        exchange="CRYPTO",
        listed_exchange="CRYPTO",
        symbol_type="crypto",
        session="24x7",
        timezone="Etc/UTC",
        pricescale=100,
        volume_precision=4,
    ),
    _SymbolDescriptor(
        market="crypto",
        symbol="ETHUSD",
        description="Ethereum / US Dollar",
        exchange="CRYPTO",
        listed_exchange="CRYPTO",
        symbol_type="crypto",
        session="24x7",
        timezone="Etc/UTC",
        pricescale=100,
        volume_precision=4,
    ),
    _SymbolDescriptor(
        market="crypto",
        symbol="SOLUSD",
        description="Solana / US Dollar",
        exchange="CRYPTO",
        listed_exchange="CRYPTO",
        symbol_type="crypto",
        session="24x7",
        timezone="Etc/UTC",
        pricescale=100,
        volume_precision=4,
    ),
    _SymbolDescriptor(
        market="stocks",
        symbol="AAPL",
        description="Apple Inc.",
        exchange="NASDAQ",
        listed_exchange="NASDAQ",
        symbol_type="stock",
        session="0930-1600",
        timezone="America/New_York",
        pricescale=100,
    ),
    _SymbolDescriptor(
        market="stocks",
        symbol="NVDA",
        description="NVIDIA Corporation",
        exchange="NASDAQ",
        listed_exchange="NASDAQ",
        symbol_type="stock",
        session="0930-1600",
        timezone="America/New_York",
        pricescale=100,
    ),
    _SymbolDescriptor(
        market="forex",
        symbol="EURUSD",
        description="Euro / US Dollar",
        exchange="FX",
        listed_exchange="FX",
        symbol_type="forex",
        session="24x7",
        timezone="Etc/UTC",
        pricescale=100000,
        volume_precision=5,
    ),
)


def _normalize_market(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if not normalized:
        return None
    if normalized not in _MARKET_ORDER:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "INVALID_MARKET", "message": f"Unsupported market '{value}'."},
        )
    return normalized


def _normalize_symbol(value: str) -> str:
    normalized = value.strip().upper()
    if not normalized:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "INVALID_SYMBOL", "message": "symbol cannot be empty."},
        )
    return normalized


def _supported_timeframes() -> tuple[str, ...]:
    rows = ["1m"]
    rows.extend(
        timeframe.strip().lower()
        for timeframe in settings.market_data_aggregate_timeframes_csv.split(",")
        if timeframe.strip()
    )
    ordered: list[str] = []
    seen: set[str] = set()
    for item in rows:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return tuple(
        sorted(
            ordered,
            key=lambda timeframe: (
                market_data_route._timeframe_step(timeframe)
                or timedelta.max
            ),
        )
    )


def _timeframe_to_resolution(timeframe: str) -> str:
    normalized = timeframe.strip().lower()
    if normalized.endswith("m"):
        return str(int(normalized[:-1] or "0"))
    if normalized.endswith("h"):
        return str(int(normalized[:-1] or "0") * 60)
    if normalized.endswith("d"):
        days = int(normalized[:-1] or "0")
        if days <= 0:
            raise ValueError(f"Unsupported timeframe '{timeframe}'.")
        return "1D" if days == 1 else f"{days}D"
    raise ValueError(f"Unsupported timeframe '{timeframe}'.")


def _resolution_to_timeframe(value: str) -> str:
    normalized = value.strip().upper()
    if not normalized:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "INVALID_RESOLUTION", "message": "resolution cannot be empty."},
        )
    if normalized == "D":
        return "1d"
    if normalized.endswith("D"):
        try:
            days = int(normalized[:-1] or "1")
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "code": "INVALID_RESOLUTION",
                    "message": f"Unsupported resolution '{value}'.",
                },
            ) from exc
        return f"{days}d"
    try:
        minutes = int(normalized)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "INVALID_RESOLUTION",
                "message": f"Unsupported resolution '{value}'.",
            },
        ) from exc
    if minutes <= 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "INVALID_RESOLUTION",
                "message": f"Unsupported resolution '{value}'.",
            },
        )
    if minutes % 60 == 0 and minutes >= 60:
        return f"{minutes // 60}h"
    return f"{minutes}m"


def _supported_resolutions() -> list[str]:
    return [_timeframe_to_resolution(item) for item in _supported_timeframes()]


def _intraday_multipliers() -> list[str]:
    values: list[str] = []
    for timeframe in _supported_timeframes():
        if timeframe.endswith("d"):
            continue
        values.append(_timeframe_to_resolution(timeframe))
    return values


def _daily_multipliers() -> list[str]:
    values: list[str] = []
    for timeframe in _supported_timeframes():
        if not timeframe.endswith("d"):
            continue
        resolution = _timeframe_to_resolution(timeframe)
        values.append(resolution[:-1] if resolution.endswith("D") else resolution)
    return values or ["1"]


def _market_prefix_to_market(prefix: str) -> str | None:
    normalized = prefix.strip().lower()
    if normalized in _MARKET_ORDER:
        return normalized
    if normalized in {"nasdaq", "nyse", "amex", "stocks"}:
        return "stocks"
    if normalized in {"crypto", "binance"}:
        return "crypto"
    if normalized in {"fx", "forex"}:
        return "forex"
    if normalized in {"cme", "futures"}:
        return "futures"
    return None


def _parse_symbol_request(
    symbol_name: str,
    market_hint: str | None = None,
) -> tuple[str, str | None]:
    raw = _normalize_symbol(symbol_name)
    normalized_market = _normalize_market(market_hint)
    if ":" not in raw:
        return raw, normalized_market
    prefix, symbol = raw.split(":", 1)
    inferred_market = _market_prefix_to_market(prefix)
    return _normalize_symbol(symbol), normalized_market or inferred_market


def _fallback_descriptor(market: str, symbol: str) -> _SymbolDescriptor:
    exchange = _MARKET_EXCHANGE.get(market, market.upper())
    return _SymbolDescriptor(
        market=market,
        symbol=symbol,
        description=f"{symbol} ({market})",
        exchange=exchange,
        listed_exchange=exchange,
        symbol_type=_MARKET_TYPE.get(market, "stock"),
        session=_MARKET_SESSION.get(market, "24x7"),
        timezone=_MARKET_TIMEZONE.get(market, "Etc/UTC"),
        pricescale=_MARKET_PRICESCALE.get(market, 100),
    )


def _descriptor_to_search_response(
    descriptor: _SymbolDescriptor,
) -> ChartingSearchSymbolResponse:
    full_name = f"{descriptor.exchange}:{descriptor.symbol}"
    return ChartingSearchSymbolResponse(
        symbol=descriptor.symbol,
        full_name=full_name,
        description=descriptor.description,
        exchange=descriptor.exchange,
        ticker=descriptor.symbol,
        type=descriptor.symbol_type,
    )


def _descriptor_to_resolve_response(
    descriptor: _SymbolDescriptor,
) -> ChartingResolveSymbolResponse:
    supported_resolutions = _supported_resolutions()
    return ChartingResolveSymbolResponse(
        name=descriptor.symbol,
        ticker=descriptor.symbol,
        full_name=f"{descriptor.exchange}:{descriptor.symbol}",
        description=descriptor.description,
        long_description=descriptor.description,
        type=descriptor.symbol_type,
        session=descriptor.session,
        session_display=descriptor.session,
        exchange=descriptor.exchange,
        listed_exchange=descriptor.listed_exchange,
        timezone=descriptor.timezone,
        pricescale=descriptor.pricescale,
        minmov=1,
        has_intraday=True,
        supported_resolutions=supported_resolutions,
        intraday_multipliers=_intraday_multipliers(),
        has_daily=True,
        daily_multipliers=_daily_multipliers(),
        has_weekly_and_monthly=False,
        weekly_multipliers=[],
        monthly_multipliers=[],
        has_empty_bars=False,
        visible_plots_set="ohlcv",
        volume_precision=descriptor.volume_precision,
        data_status="streaming",
        delay=0,
        minsy_market=descriptor.market,
    )


def _bar_timestamp_ms(bar: RuntimeBar | OhlcvBar, *, timeframe: str) -> int:
    timestamp = bar.timestamp.astimezone(UTC)
    if timeframe.endswith("d"):
        timestamp = datetime(
            timestamp.year,
            timestamp.month,
            timestamp.day,
            tzinfo=UTC,
        )
    return int(timestamp.timestamp() * 1000)


def _bar_to_response(
    bar: RuntimeBar | OhlcvBar,
    *,
    timeframe: str,
) -> ChartingBarResponse:
    return ChartingBarResponse(
        time=_bar_timestamp_ms(bar, timeframe=timeframe),
        open=float(bar.open),
        high=float(bar.high),
        low=float(bar.low),
        close=float(bar.close),
        volume=float(bar.volume),
    )


def _slice_history_window(
    *,
    bars: list[RuntimeBar | OhlcvBar],
    from_dt: datetime,
    to_dt: datetime,
    count_back: int,
) -> tuple[list[RuntimeBar | OhlcvBar], int | None]:
    ordered = sorted(
        (
            bar
            for bar in bars
            if isinstance(getattr(bar, "timestamp", None), datetime)
        ),
        key=lambda item: item.timestamp,
    )
    in_range = [
        bar
        for bar in ordered
        if from_dt <= bar.timestamp.astimezone(UTC) < to_dt
    ]
    if len(in_range) >= count_back:
        # TradingView expects the full requested [from, to) range even when the
        # number of bars inside that range exceeds countBack. Truncating the
        # range here can break pending subscriber normalization for studies and
        # backfill requests.
        return in_range, None

    older = [
        bar for bar in ordered if bar.timestamp.astimezone(UTC) < from_dt
    ]
    next_time = (
        int(older[-1].timestamp.astimezone(UTC).timestamp() * 1000)
        if older
        else None
    )
    needed = max(0, count_back - len(in_range))
    if needed > 0 and older:
        in_range = older[-needed:] + in_range
    return in_range, next_time


def _aggregate_bars_from_source(
    *,
    source_bars: list[OhlcvBar],
    source_timeframe: str,
    target_timeframe: str,
    target_bars: int,
) -> list[OhlcvBar]:
    source_step = market_data_route._timeframe_step(source_timeframe)
    target_step = market_data_route._timeframe_step(target_timeframe)
    if source_step is None or target_step is None:
        return []
    source_ms = int(source_step.total_seconds() * 1000)
    target_ms = int(target_step.total_seconds() * 1000)
    if source_ms <= 0 or target_ms <= 0 or target_ms < source_ms:
        return []
    if target_ms % source_ms != 0:
        return []

    ordered = sorted(source_bars, key=lambda item: item.timestamp)
    if not ordered:
        return []

    buckets: dict[int, dict[str, Decimal | datetime]] = {}
    for bar in ordered:
        ts = bar.timestamp.astimezone(UTC)
        ts_ms = int(ts.timestamp() * 1000)
        bucket_ts_ms = (ts_ms // target_ms) * target_ms
        current = buckets.get(bucket_ts_ms)
        open_ = Decimal(str(bar.open))
        high = Decimal(str(bar.high))
        low = Decimal(str(bar.low))
        close = Decimal(str(bar.close))
        volume = Decimal(str(bar.volume))
        if current is None:
            buckets[bucket_ts_ms] = {
                "timestamp": datetime.fromtimestamp(bucket_ts_ms / 1000.0, tz=UTC),
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
            continue
        current["high"] = max(current["high"], high)
        current["low"] = min(current["low"], low)
        current["close"] = close
        current["volume"] = current["volume"] + volume

    ordered_buckets = [buckets[key] for key in sorted(buckets)]
    if len(ordered_buckets) > target_bars:
        ordered_buckets = ordered_buckets[-target_bars:]
    return [
        OhlcvBar(
            timestamp=item["timestamp"],  # type: ignore[arg-type]
            open=item["open"],  # type: ignore[arg-type]
            high=item["high"],  # type: ignore[arg-type]
            low=item["low"],  # type: ignore[arg-type]
            close=item["close"],  # type: ignore[arg-type]
            volume=item["volume"],  # type: ignore[arg-type]
        )
        for item in ordered_buckets
    ]


async def _catalog_descriptors(
    *,
    db: AsyncSession,
    query_text: str | None,
    market: str | None,
    limit: int,
) -> list[_SymbolDescriptor]:
    stmt = select(MarketDataCatalog.market, MarketDataCatalog.symbol).distinct()
    if market:
        stmt = stmt.where(MarketDataCatalog.market == market)
    if query_text:
        stmt = stmt.where(MarketDataCatalog.symbol.ilike(f"%{query_text}%"))
    stmt = stmt.order_by(MarketDataCatalog.market, MarketDataCatalog.symbol).limit(
        max(limit * 3, limit)
    )
    rows = (await db.execute(stmt)).all()
    return [_fallback_descriptor(market=row[0], symbol=row[1]) for row in rows]


async def _resolve_symbol_descriptor(
    *,
    db: AsyncSession,
    symbol_name: str,
    market_hint: str | None = None,
) -> _SymbolDescriptor | None:
    symbol, normalized_market = _parse_symbol_request(symbol_name, market_hint)

    for descriptor in _FEATURED_SYMBOLS:
        if descriptor.symbol != symbol:
            continue
        if normalized_market is not None and descriptor.market != normalized_market:
            continue
        return descriptor

    stmt = select(MarketDataCatalog.market, MarketDataCatalog.symbol).distinct().where(
        MarketDataCatalog.symbol == symbol
    )
    if normalized_market is not None:
        stmt = stmt.where(MarketDataCatalog.market == normalized_market)
    stmt = stmt.order_by(MarketDataCatalog.market, MarketDataCatalog.symbol).limit(5)
    rows = (await db.execute(stmt)).all()
    if not rows:
        return None

    ordered = sorted(
        rows,
        key=lambda row: _MARKET_ORDER.get(row[0], 999),
    )
    market = ordered[0][0]
    return _fallback_descriptor(market=market, symbol=symbol)


async def _fetch_provider_history(
    *,
    provider: AlpacaRestProvider,
    market: str,
    symbol: str,
    timeframe: str,
    from_dt: datetime,
    to_dt: datetime,
    limit: int,
) -> list[OhlcvBar]:
    step = market_data_route._timeframe_step(timeframe)
    if step is None:
        return []
    since = _provider_history_since(
        from_dt=from_dt,
        to_dt=to_dt,
        step=step,
        limit=limit,
    )
    rows = await provider.fetch_recent_bars(
        symbol=symbol,
        market=market,
        timeframe=timeframe,
        since=since,
        until=to_dt,
        limit=limit,
    )
    if rows or not (market == "crypto" and timeframe == "4h"):
        return rows

    fallback_limit = min(
        int(settings.market_data_ring_capacity_aggregated),
        max(limit * 4, 32),
    )
    fallback_since = from_dt - (timedelta(hours=1) * fallback_limit)
    source_rows = await provider.fetch_recent_bars(
        symbol=symbol,
        market=market,
        timeframe="1h",
        since=fallback_since,
        until=to_dt,
        limit=fallback_limit,
    )
    return _aggregate_bars_from_source(
        source_bars=source_rows,
        source_timeframe="1h",
        target_timeframe="4h",
        target_bars=limit,
    )


def _provider_history_since(
    *,
    from_dt: datetime,
    to_dt: datetime,
    step: timedelta,
    limit: int,
) -> datetime:
    lookback_bars = max(limit + 8, 32)
    bounded_from = from_dt - (step * 4)
    bounded_countback = to_dt - (step * lookback_bars)
    return min(bounded_from, bounded_countback)


def _prefer_newer_history(
    current: list[RuntimeBar | OhlcvBar],
    candidate: list[RuntimeBar | OhlcvBar],
) -> list[RuntimeBar | OhlcvBar]:
    if not candidate:
        return current
    if not current:
        return candidate
    current_latest = current[-1].timestamp
    candidate_latest = candidate[-1].timestamp
    if candidate_latest > current_latest:
        return candidate
    if candidate_latest < current_latest:
        return current
    return candidate if len(candidate) >= len(current) else current


def _merge_history_groups(
    *,
    groups: list[list[RuntimeBar | OhlcvBar]],
    limit: int,
) -> list[RuntimeBar | OhlcvBar]:
    merged: dict[datetime, RuntimeBar | OhlcvBar] = {}
    for group in groups:
        for item in group:
            timestamp = getattr(item, "timestamp", None)
            if not isinstance(timestamp, datetime):
                continue
            merged[timestamp.astimezone(UTC)] = item
    ordered = [merged[key] for key in sorted(merged)]
    if len(ordered) > limit:
        ordered = ordered[-limit:]
    return ordered


def _load_local_chart_history(
    *,
    market: str,
    symbol: str,
    timeframe: str,
    from_dt: datetime,
    to_dt: datetime,
    limit: int,
) -> list[OhlcvBar]:
    step = market_data_route._timeframe_step(timeframe)
    if step is None:
        return []
    since = _provider_history_since(
        from_dt=from_dt,
        to_dt=to_dt,
        step=step,
        limit=limit,
    )
    try:
        frame = _LOCAL_DATA_LOADER.load(
            market=market,
            symbol=symbol,
            timeframe=timeframe,
            start_date=since,
            end_date=to_dt,
        )
    except (FileNotFoundError, ValueError, OSError, ImportError):
        return []

    rows: list[OhlcvBar] = []
    for timestamp, row in frame.iterrows():
        ts_value = (
            timestamp.to_pydatetime()
            if hasattr(timestamp, "to_pydatetime")
            else timestamp
        )
        if not isinstance(ts_value, datetime):
            continue
        rows.append(
            OhlcvBar(
                timestamp=ts_value.astimezone(UTC),
                open=Decimal(str(row["open"])),
                high=Decimal(str(row["high"])),
                low=Decimal(str(row["low"])),
                close=Decimal(str(row["close"])),
                volume=Decimal(str(row["volume"])),
            )
        )
    return rows


async def _load_chart_history(
    *,
    market: str,
    symbol: str,
    timeframe: str,
    from_dt: datetime,
    to_dt: datetime,
    count_back: int,
) -> list[RuntimeBar | OhlcvBar]:
    step = market_data_route._timeframe_step(timeframe)
    if step is None:
        return []

    range_bars = max(
        1,
        int(ceil(max((to_dt - from_dt).total_seconds(), 0) / step.total_seconds())),
    )
    target_bars = min(5000, max(count_back + 32, range_bars + 8))

    bars = market_data_runtime.get_recent_bars(
        market=market,
        symbol=symbol,
        timeframe=timeframe,
        limit=target_bars,
    )
    sliced, _ = _slice_history_window(
        bars=bars,
        from_dt=from_dt,
        to_dt=to_dt,
        count_back=count_back,
    )
    if len(sliced) >= min(count_back, target_bars):
        return bars

    local_rows = _load_local_chart_history(
        market=market,
        symbol=symbol,
        timeframe=timeframe,
        from_dt=from_dt,
        to_dt=to_dt,
        limit=target_bars,
    )
    if local_rows:
        bars = _merge_history_groups(
            groups=[local_rows, list(bars)],
            limit=target_bars,
        )
        sliced, _ = _slice_history_window(
            bars=bars,
            from_dt=from_dt,
            to_dt=to_dt,
            count_back=count_back,
        )
        if len(sliced) >= min(count_back, target_bars):
            return bars

    if reserve_market_data_refresh_slot(market, symbol):
        try:
            enqueue_market_data_refresh(
                market=market,
                symbol=symbol,
                requested_timeframe=timeframe,
                min_bars=target_bars,
            )
        except Exception:
            pass

    provider = AlpacaRestProvider()
    try:
        provider_rows = await _fetch_provider_history(
            provider=provider,
            market=market,
            symbol=symbol,
            timeframe=timeframe,
            from_dt=from_dt,
            to_dt=to_dt,
            limit=target_bars,
        )
    finally:
        await provider.aclose()

    if provider_rows:
        market_data_runtime.hydrate_bars(
            market=market,
            symbol=symbol,
            timeframe=timeframe,
            bars=provider_rows,
        )
        runtime_rows = market_data_runtime.get_recent_bars(
            market=market,
            symbol=symbol,
            timeframe=timeframe,
            limit=target_bars,
        )
        bars = _merge_history_groups(
            groups=[local_rows, list(bars), list(runtime_rows or provider_rows)],
            limit=target_bars,
        ) or _prefer_newer_history(bars, runtime_rows or provider_rows)

    now_utc = datetime.now(UTC)
    if to_dt >= (now_utc - step):
        live_bar = await market_data_route._build_live_bar(
            market=market,
            symbol=symbol,
            timeframe=timeframe,
            historical_bars=[
                RuntimeBar(
                    timestamp=item.timestamp,
                    open=float(item.open),
                    high=float(item.high),
                    low=float(item.low),
                    close=float(item.close),
                    volume=float(item.volume),
                )
                for item in bars
            ],
        )
        if live_bar is not None:
            runtime_bars = [
                RuntimeBar(
                    timestamp=item.timestamp,
                    open=float(item.open),
                    high=float(item.high),
                    low=float(item.low),
                    close=float(item.close),
                    volume=float(item.volume),
                )
                for item in bars
            ]
            merged = market_data_route._merge_live_bar(
                bars=runtime_bars,
                live_bar=live_bar,
                limit=target_bars,
            )
            bars = merged

    return bars


@router.get("/config", response_model=ChartingDatafeedConfigResponse)
async def get_config(
    user: User = Depends(get_current_user),
) -> ChartingDatafeedConfigResponse:
    _ = user
    return ChartingDatafeedConfigResponse(
        supports_search=True,
        supports_group_request=False,
        supported_resolutions=_supported_resolutions(),
        supports_marks=False,
        supports_timescale_marks=False,
        supports_time=True,
        exchanges=[
            ChartingExchangeResponse(value="CRYPTO", name="Crypto", desc="Crypto"),
            ChartingExchangeResponse(value="NASDAQ", name="NASDAQ", desc="US Stocks"),
            ChartingExchangeResponse(value="FX", name="FX", desc="Forex"),
            ChartingExchangeResponse(value="CME", name="CME", desc="Futures"),
        ],
        symbols_types=[
            ChartingSymbolTypeResponse(name="Crypto", value="crypto"),
            ChartingSymbolTypeResponse(name="Stock", value="stock"),
            ChartingSymbolTypeResponse(name="Forex", value="forex"),
            ChartingSymbolTypeResponse(name="Futures", value="futures"),
        ],
    )


@router.get(
    "/searchSymbols",
    response_model=list[ChartingSearchSymbolResponse],
)
async def search_symbols(
    userInput: str = Query(default="", max_length=64),
    exchange: str = Query(default=""),
    symbolType: str = Query(default=""),
    limit: int = Query(default=30, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> list[ChartingSearchSymbolResponse]:
    _ = user
    query_text = userInput.strip().upper()
    exchange_filter = exchange.strip().upper()
    type_filter = symbolType.strip().lower()
    market_hint = _market_prefix_to_market(exchange_filter)
    candidates = list(_FEATURED_SYMBOLS)
    candidates.extend(
        await _catalog_descriptors(
            db=db,
            query_text=query_text or None,
            market=market_hint,
            limit=limit,
        )
    )

    rows: list[_SymbolDescriptor] = []
    seen: set[tuple[str, str]] = set()
    for descriptor in candidates:
        key = (descriptor.market, descriptor.symbol)
        if key in seen:
            continue
        seen.add(key)
        if market_hint is not None and descriptor.market != market_hint:
            continue
        if exchange_filter and descriptor.exchange.upper() != exchange_filter:
            continue
        if type_filter and descriptor.symbol_type != type_filter:
            continue
        if query_text:
            haystacks = (
                descriptor.symbol.upper(),
                descriptor.description.upper(),
                f"{descriptor.exchange}:{descriptor.symbol}".upper(),
            )
            if not any(query_text in item for item in haystacks):
                continue
        rows.append(descriptor)

    rows.sort(
        key=lambda descriptor: (
            0 if query_text and descriptor.symbol.startswith(query_text) else 1,
            descriptor.symbol.find(query_text) if query_text in descriptor.symbol else 999,
            _MARKET_ORDER.get(descriptor.market, 999),
            descriptor.symbol,
        )
    )
    return [
        _descriptor_to_search_response(item)
        for item in rows[:limit]
    ]


@router.get(
    "/resolveSymbol",
    response_model=ChartingResolveSymbolResponse,
)
async def resolve_symbol(
    symbolName: str = Query(min_length=1, max_length=64),
    market: str | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> ChartingResolveSymbolResponse:
    _ = user
    descriptor = await _resolve_symbol_descriptor(
        db=db,
        symbol_name=symbolName,
        market_hint=market,
    )
    if descriptor is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "UNKNOWN_SYMBOL",
                "message": f"Symbol '{symbolName}' is not available.",
            },
        )
    return _descriptor_to_resolve_response(descriptor)


@router.get(
    "/getBars",
    response_model=ChartingHistoryResponse,
    response_model_exclude_none=True,
)
async def get_bars(
    symbol: str = Query(min_length=1, max_length=64),
    resolution: str = Query(min_length=1, max_length=16),
    from_ts: int = Query(alias="from", ge=0),
    to_ts: int = Query(alias="to", ge=0),
    count_back: int = Query(alias="countBack", default=300, ge=1, le=5000),
    market: str | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> ChartingHistoryResponse:
    _ = user
    logger.info(
        "[chart-datafeed] getBars request symbol=%s market=%s resolution=%s from=%s to=%s countBack=%s",
        symbol,
        market,
        resolution,
        from_ts,
        to_ts,
        count_back,
    )
    descriptor = await _resolve_symbol_descriptor(
        db=db,
        symbol_name=symbol,
        market_hint=market,
    )
    if descriptor is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "UNKNOWN_SYMBOL",
                "message": f"Symbol '{symbol}' is not available.",
            },
        )

    timeframe = _resolution_to_timeframe(resolution)
    if timeframe not in _supported_timeframes():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "UNSUPPORTED_RESOLUTION",
                "message": f"Resolution '{resolution}' is not supported.",
            },
        )

    from_dt = datetime.fromtimestamp(from_ts, tz=UTC)
    to_dt = datetime.fromtimestamp(max(to_ts, from_ts + 1), tz=UTC)
    history_rows = await _load_chart_history(
        market=descriptor.market,
        symbol=descriptor.symbol,
        timeframe=timeframe,
        from_dt=from_dt,
        to_dt=to_dt,
        count_back=count_back,
    )
    sliced_rows, next_time = _slice_history_window(
        bars=history_rows,
        from_dt=from_dt,
        to_dt=to_dt,
        count_back=count_back,
    )
    first_bar_time = (
        _bar_timestamp_ms(sliced_rows[0], timeframe=timeframe) if sliced_rows else None
    )
    last_bar_time = (
        _bar_timestamp_ms(sliced_rows[-1], timeframe=timeframe) if sliced_rows else None
    )
    logger.info(
        "[chart-datafeed] getBars response symbol=%s market=%s resolution=%s timeframe=%s history_rows=%s sliced_rows=%s next_time=%s first_bar_time=%s last_bar_time=%s",
        descriptor.symbol,
        descriptor.market,
        resolution,
        timeframe,
        len(history_rows),
        len(sliced_rows),
        next_time,
        first_bar_time,
        last_bar_time,
    )
    if not sliced_rows:
        return ChartingHistoryResponse(
            bars=[],
            meta=ChartingHistoryMetadataResponse(noData=True, nextTime=next_time),
        )

    return ChartingHistoryResponse(
        bars=[_bar_to_response(bar, timeframe=timeframe) for bar in sliced_rows],
        meta=ChartingHistoryMetadataResponse(noData=False, nextTime=None),
    )
