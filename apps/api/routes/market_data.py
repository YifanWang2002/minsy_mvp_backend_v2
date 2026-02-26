"""Market-data snapshot and subscription endpoints."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.middleware.auth import get_current_user
from apps.api.schemas.events import (
    MarketDataBarResponse,
    MarketDataBarsResponse,
    MarketDataQuoteResponse,
    MarketDataSubscriptionResponse,
)
from apps.api.schemas.requests import MarketDataSubscriptionRequest
from apps.api.dependencies import get_db
from packages.infra.providers.market_data.alpaca_rest import AlpacaRestProvider
from packages.domain.market_data.runtime import market_data_runtime
from packages.infra.db.models.market_data_error_event import MarketDataErrorEvent
from packages.infra.db.models.user import User
from apps.api.services.trading_queue_service import enqueue_market_data_refresh

router = APIRouter(prefix="/market-data", tags=["market-data"])


def _normalize_market(market: str) -> str:
    normalized = market.strip().lower()
    if normalized not in {"stocks", "crypto", "forex", "futures", "commodities"}:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "INVALID_MARKET", "message": f"Unsupported market '{market}'."},
        )
    return normalized


@router.get("/quote", response_model=MarketDataQuoteResponse)
async def get_latest_quote(
    symbol: str = Query(min_length=1, max_length=32),
    market: str = Query(default="stocks"),
    refresh_if_missing: bool = Query(default=False),
    user: User = Depends(get_current_user),
) -> MarketDataQuoteResponse:
    _ = user
    normalized_market = _normalize_market(market)
    normalized_symbol = symbol.strip().upper()

    quote = market_data_runtime.get_latest_quote(
        market=normalized_market,
        symbol=normalized_symbol,
    )
    if quote is None and refresh_if_missing:
        provider = AlpacaRestProvider()
        try:
            quote = await provider.fetch_quote(symbol=normalized_symbol, market=normalized_market)
            if quote is not None:
                market_data_runtime.upsert_quote(
                    market=normalized_market,
                    symbol=normalized_symbol,
                    quote=quote,
                )
        finally:
            await provider.aclose()

    if quote is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "QUOTE_NOT_FOUND", "message": "Quote not found in runtime cache."},
        )

    return MarketDataQuoteResponse(
        market=normalized_market,
        symbol=normalized_symbol,
        bid=float(quote.bid) if quote.bid is not None else None,
        ask=float(quote.ask) if quote.ask is not None else None,
        last=float(quote.last) if quote.last is not None else None,
        timestamp=quote.timestamp,
    )


@router.get("/bars", response_model=MarketDataBarsResponse)
async def get_recent_bars(
    symbol: str = Query(min_length=1, max_length=32),
    market: str = Query(default="stocks"),
    timeframe: str = Query(default="1m"),
    limit: int = Query(default=120, ge=1, le=5000),
    refresh_if_empty: bool = Query(default=False),
    user: User = Depends(get_current_user),
) -> MarketDataBarsResponse:
    _ = user
    normalized_market = _normalize_market(market)
    normalized_symbol = symbol.strip().upper()
    normalized_timeframe = timeframe.strip().lower()

    bars = market_data_runtime.get_recent_bars(
        market=normalized_market,
        symbol=normalized_symbol,
        timeframe=normalized_timeframe,
        limit=limit,
    )
    if not bars and refresh_if_empty:
        enqueue_market_data_refresh(market=normalized_market, symbol=normalized_symbol)
        bars = market_data_runtime.get_recent_bars(
            market=normalized_market,
            symbol=normalized_symbol,
            timeframe=normalized_timeframe,
            limit=limit,
        )

    return MarketDataBarsResponse(
        market=normalized_market,
        symbol=normalized_symbol,
        timeframe=normalized_timeframe,
        bars=[
            MarketDataBarResponse(
                timestamp=bar.timestamp,
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=bar.volume,
            )
            for bar in bars
        ],
    )


@router.get("/subscriptions", response_model=MarketDataSubscriptionResponse)
async def get_subscription_state(
    user: User = Depends(get_current_user),
) -> MarketDataSubscriptionResponse:
    subscriber_id = str(user.id)
    subscriber_symbols = list(market_data_runtime.subscriber_symbols(subscriber_id))
    return MarketDataSubscriptionResponse(
        subscriber_id=subscriber_id,
        added_symbols=[],
        removed_symbols=[],
        active_symbols=subscriber_symbols,
    )


@router.post("/subscriptions", response_model=MarketDataSubscriptionResponse)
async def subscribe_symbols(
    payload: MarketDataSubscriptionRequest,
    market: str = Query(default="stocks"),
    user: User = Depends(get_current_user),
) -> MarketDataSubscriptionResponse:
    normalized_market = _normalize_market(market)
    subscriber_id = str(user.id)
    delta = market_data_runtime.subscribe(
        subscriber_id,
        payload.symbols,
        market=normalized_market,
    )
    for symbol in delta.added_symbols:
        enqueue_market_data_refresh(market=normalized_market, symbol=symbol)
    return MarketDataSubscriptionResponse(
        subscriber_id=subscriber_id,
        added_symbols=list(delta.added_symbols),
        removed_symbols=list(delta.removed_symbols),
        active_symbols=list(market_data_runtime.subscriber_symbols(subscriber_id)),
    )


@router.delete("/subscriptions", response_model=MarketDataSubscriptionResponse)
async def unsubscribe_symbols(
    user: User = Depends(get_current_user),
) -> MarketDataSubscriptionResponse:
    subscriber_id = str(user.id)
    delta = market_data_runtime.unsubscribe(subscriber_id)
    return MarketDataSubscriptionResponse(
        subscriber_id=subscriber_id,
        added_symbols=list(delta.added_symbols),
        removed_symbols=list(delta.removed_symbols),
        active_symbols=list(market_data_runtime.subscriber_symbols(subscriber_id)),
    )


@router.get("/checkpoints")
async def get_runtime_checkpoints(
    user: User = Depends(get_current_user),
) -> dict[str, object]:
    _ = user
    return {
        "updated_at": datetime.now(UTC).isoformat(),
        "checkpoints": market_data_runtime.checkpoints(),
    }


@router.get("/health")
async def get_market_data_health(
    window_minutes: int = Query(default=60, ge=1, le=1440),
    max_events: int = Query(default=20, ge=1, le=200),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    _ = user
    now = datetime.now(UTC)
    since = now - timedelta(minutes=window_minutes)

    grouped = await db.execute(
        select(
            MarketDataErrorEvent.error_type,
            MarketDataErrorEvent.http_status,
            func.count(MarketDataErrorEvent.id),
        )
        .where(MarketDataErrorEvent.occurred_at >= since)
        .group_by(MarketDataErrorEvent.error_type, MarketDataErrorEvent.http_status)
        .order_by(func.count(MarketDataErrorEvent.id).desc())
    )
    latest_rows = (
        await db.scalars(
            select(MarketDataErrorEvent)
            .where(MarketDataErrorEvent.occurred_at >= since)
            .order_by(MarketDataErrorEvent.occurred_at.desc())
            .limit(max_events),
        )
    ).all()

    active_instruments = market_data_runtime.active_subscriptions()
    symbol_lags = []
    now_ms = int(now.timestamp() * 1000)
    for market, symbol in active_instruments:
        checkpoint = market_data_runtime.get_checkpoint(market=market, symbol=symbol, timeframe="1m")
        if checkpoint is None:
            symbol_lags.append(
                {
                    "market": market,
                    "symbol": symbol,
                    "lag_seconds": None,
                    "checkpoint_ms": None,
                }
            )
            continue
        lag_seconds = max(0.0, (now_ms - int(checkpoint)) / 1000.0)
        symbol_lags.append(
            {
                "market": market,
                "symbol": symbol,
                "lag_seconds": round(lag_seconds, 3),
                "checkpoint_ms": int(checkpoint),
            }
        )

    return {
        "updated_at": now.isoformat(),
        "window_minutes": window_minutes,
        "redis_data_plane": market_data_runtime.redis_data_plane_status(),
        "runtime_metrics": market_data_runtime.runtime_metrics(),
        "refresh_scheduler_metrics": market_data_runtime.refresh_scheduler_metrics(),
        "active_subscriptions": [
            {"market": market, "symbol": symbol}
            for market, symbol in active_instruments
        ],
        "symbol_lag_seconds": symbol_lags,
        "error_summary": [
            {
                "error_type": str(error_type),
                "http_status": int(http_status) if http_status is not None else None,
                "count": int(count),
            }
            for error_type, http_status, count in grouped.all()
        ],
        "recent_errors": [
            {
                "id": str(row.id),
                "market": row.market,
                "symbol": row.symbol,
                "error_type": row.error_type,
                "http_status": row.http_status,
                "endpoint": row.endpoint,
                "message": row.message,
                "occurred_at": row.occurred_at.isoformat(),
                "metadata": row.metadata_ if isinstance(row.metadata_, dict) else {},
            }
            for row in latest_rows
        ],
    }
