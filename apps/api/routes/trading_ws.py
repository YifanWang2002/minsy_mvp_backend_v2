"""WebSocket endpoint for real-time portfolio streaming (Web platform)."""

from __future__ import annotations

import asyncio
import json
import time
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.services.trading_queue_service import enqueue_market_data_refresh
from packages.domain.market_data.refresh_dedupe import reserve_market_data_refresh_slot
from packages.domain.market_data.runtime import RuntimeBar, market_data_runtime
from packages.domain.user.services.auth_service import AuthService
from packages.infra.db.models.trading_event_outbox import TradingEventOutbox
from packages.infra.db.models.user import User
from packages.infra.db.session import get_db_session
from packages.infra.observability.logger import logger
from packages.infra.providers.market_data.alpaca_rest import AlpacaRestProvider

from . import market_data as market_data_route
from .trading_stream import _load_owned_deployment, _poll_outbox_events

router = APIRouter(prefix="/ws", tags=["trading-ws"])

_POLL_INTERVAL = 1.0
_HEARTBEAT_INTERVAL = 15.0
_BAR_PUSH_INTERVAL = 2.0
_BAR_DEFAULT_LIMIT = 600
_BAR_PUSH_LIMIT = 5
_BAR_WARMUP_WAIT_INTERVAL = 0.25
_BAR_WARMUP_MAX_WAIT_SECONDS = 3.0


def _bar_to_dict(bar: RuntimeBar) -> dict:
    """Convert a RuntimeBar to a compact dict for WS transmission."""
    return {
        "t": int(bar.timestamp.timestamp()),
        "o": bar.open,
        "h": bar.high,
        "l": bar.low,
        "c": bar.close,
        "v": bar.volume,
    }


def _bar_ts_seconds(bar: RuntimeBar) -> int:
    return int(bar.timestamp.timestamp())


def _timeframe_step(value: str) -> timedelta | None:
    normalized = value.strip().lower()
    if not normalized:
        return None
    if normalized.endswith("m"):
        try:
            minutes = int(normalized[:-1] or "0")
        except ValueError:
            return None
        if minutes <= 0:
            return None
        return timedelta(minutes=minutes)
    if normalized.endswith("h"):
        try:
            hours = int(normalized[:-1] or "0")
        except ValueError:
            return None
        if hours <= 0:
            return None
        return timedelta(hours=hours)
    if normalized.endswith("d"):
        try:
            days = int(normalized[:-1] or "0")
        except ValueError:
            return None
        if days <= 0:
            return None
        return timedelta(days=days)
    return None


async def _hydrate_bar_snapshot_from_provider(
    *,
    market: str,
    symbol: str,
    timeframe: str,
    limit: int,
) -> list[RuntimeBar]:
    step = _timeframe_step(timeframe)
    if step is None:
        return []

    if market == "stocks":
        lookback = max(step * max(limit * 5, 1), timedelta(days=3))
    else:
        lookback = max(step * max(limit * 2, 1), timedelta(hours=6))
    since = datetime.now(UTC) - lookback

    provider = AlpacaRestProvider()
    try:
        rows = await provider.fetch_recent_bars(
            symbol=symbol,
            market=market,
            timeframe=timeframe,
            since=since,
            limit=limit,
        )
    except Exception:
        return []
    finally:
        await provider.aclose()

    if not rows:
        return []

    market_data_runtime.hydrate_bars(
        market=market,
        symbol=symbol,
        timeframe=timeframe,
        bars=rows,
    )
    return await asyncio.to_thread(
        market_data_runtime.get_recent_bars,
        market=market,
        symbol=symbol,
        timeframe=timeframe,
        limit=limit,
    )


async def _finalize_bar_window(
    *,
    market: str,
    symbol: str,
    timeframe: str,
    limit: int,
    bars: list[RuntimeBar],
) -> list[RuntimeBar]:
    live_bar = await market_data_route._build_live_bar(
        market=market,
        symbol=symbol,
        timeframe=timeframe,
        historical_bars=bars,
    )
    return market_data_route._merge_live_bar(
        bars=bars,
        live_bar=live_bar,
        limit=limit,
    )


async def _read_bar_window(
    *,
    market: str,
    symbol: str,
    timeframe: str,
    limit: int,
) -> list[RuntimeBar]:
    bars = await asyncio.to_thread(
        market_data_runtime.get_recent_bars,
        market=market,
        symbol=symbol,
        timeframe=timeframe,
        limit=limit,
    )
    return await _finalize_bar_window(
        market=market,
        symbol=symbol,
        timeframe=timeframe,
        limit=limit,
        bars=bars,
    )


def _bars_ready_for_snapshot(
    *,
    market: str,
    timeframe: str,
    bars: list[RuntimeBar],
    limit: int,
) -> bool:
    if len(bars) < limit:
        return False
    return not market_data_route._bars_are_stale(
        market=market,
        timeframe=timeframe,
        bars=bars,
    )


def _prefer_bar_window(
    current: list[RuntimeBar],
    candidate: list[RuntimeBar],
) -> list[RuntimeBar]:
    if not current:
        return candidate
    if not candidate:
        return current
    current_latest = current[-1].timestamp
    candidate_latest = candidate[-1].timestamp
    if candidate_latest > current_latest:
        return candidate
    if candidate_latest < current_latest:
        return current
    if len(candidate) >= len(current):
        return candidate
    return current


def _bar_subscription_subscriber_id(
    *,
    connection_id: str,
    market: str,
    symbol: str,
    timeframe: str,
) -> str:
    return f"ws-bars:{connection_id}:{market}:{symbol}:{timeframe}"


async def _load_bar_snapshot(
    *,
    market: str,
    symbol: str,
    timeframe: str,
    limit: int,
) -> list[RuntimeBar]:
    bars = await _read_bar_window(
        market=market,
        symbol=symbol,
        timeframe=timeframe,
        limit=limit,
    )
    if _bars_ready_for_snapshot(
        market=market,
        timeframe=timeframe,
        bars=bars,
        limit=limit,
    ):
        return bars

    if reserve_market_data_refresh_slot(market, symbol):
        enqueue_market_data_refresh(
            market=market,
            symbol=symbol,
            requested_timeframe=timeframe,
            min_bars=limit,
        )
        hydrated = await _hydrate_bar_snapshot_from_provider(
            market=market,
            symbol=symbol,
            timeframe=timeframe,
            limit=limit,
        )
        hydrated = await _finalize_bar_window(
            market=market,
            symbol=symbol,
            timeframe=timeframe,
            limit=limit,
            bars=hydrated,
        )
        bars = _prefer_bar_window(bars, hydrated)
        if _bars_ready_for_snapshot(
            market=market,
            timeframe=timeframe,
            bars=bars,
            limit=limit,
        ):
            return bars

    best = bars
    deadline = time.monotonic() + _BAR_WARMUP_MAX_WAIT_SECONDS
    while time.monotonic() < deadline:
        await asyncio.sleep(_BAR_WARMUP_WAIT_INTERVAL)
        candidate = await _read_bar_window(
            market=market,
            symbol=symbol,
            timeframe=timeframe,
            limit=limit,
        )
        best = _prefer_bar_window(best, candidate)
        if _bars_ready_for_snapshot(
            market=market,
            timeframe=timeframe,
            bars=candidate,
            limit=limit,
        ):
            return candidate
    return best


async def _authenticate_ws(token: str | None) -> User | None:
    """Validate a bearer token from the WebSocket query string."""
    if not token or not token.strip():
        return None
    async for db in get_db_session():
        try:
            service = AuthService(db)
            return await service.get_current_user(token.strip())
        except Exception:
            return None
    return None


async def _resolve_cursor(
    db: AsyncSession,
    deployment_id: str,
    requested_cursor: int | None,
) -> int:
    """Resolve the starting cursor for a deployment subscription."""
    if requested_cursor is not None:
        return requested_cursor
    latest = await db.scalar(
        select(TradingEventOutbox.event_seq)
        .where(TradingEventOutbox.deployment_id == deployment_id)
        .order_by(TradingEventOutbox.event_seq.desc())
        .limit(1)
    )
    return int(latest or 0)


@router.websocket("/trading")
async def ws_trading(
    websocket: WebSocket,
    token: str | None = Query(default=None),
) -> None:
    """Multiplexed WebSocket for portfolio event streaming.

    Protocol
    --------
    Client → Server:
        {"action": "subscribe", "deployment_ids": ["d1"], "cursors": {"d1": 105}}
        {"action": "unsubscribe", "deployment_ids": ["d1"]}
        {"action": "subscribe_bars", "market": "stocks", "symbol": "AAPL", "timeframe": "1m", "limit": 600}
        {"action": "unsubscribe_bars", "market": "stocks", "symbol": "AAPL", "timeframe": "1m"}
        {"action": "ping"}

    Server → Client:
        {"event": "subscribed", "deployment_ids": ["d1"], "cursors": {"d1": 105}}
        {"event": "unsubscribed", "deployment_ids": ["d1"]}
        {"event": "<event_type>", "deployment_id": "d1", "event_seq": 106, "payload": {...}}
        {"event": "bar_snapshot", "market": "stocks", "symbol": "AAPL", "timeframe": "1m", "bars": [...]}
        {"event": "bar_update", "market": "stocks", "symbol": "AAPL", "timeframe": "1m", "bars": [...]}
        {"event": "bars_subscribed", "market": "stocks", "symbol": "AAPL", "timeframe": "1m"}
        {"event": "bars_unsubscribed", "market": "stocks", "symbol": "AAPL", "timeframe": "1m"}
        {"event": "pong", "server_time": "..."}
        {"event": "heartbeat", "server_time": "...", "subscriptions": ["d1"]}
        {"event": "error", "message": "..."}
    """
    user = await _authenticate_ws(token)
    if user is None:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="auth_failed")
        return

    await websocket.accept()
    user_id = str(user.id)
    connection_id = uuid4().hex

    # deployment_id → cursor
    subscriptions: dict[str, int] = {}
    # (market, symbol, timeframe) → last_sent_ts_seconds
    bar_subscriptions: dict[tuple[str, str, str], int] = {}
    bar_runtime_subscribers: dict[tuple[str, str, str], str] = {}
    running = True

    async def _send_json(data: dict) -> None:
        try:
            await websocket.send_text(json.dumps(data, ensure_ascii=True))
        except Exception:
            nonlocal running
            running = False

    def _unsubscribe_bar_key(key: tuple[str, str, str]) -> None:
        subscriber_id = bar_runtime_subscribers.pop(key, None)
        if subscriber_id is None:
            return
        market_data_runtime.unsubscribe(subscriber_id)

    async def _handle_subscribe(
        msg: dict,
        db: AsyncSession,
    ) -> None:
        deployment_ids = msg.get("deployment_ids")
        if not isinstance(deployment_ids, list):
            await _send_json({"event": "error", "message": "deployment_ids must be a list"})
            return
        cursors_raw = msg.get("cursors") or {}
        cursors_map = cursors_raw if isinstance(cursors_raw, dict) else {}
        confirmed_ids: list[str] = []
        confirmed_cursors: dict[str, int] = {}
        for did in deployment_ids:
            did_str = str(did).strip()
            if not did_str:
                continue
            try:
                await _load_owned_deployment(db, deployment_id=did_str, user_id=user_id)
            except Exception:
                await _send_json({
                    "event": "error",
                    "message": f"deployment {did_str} not found or not owned",
                })
                continue
            requested = cursors_map.get(did_str)
            cursor_val = None
            if isinstance(requested, int):
                cursor_val = requested
            elif isinstance(requested, str):
                try:
                    cursor_val = int(requested)
                except ValueError:
                    cursor_val = None
            resolved = await _resolve_cursor(db, did_str, cursor_val)
            subscriptions[did_str] = resolved
            confirmed_ids.append(did_str)
            confirmed_cursors[did_str] = resolved
        await db.rollback()
        if confirmed_ids:
            await _send_json({
                "event": "subscribed",
                "deployment_ids": confirmed_ids,
                "cursors": confirmed_cursors,
            })

    async def _handle_unsubscribe(msg: dict) -> None:
        deployment_ids = msg.get("deployment_ids")
        if not isinstance(deployment_ids, list):
            await _send_json({"event": "error", "message": "deployment_ids must be a list"})
            return
        removed: list[str] = []
        for did in deployment_ids:
            did_str = str(did).strip()
            if did_str in subscriptions:
                del subscriptions[did_str]
                removed.append(did_str)
        if removed:
            await _send_json({"event": "unsubscribed", "deployment_ids": removed})

    async def _handle_subscribe_bars(msg: dict) -> None:
        market = str(msg.get("market", "stocks")).strip().lower() or "stocks"
        symbol = str(msg.get("symbol", "")).strip().upper()
        timeframe = str(msg.get("timeframe", "1m")).strip().lower() or "1m"
        limit = msg.get("limit", _BAR_DEFAULT_LIMIT)
        if not isinstance(limit, int) or limit < 1:
            limit = _BAR_DEFAULT_LIMIT
        limit = min(limit, 5000)
        if not symbol:
            await _send_json({"event": "error", "message": "symbol is required for subscribe_bars"})
            return
        key = (market, symbol, timeframe)
        subscriber_id = bar_runtime_subscribers.get(key)
        if subscriber_id is None:
            subscriber_id = _bar_subscription_subscriber_id(
                connection_id=connection_id,
                market=market,
                symbol=symbol,
                timeframe=timeframe,
            )
            bar_runtime_subscribers[key] = subscriber_id
        market_data_runtime.subscribe(
            subscriber_id,
            [symbol],
            market=market,
        )
        bars = await _load_bar_snapshot(
            market=market,
            symbol=symbol,
            timeframe=timeframe,
            limit=limit,
        )
        bar_dicts = [_bar_to_dict(b) for b in bars]
        await _send_json({
            "event": "bar_snapshot",
            "market": market,
            "symbol": symbol,
            "timeframe": timeframe,
            "bars": bar_dicts,
        })
        last_ts = max((_bar_ts_seconds(b) for b in bars), default=0)
        bar_subscriptions[key] = last_ts
        await _send_json({
            "event": "bars_subscribed",
            "market": market,
            "symbol": symbol,
            "timeframe": timeframe,
        })

    async def _handle_unsubscribe_bars(msg: dict) -> None:
        market = str(msg.get("market", "stocks")).strip().lower() or "stocks"
        symbol = str(msg.get("symbol", "")).strip().upper()
        timeframe = str(msg.get("timeframe", "1m")).strip().lower() or "1m"
        key = (market, symbol, timeframe)
        if key in bar_subscriptions:
            del bar_subscriptions[key]
        _unsubscribe_bar_key(key)
        await _send_json({
            "event": "bars_unsubscribed",
            "market": market,
            "symbol": symbol,
            "timeframe": timeframe,
        })

    async def _read_client_messages() -> None:
        """Read and dispatch incoming client messages."""
        nonlocal running
        try:
            while running:
                raw = await websocket.receive_text()
                try:
                    msg = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    await _send_json({"event": "error", "message": "invalid json"})
                    continue
                if not isinstance(msg, dict):
                    await _send_json({"event": "error", "message": "expected json object"})
                    continue
                action = str(msg.get("action", "")).strip().lower()
                if action == "ping":
                    await _send_json({
                        "event": "pong",
                        "server_time": datetime.now(UTC).isoformat(),
                    })
                elif action == "subscribe":
                    async for db in get_db_session():
                        await _handle_subscribe(msg, db)
                elif action == "unsubscribe":
                    await _handle_unsubscribe(msg)
                elif action == "subscribe_bars":
                    await _handle_subscribe_bars(msg)
                elif action == "unsubscribe_bars":
                    await _handle_unsubscribe_bars(msg)
                else:
                    await _send_json({"event": "error", "message": f"unknown action: {action}"})
        except WebSocketDisconnect:
            running = False
        except Exception as exc:
            logger.debug("WS reader error: %s", exc)
            running = False

    async def _push_events() -> None:
        """Poll outbox and push events to the client."""
        nonlocal running
        last_heartbeat = time.monotonic()
        try:
            while running:
                if not subscriptions:
                    await asyncio.sleep(_POLL_INTERVAL)
                    now = time.monotonic()
                    if now - last_heartbeat >= _HEARTBEAT_INTERVAL:
                        await _send_json({
                            "event": "heartbeat",
                            "server_time": datetime.now(UTC).isoformat(),
                            "subscriptions": [],
                        })
                        last_heartbeat = now
                    continue

                any_events = False
                async for db in get_db_session():
                    try:
                        for did, cursor in list(subscriptions.items()):
                            rows = await _poll_outbox_events(
                                db,
                                deployment_id=did,
                                cursor=cursor,
                            )
                            for row in rows:
                                payload = row.payload if isinstance(row.payload, dict) else {}
                                payload_dict = dict(payload)
                                payload_dict.setdefault("deployment_id", did)
                                payload_dict.setdefault("event_seq", row.event_seq)
                                await _send_json({
                                    "event": row.event_type,
                                    "deployment_id": did,
                                    "event_seq": int(row.event_seq),
                                    "payload": payload_dict,
                                })
                                subscriptions[did] = max(cursor, int(row.event_seq))
                                any_events = True
                    finally:
                        await db.rollback()

                now = time.monotonic()
                if not any_events and now - last_heartbeat >= _HEARTBEAT_INTERVAL:
                    await _send_json({
                        "event": "heartbeat",
                        "server_time": datetime.now(UTC).isoformat(),
                        "subscriptions": list(subscriptions.keys()),
                    })
                    last_heartbeat = now

                await asyncio.sleep(_POLL_INTERVAL)
        except Exception as exc:
            logger.debug("WS pusher error: %s", exc)
            running = False

    async def _push_bars() -> None:
        """Poll MarketDataRuntime and push incremental bar updates."""
        nonlocal running
        try:
            while running:
                if not bar_subscriptions:
                    await asyncio.sleep(_BAR_PUSH_INTERVAL)
                    continue
                for key, last_ts in list(bar_subscriptions.items()):
                    if not running:
                        break
                    market, symbol, timeframe = key
                    bars = await _read_bar_window(
                        market=market,
                        symbol=symbol,
                        timeframe=timeframe,
                        limit=_BAR_PUSH_LIMIT,
                    )
                    if (
                        not bars
                        or market_data_route._bars_are_stale(
                            market=market,
                            timeframe=timeframe,
                            bars=bars,
                        )
                    ) and reserve_market_data_refresh_slot(market, symbol):
                        enqueue_market_data_refresh(
                            market=market,
                            symbol=symbol,
                            requested_timeframe=timeframe,
                            min_bars=_BAR_PUSH_LIMIT,
                        )
                    # Send bars with ts >= last_sent (catches updates to current bar)
                    new_bars = [b for b in bars if _bar_ts_seconds(b) >= last_ts]
                    if not new_bars:
                        continue
                    bar_dicts = [_bar_to_dict(b) for b in new_bars]
                    await _send_json({
                        "event": "bar_update",
                        "market": market,
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "bars": bar_dicts,
                    })
                    latest_ts = max(_bar_ts_seconds(b) for b in new_bars)
                    bar_subscriptions[key] = latest_ts
                await asyncio.sleep(_BAR_PUSH_INTERVAL)
        except Exception as exc:
            logger.debug("WS bar pusher error: %s", exc)
            running = False

    reader_task = asyncio.create_task(_read_client_messages())
    pusher_task = asyncio.create_task(_push_events())
    bar_pusher_task = asyncio.create_task(_push_bars())

    try:
        done, pending = await asyncio.wait(
            [reader_task, pusher_task, bar_pusher_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        running = False
        for task in pending:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
    except Exception:
        running = False
        reader_task.cancel()
        pusher_task.cancel()
        bar_pusher_task.cancel()
    finally:
        for key in list(bar_runtime_subscribers):
            _unsubscribe_bar_key(key)
        try:
            await websocket.close()
        except Exception:
            pass
