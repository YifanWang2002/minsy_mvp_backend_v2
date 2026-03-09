"""WebSocket endpoint for real-time portfolio streaming (Web platform)."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect, status

from apps.api.services.trading_queue_service import enqueue_market_data_refresh
from apps.api.services.trading_stream_replay_service import (
    load_owned_deployment,
    poll_outbox_events,
    resolve_replay_cursor,
)
from packages.domain.market_data.bar_event_pubsub import (
    bar_event_channel,
    decode_bar_realtime_event,
)
from packages.domain.market_data.refresh_dedupe import reserve_market_data_refresh_slot
from packages.domain.market_data.runtime import RuntimeBar, market_data_runtime
from packages.domain.trading.services.realtime_transport_policy import (
    RealtimeTransportPolicy,
    RealtimeTransportState,
)
from packages.domain.trading.services.trading_event_pubsub import (
    decode_realtime_event,
    trading_event_channel,
)
from packages.domain.user.services.auth_service import AuthService
from packages.infra.db.models.user import User
from packages.infra.db.session import get_db_session
from packages.infra.observability.logger import logger
from packages.infra.providers.market_data.alpaca_rest import AlpacaRestProvider
from packages.infra.redis.client import get_redis_client
from packages.shared_settings.schema.settings import settings

from . import market_data as market_data_route

router = APIRouter(prefix="/ws", tags=["trading-ws"])

# Backward-compatible aliases for existing tests/patch targets.
_load_owned_deployment = load_owned_deployment
_poll_outbox_events = poll_outbox_events

_HEARTBEAT_INTERVAL = 15.0
_BAR_DEFAULT_LIMIT = 600
_BAR_WARMUP_WAIT_INTERVAL = 0.25
_BAR_WARMUP_MAX_WAIT_SECONDS = 3.0
_PUBSUB_WAIT_SECONDS = max(0.05, float(settings.trading_ws_pubsub_wait_seconds))
_FALLBACK_POLL_SECONDS = max(0.1, float(settings.trading_ws_fallback_poll_seconds))
_RECONCILE_SECONDS = max(0.2, float(settings.trading_ws_reconcile_seconds))
_PUBSUB_PROBE_BASE_SECONDS = max(0.2, float(settings.trading_ws_pubsub_probe_base_seconds))
_PUBSUB_PROBE_MAX_SECONDS = max(
    _PUBSUB_PROBE_BASE_SECONDS,
    float(settings.trading_ws_pubsub_probe_max_seconds),
)


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
        try:
            enqueue_market_data_refresh(
                market=market,
                symbol=symbol,
                requested_timeframe=timeframe,
                min_bars=limit,
            )
        except Exception as exc:
            logger.warning(
                "WS bars refresh enqueue failed market=%s symbol=%s error=%s",
                market,
                symbol,
                type(exc).__name__,
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
    db: object,
    deployment_id: str,
    requested_cursor: int | None,
) -> int:
    resolved = await resolve_replay_cursor(
        db,
        deployment_id=deployment_id,
        requested_cursor=requested_cursor,
    )
    return int(resolved.cursor)


@dataclass(slots=True)
class ConnectionRuntimeState:
    """Mutable state for one active WS connection."""

    transport: RealtimeTransportState
    bar_last_payload_signature: dict[tuple[str, str, str], str]


def _bars_payload_signature(bars: list[dict]) -> str:
    """Stable signature to suppress duplicate bar_update payloads."""
    return json.dumps(
        bars,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


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
        {"event": "subscribed", "deployment_ids": ["d1"], "cursors": {"d1": 105}, "transport_mode": "pubsub"}
        {"event": "unsubscribed", "deployment_ids": ["d1"]}
        {"event": "transport_mode", "mode": "pubsub|polling_fallback", "reason": "...", "server_time": "..."}
        {"event": "<event_type>", "deployment_id": "d1", "event_seq": 106, "payload": {...}}
        {"event": "bar_snapshot", "market": "stocks", "symbol": "AAPL", "timeframe": "1m", "bars": [...]}
        {"event": "bar_update", "market": "stocks", "symbol": "AAPL", "timeframe": "1m", "bars": [...]}  # realtime pub/sub
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
    deployment_channels: dict[str, str] = {}
    # (market, symbol, timeframe) → last_sent_ts_seconds
    bar_subscriptions: dict[tuple[str, str, str], int] = {}
    bar_runtime_subscribers: dict[tuple[str, str, str], str] = {}
    bar_pubsub_channel_refs: dict[str, int] = {}
    running = True

    try:
        redis_client = get_redis_client()
    except RuntimeError:
        redis_client = None
    pubsub = (
        redis_client.pubsub(ignore_subscribe_messages=True)
        if redis_client is not None
        else None
    )
    pubsub_enabled = pubsub is not None
    transport_policy = RealtimeTransportPolicy(
        fallback_poll_seconds=_FALLBACK_POLL_SECONDS,
        reconcile_seconds=_RECONCILE_SECONDS,
        pubsub_probe_base_seconds=_PUBSUB_PROBE_BASE_SECONDS,
        pubsub_probe_max_seconds=_PUBSUB_PROBE_MAX_SECONDS,
    )
    runtime_state = ConnectionRuntimeState(
        transport=transport_policy.initial_state(
            pubsub_available=pubsub_enabled,
            now=time.monotonic(),
        ),
        bar_last_payload_signature={},
    )

    async def _send_json(data: dict) -> None:
        try:
            await websocket.send_text(json.dumps(data, ensure_ascii=True))
        except Exception:
            nonlocal running
            running = False

    async def _maybe_send_heartbeat(last_heartbeat: float) -> float:
        now = time.monotonic()
        if now - last_heartbeat < _HEARTBEAT_INTERVAL:
            return last_heartbeat
        await _send_json(
            {
                "event": "heartbeat",
                "server_time": datetime.now(UTC).isoformat(),
                "subscriptions": list(subscriptions.keys()),
            }
        )
        return now

    async def _emit_transport_mode(*, mode: str, reason: str) -> None:
        await _send_json(
            {
                "event": "transport_mode",
                "mode": mode,
                "reason": reason,
                "server_time": datetime.now(UTC).isoformat(),
            }
        )

    async def _subscribe_pubsub_channel(channel: str) -> bool:
        if not pubsub_enabled or pubsub is None:
            return False
        try:
            await pubsub.subscribe(channel)
            return True
        except Exception as exc:
            logger.debug("WS pubsub subscribe failed channel=%s error=%s", channel, type(exc).__name__)
            return False

    async def _unsubscribe_pubsub_channel(channel: str) -> None:
        if not pubsub_enabled or pubsub is None:
            return
        try:
            await pubsub.unsubscribe(channel)
        except Exception as exc:
            logger.debug("WS pubsub unsubscribe failed channel=%s error=%s", channel, type(exc).__name__)

    async def _forward_deployment_event(
        *,
        deployment_id: str,
        event_type: str,
        event_seq: int,
        payload: dict,
    ) -> None:
        current = subscriptions.get(deployment_id)
        if current is None or event_seq <= current:
            return
        payload_dict = dict(payload) if isinstance(payload, dict) else {}
        payload_dict.setdefault("deployment_id", deployment_id)
        payload_dict.setdefault("event_seq", event_seq)
        await _send_json(
            {
                "event": event_type,
                "deployment_id": deployment_id,
                "event_seq": event_seq,
                "payload": payload_dict,
            }
        )
        latest = subscriptions.get(deployment_id)
        if latest is not None:
            subscriptions[deployment_id] = max(latest, event_seq)

    async def _forward_outbox_row(deployment_id: str, row: object) -> None:
        payload = row.payload if isinstance(row.payload, dict) else {}
        try:
            seq = int(row.event_seq)
        except (TypeError, ValueError):
            return
        await _forward_deployment_event(
            deployment_id=deployment_id,
            event_type=str(row.event_type),
            event_seq=seq,
            payload=payload,
        )

    def _unsubscribe_bar_key(key: tuple[str, str, str]) -> None:
        subscriber_id = bar_runtime_subscribers.pop(key, None)
        if subscriber_id is None:
            return
        market_data_runtime.unsubscribe(subscriber_id)

    async def _replay_deployment_events(
        db: object,
        *,
        deployment_id: str,
        cursor: int,
        limit: int = 300,
    ) -> None:
        replay_cursor = cursor
        while running:
            rows = await _poll_outbox_events(
                db,
                deployment_id=deployment_id,
                cursor=replay_cursor,
                limit=limit,
            )
            if not rows:
                break
            for row in rows:
                await _forward_outbox_row(deployment_id, row)
                try:
                    replay_cursor = max(replay_cursor, int(row.event_seq))
                except (TypeError, ValueError):
                    continue
            if len(rows) < limit:
                break
        current = subscriptions.get(deployment_id)
        if current is not None:
            subscriptions[deployment_id] = max(current, replay_cursor)

    async def _poll_outbox_once(*, limit: int) -> None:
        if not subscriptions:
            return
        async for db in get_db_session():
            try:
                for deployment_id, cursor in list(subscriptions.items()):
                    if deployment_id not in subscriptions:
                        continue
                    await _replay_deployment_events(
                        db,
                        deployment_id=deployment_id,
                        cursor=cursor,
                        limit=limit,
                    )
            finally:
                await db.rollback()

    async def _switch_to_fallback(reason: str) -> None:
        changed = transport_policy.record_pubsub_failure(
            runtime_state.transport,
            now=time.monotonic(),
            error=reason,
        )
        if changed:
            await _emit_transport_mode(mode="polling_fallback", reason=reason)

    async def _switch_to_pubsub(reason: str) -> None:
        changed = transport_policy.record_pubsub_success(
            runtime_state.transport,
            now=time.monotonic(),
        )
        if changed:
            await _emit_transport_mode(mode="pubsub", reason=reason)

    async def _handle_subscribe(
        msg: dict,
        db: object,
    ) -> None:
        deployment_ids = msg.get("deployment_ids")
        if not isinstance(deployment_ids, list):
            await _send_json({"event": "error", "message": "deployment_ids must be a list"})
            return
        cursors_raw = msg.get("cursors") or {}
        cursors_map = cursors_raw if isinstance(cursors_raw, dict) else {}
        confirmed_ids: list[str] = []
        confirmed_cursors: dict[str, int] = {}
        try:
            for did in deployment_ids:
                did_str = str(did).strip()
                if not did_str:
                    continue
                try:
                    await _load_owned_deployment(db, deployment_id=did_str, user_id=user_id)
                except Exception:
                    await _send_json(
                        {
                            "event": "error",
                            "message": f"deployment {did_str} not found or not owned",
                        }
                    )
                    continue
                requested = cursors_map.get(did_str)
                cursor_val = None
                if isinstance(requested, int):
                    cursor_val = requested if requested > 0 else None
                elif isinstance(requested, str):
                    try:
                        parsed_cursor = int(requested)
                        cursor_val = parsed_cursor if parsed_cursor > 0 else None
                    except ValueError:
                        cursor_val = None
                resolved = await _resolve_cursor(
                    db,
                    did_str,
                    cursor_val,
                )
                subscriptions[did_str] = resolved
                confirmed_ids.append(did_str)
                confirmed_cursors[did_str] = resolved
                if pubsub_enabled and did_str not in deployment_channels:
                    channel = trading_event_channel(did_str)
                    deployment_channels[did_str] = channel
                    subscribed = await _subscribe_pubsub_channel(channel)
                    if not subscribed:
                        await _switch_to_fallback("pubsub_subscribe_failed")
            if confirmed_ids:
                await _send_json(
                    {
                        "event": "subscribed",
                        "deployment_ids": confirmed_ids,
                        "cursors": confirmed_cursors,
                        "transport_mode": runtime_state.transport.mode,
                    }
                )
                for deployment_id in confirmed_ids:
                    await _replay_deployment_events(
                        db,
                        deployment_id=deployment_id,
                        cursor=confirmed_cursors[deployment_id],
                        limit=300,
                    )
        finally:
            await db.rollback()

    async def _handle_unsubscribe(msg: dict) -> None:
        deployment_ids = msg.get("deployment_ids")
        if not isinstance(deployment_ids, list):
            await _send_json({"event": "error", "message": "deployment_ids must be a list"})
            return
        removed: list[str] = []
        for did in deployment_ids:
            did_str = str(did).strip()
            if did_str not in subscriptions:
                continue
            del subscriptions[did_str]
            removed.append(did_str)
            channel = deployment_channels.pop(did_str, None)
            if channel is not None:
                await _unsubscribe_pubsub_channel(channel)
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
        is_new_key = key not in bar_subscriptions
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
        await _send_json(
            {
                "event": "bar_snapshot",
                "market": market,
                "symbol": symbol,
                "timeframe": timeframe,
                "bars": bar_dicts,
            }
        )
        last_ts = max((_bar_ts_seconds(b) for b in bars), default=0)
        bar_subscriptions[key] = last_ts
        if bar_dicts:
            runtime_state.bar_last_payload_signature[key] = _bars_payload_signature(bar_dicts)
        if pubsub_enabled and is_new_key:
            channel = bar_event_channel(market=market, symbol=symbol)
            next_ref = bar_pubsub_channel_refs.get(channel, 0) + 1
            bar_pubsub_channel_refs[channel] = next_ref
            if next_ref == 1:
                subscribed = await _subscribe_pubsub_channel(channel)
                if not subscribed:
                    await _switch_to_fallback("pubsub_subscribe_failed")
        await _send_json(
            {
                "event": "bars_subscribed",
                "market": market,
                "symbol": symbol,
                "timeframe": timeframe,
            }
        )

    async def _handle_unsubscribe_bars(msg: dict) -> None:
        market = str(msg.get("market", "stocks")).strip().lower() or "stocks"
        symbol = str(msg.get("symbol", "")).strip().upper()
        timeframe = str(msg.get("timeframe", "1m")).strip().lower() or "1m"
        key = (market, symbol, timeframe)
        removed = key in bar_subscriptions
        bar_subscriptions.pop(key, None)
        runtime_state.bar_last_payload_signature.pop(key, None)
        _unsubscribe_bar_key(key)
        if pubsub_enabled and removed:
            channel = bar_event_channel(market=market, symbol=symbol)
            current_ref = bar_pubsub_channel_refs.get(channel, 0)
            if current_ref <= 1:
                bar_pubsub_channel_refs.pop(channel, None)
                await _unsubscribe_pubsub_channel(channel)
            else:
                bar_pubsub_channel_refs[channel] = current_ref - 1
        await _send_json(
            {
                "event": "bars_unsubscribed",
                "market": market,
                "symbol": symbol,
                "timeframe": timeframe,
            }
        )

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
                    await _send_json(
                        {
                            "event": "pong",
                            "server_time": datetime.now(UTC).isoformat(),
                        }
                    )
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

    async def _process_pubsub_message(message: object) -> None:
        if not isinstance(message, dict):
            return
        data = message.get("data")
        realtime_event = decode_realtime_event(data)
        if realtime_event is not None:
            await _forward_deployment_event(
                deployment_id=realtime_event.deployment_id,
                event_type=realtime_event.event_type,
                event_seq=realtime_event.event_seq,
                payload=realtime_event.payload,
            )
            return

        bar_payload = decode_bar_realtime_event(data)
        if bar_payload is None:
            return
        key = (
            str(bar_payload.get("market", "")).strip().lower(),
            str(bar_payload.get("symbol", "")).strip().upper(),
            str(bar_payload.get("timeframe", "")).strip().lower(),
        )
        last_ts = bar_subscriptions.get(key)
        if last_ts is None:
            return
        raw_bars = bar_payload.get("bars")
        if not isinstance(raw_bars, list):
            return
        next_bars: list[dict] = []
        latest_ts = last_ts
        for item in raw_bars:
            if not isinstance(item, dict):
                continue
            raw_ts = item.get("t")
            try:
                ts = int(raw_ts)
            except (TypeError, ValueError):
                continue
            if ts < last_ts:
                continue
            next_bars.append(dict(item))
            latest_ts = max(latest_ts, ts)
        if not next_bars:
            return
        signature = _bars_payload_signature(next_bars)
        previous_signature = runtime_state.bar_last_payload_signature.get(key)
        if previous_signature == signature and latest_ts == last_ts:
            return
        await _send_json(
            {
                "event": "bar_update",
                "market": key[0],
                "symbol": key[1],
                "timeframe": key[2],
                "bars": next_bars,
            }
        )
        bar_subscriptions[key] = latest_ts
        runtime_state.bar_last_payload_signature[key] = signature

    async def _push_transport_events() -> None:
        nonlocal running
        last_heartbeat = time.monotonic()
        while running:
            now = time.monotonic()
            did_poll = False

            if runtime_state.transport.mode == "pubsub":
                if pubsub_enabled and pubsub is not None:
                    try:
                        message = await pubsub.get_message(
                            ignore_subscribe_messages=True,
                            timeout=_PUBSUB_WAIT_SECONDS,
                        )
                        await _process_pubsub_message(message)
                    except Exception as exc:
                        await _switch_to_fallback(f"pubsub_read_error:{type(exc).__name__}")
                else:
                    await _switch_to_fallback("pubsub_unavailable")
            else:
                if transport_policy.should_poll_fallback(runtime_state.transport, now=now):
                    await _poll_outbox_once(limit=120)
                    transport_policy.mark_fallback_poll(runtime_state.transport, now=time.monotonic())
                    did_poll = True
                if (
                    pubsub_enabled
                    and pubsub is not None
                    and transport_policy.should_probe_pubsub(runtime_state.transport, now=now)
                ):
                    try:
                        probe_message = await pubsub.get_message(
                            ignore_subscribe_messages=True,
                            timeout=0.05,
                        )
                        await _switch_to_pubsub("pubsub_recovered")
                        await _process_pubsub_message(probe_message)
                    except Exception as exc:
                        transport_policy.record_pubsub_failure(
                            runtime_state.transport,
                            now=time.monotonic(),
                            error=f"pubsub_probe_error:{type(exc).__name__}",
                        )

            if subscriptions and transport_policy.should_reconcile(runtime_state.transport, now=time.monotonic()):
                await _poll_outbox_once(limit=300)
                transport_policy.mark_reconcile(runtime_state.transport, now=time.monotonic())
                did_poll = True

            last_heartbeat = await _maybe_send_heartbeat(last_heartbeat)
            if runtime_state.transport.mode == "polling_fallback" and not did_poll:
                await asyncio.sleep(0.05)

    reader_task = asyncio.create_task(_read_client_messages())
    pusher_task = asyncio.create_task(_push_transport_events())

    try:
        done, pending = await asyncio.wait(
            [reader_task, pusher_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        running = False
        for task in pending:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        for task in done:
            if task.cancelled():
                continue
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
    except Exception:
        running = False
        for task in (reader_task, pusher_task):
            task.cancel()
    finally:
        for key in list(bar_runtime_subscribers):
            _unsubscribe_bar_key(key)
        if pubsub is not None:
            try:
                await pubsub.aclose()
            except Exception:
                pass
        try:
            await websocket.close()
        except Exception:
            pass
