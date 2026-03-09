"""Redis pub/sub helpers for realtime trading event delivery."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from packages.infra.observability.logger import logger
from packages.infra.redis.client import get_redis_client

_TRADING_EVENT_CHANNEL_PREFIX = "trading_events:deployment:"


@dataclass(frozen=True, slots=True)
class TradingRealtimeEvent:
    """Wire payload used for realtime trading-event pub/sub delivery."""

    deployment_id: str
    event_type: str
    event_seq: int
    payload: dict[str, Any]


def trading_event_channel(deployment_id: str) -> str:
    """Return Redis channel name for a deployment event stream."""
    return f"{_TRADING_EVENT_CHANNEL_PREFIX}{deployment_id.strip()}"


def encode_realtime_event(event: TradingRealtimeEvent) -> str:
    """Serialize an event to compact JSON for Redis transport."""
    payload = event.payload if isinstance(event.payload, dict) else {}
    return json.dumps(
        {
            "deployment_id": event.deployment_id,
            "event": event.event_type,
            "event_seq": int(event.event_seq),
            "payload": payload,
        },
        ensure_ascii=True,
        separators=(",", ":"),
    )


def decode_realtime_event(raw: object) -> TradingRealtimeEvent | None:
    """Parse pub/sub payload into a typed realtime event envelope."""
    text: str
    if isinstance(raw, bytes):
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return None
    elif isinstance(raw, str):
        text = raw
    else:
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    deployment_id = str(parsed.get("deployment_id") or "").strip()
    event_type = str(parsed.get("event") or "").strip()
    payload = parsed.get("payload")
    if not deployment_id or not event_type or not isinstance(payload, dict):
        return None
    raw_seq = parsed.get("event_seq")
    try:
        event_seq = int(raw_seq)
    except (TypeError, ValueError):
        return None
    if event_seq <= 0:
        return None
    return TradingRealtimeEvent(
        deployment_id=deployment_id,
        event_type=event_type,
        event_seq=event_seq,
        payload=payload,
    )


async def publish_realtime_event(event: TradingRealtimeEvent) -> bool:
    """Publish a realtime event to Redis if the shared client is available."""
    try:
        redis = get_redis_client()
    except RuntimeError:
        return False
    try:
        channel = trading_event_channel(event.deployment_id)
        await redis.publish(channel, encode_realtime_event(event))
        return True
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "[trading-event-pubsub] publish failed deployment_id=%s seq=%s error=%s",
            event.deployment_id,
            event.event_seq,
            type(exc).__name__,
        )
        return False
