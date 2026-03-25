"""Redis pub/sub helpers for chart annotation realtime events."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from packages.infra.redis.client import get_redis_client

_ANNOTATION_CHANNEL_PREFIX = "chart_annotations:user:"


@dataclass(frozen=True, slots=True)
class ChartAnnotationRealtimeEvent:
    """Wire payload used for chart annotation realtime delivery."""

    owner_user_id: str
    event_type: str
    event_seq: int
    payload: dict[str, Any]


def chart_annotation_channel(owner_user_id: str) -> str:
    """Return Redis channel name for a user's annotation stream."""
    return f"{_ANNOTATION_CHANNEL_PREFIX}{owner_user_id.strip()}"


def encode_chart_annotation_event(event: ChartAnnotationRealtimeEvent) -> str:
    """Serialize an annotation event for Redis transport."""
    return json.dumps(
        {
            "owner_user_id": event.owner_user_id,
            "event": event.event_type,
            "event_seq": int(event.event_seq),
            "payload": event.payload if isinstance(event.payload, dict) else {},
        },
        ensure_ascii=True,
        separators=(",", ":"),
    )


def decode_chart_annotation_event(raw: object) -> ChartAnnotationRealtimeEvent | None:
    """Parse pub/sub payload into a typed annotation event."""
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
    owner_user_id = str(parsed.get("owner_user_id") or "").strip()
    event_type = str(parsed.get("event") or "").strip()
    payload = parsed.get("payload")
    try:
        event_seq = int(parsed.get("event_seq"))
    except (TypeError, ValueError):
        return None
    if not owner_user_id or not event_type or not isinstance(payload, dict) or event_seq <= 0:
        return None
    return ChartAnnotationRealtimeEvent(
        owner_user_id=owner_user_id,
        event_type=event_type,
        event_seq=event_seq,
        payload=payload,
    )


async def publish_chart_annotation_event(event: ChartAnnotationRealtimeEvent) -> bool:
    """Publish a realtime annotation event if Redis is available."""
    try:
        redis = get_redis_client()
    except RuntimeError:
        return False
    await redis.publish(
        chart_annotation_channel(event.owner_user_id),
        encode_chart_annotation_event(event),
    )
    return True
