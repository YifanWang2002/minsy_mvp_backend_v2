"""Redis pub/sub helpers for market bar realtime updates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from packages.infra.observability.logger import logger
from packages.infra.redis.client import get_sync_redis_client

_BAR_EVENT_CHANNEL_PREFIX = "market_bars:"

if TYPE_CHECKING:
    from packages.domain.market_data.runtime import RuntimeBar


@dataclass(frozen=True, slots=True)
class BarRealtimeEvent:
    """Wire payload for one incremental bar update event."""

    market: str
    symbol: str
    timeframe: str
    bars: tuple[dict[str, float | int], ...]


def bar_event_channel(*, market: str, symbol: str) -> str:
    """Return Redis channel for a market+symbol realtime bar stream."""
    normalized_market = market.strip().lower()
    normalized_symbol = symbol.strip().upper()
    return f"{_BAR_EVENT_CHANNEL_PREFIX}{normalized_market}:{normalized_symbol}"


def _runtime_bar_to_wire(bar: RuntimeBar) -> dict[str, float | int]:
    return {
        "t": int(bar.timestamp.astimezone(UTC).timestamp()),
        "o": float(bar.open),
        "h": float(bar.high),
        "l": float(bar.low),
        "c": float(bar.close),
        "v": float(bar.volume),
    }


def build_bar_realtime_event(
    *,
    market: str,
    symbol: str,
    timeframe: str,
    bars: list[RuntimeBar],
) -> BarRealtimeEvent:
    """Build a pub/sub event envelope from runtime bars."""
    normalized_market = market.strip().lower()
    normalized_symbol = symbol.strip().upper()
    normalized_timeframe = timeframe.strip().lower()
    return BarRealtimeEvent(
        market=normalized_market,
        symbol=normalized_symbol,
        timeframe=normalized_timeframe,
        bars=tuple(_runtime_bar_to_wire(item) for item in bars),
    )


def encode_bar_realtime_event(event: BarRealtimeEvent) -> str:
    """Serialize bar event to compact JSON for Redis transport."""
    return json.dumps(
        {
            "event": "bar_update",
            "market": event.market,
            "symbol": event.symbol,
            "timeframe": event.timeframe,
            "bars": list(event.bars),
            "published_at": datetime.now(UTC).isoformat(),
        },
        ensure_ascii=True,
        separators=(",", ":"),
    )


def decode_bar_realtime_event(raw: object) -> dict[str, Any] | None:
    """Decode a pub/sub message into a WS-compatible bar_update payload."""
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
    event = str(parsed.get("event") or "").strip().lower()
    market = str(parsed.get("market") or "").strip().lower()
    symbol = str(parsed.get("symbol") or "").strip().upper()
    timeframe = str(parsed.get("timeframe") or "").strip().lower()
    bars = parsed.get("bars")
    if event != "bar_update" or not market or not symbol or not timeframe:
        return None
    if not isinstance(bars, list):
        return None
    return {
        "event": "bar_update",
        "market": market,
        "symbol": symbol,
        "timeframe": timeframe,
        "bars": bars,
    }


def publish_bar_realtime_event(event: BarRealtimeEvent) -> bool:
    """Publish a bar update event if sync Redis client is available."""
    try:
        redis = get_sync_redis_client()
    except Exception:  # noqa: BLE001
        return False
    try:
        redis.publish(
            bar_event_channel(market=event.market, symbol=event.symbol),
            encode_bar_realtime_event(event),
        )
        return True
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "[bar-event-pubsub] publish failed market=%s symbol=%s timeframe=%s error=%s",
            event.market,
            event.symbol,
            event.timeframe,
            type(exc).__name__,
        )
        return False
