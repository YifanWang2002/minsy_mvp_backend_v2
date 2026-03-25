"""Project trading and backtest facts into canonical annotation documents."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from packages.infra.db.models.fill import Fill
from packages.infra.db.models.order import Order
from packages.infra.db.models.position import Position
from packages.infra.db.models.signal_event import SignalEvent


def _iso_seconds(value: datetime | None) -> int | None:
    if value is None:
        return None
    return int(value.astimezone(UTC).timestamp())


def _base_scope(
    *,
    market: str,
    symbol: str,
    timeframe: str,
    deployment_id: UUID | None = None,
    strategy_id: UUID | None = None,
    backtest_id: UUID | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "market": market,
        "symbol": symbol,
        "timeframe": timeframe,
        "anchor_space": "time_price",
    }
    if deployment_id is not None:
        payload["deployment_id"] = str(deployment_id)
    if strategy_id is not None:
        payload["strategy_id"] = str(strategy_id)
    if backtest_id is not None:
        payload["backtest_id"] = str(backtest_id)
    return payload


def build_execution_annotation_documents(
    *,
    market: str,
    symbol: str,
    timeframe: str,
    deployment_id: UUID,
    signal_events: Iterable[SignalEvent],
    orders: Iterable[Order],
    fills: Iterable[Fill],
    positions: Iterable[Position],
) -> list[dict[str, Any]]:
    """Project execution facts into canonical annotation documents."""
    scope = _base_scope(
        market=market,
        symbol=symbol,
        timeframe=timeframe,
        deployment_id=deployment_id,
    )
    docs: list[dict[str, Any]] = []
    for signal_event in signal_events:
        timestamp = _iso_seconds(signal_event.bar_time)
        if timestamp is None:
            continue
        docs.append(
            {
                "id": f"signal:{signal_event.id}",
                "source": {
                    "type": "strategy_runtime",
                    "source_id": str(signal_event.id),
                },
                "scope": scope,
                "semantic": {
                    "kind": "signal",
                    "role": "execution",
                    "direction": str(signal_event.signal).lower(),
                    "status": "active",
                },
                "tool": {
                    "family": "marker",
                    "vendor": "tradingview",
                    "vendor_type": "arrow_marker",
                },
                "anchors": {
                    "points": [{"time": timestamp}],
                },
                "geometry": {"type": "point"},
                "content": {"text": signal_event.reason},
                "style": {},
                "relations": {},
                "lifecycle": {"editable": False},
                "vendor_native": {},
            }
        )
    fill_by_order_id = {fill.order_id: fill for fill in fills}
    for order in orders:
        if order.symbol != symbol:
            continue
        fill = fill_by_order_id.get(order.id)
        fill_time = _iso_seconds(fill.filled_at if fill is not None else order.submitted_at)
        if fill_time is None:
            continue
        docs.append(
            {
                "id": f"order:{order.id}",
                "source": {
                    "type": "strategy_runtime",
                    "source_id": str(order.id),
                },
                "scope": scope,
                "semantic": {
                    "kind": "entry" if str(order.side).lower() == "buy" else "exit",
                    "role": "execution",
                    "direction": "long" if str(order.side).lower() == "buy" else "short",
                    "status": str(order.status).lower(),
                },
                "tool": {
                    "family": "marker",
                    "vendor": "tradingview",
                    "vendor_type": "arrow_up" if str(order.side).lower() == "buy" else "arrow_down",
                },
                "anchors": {
                    "points": [
                        {
                            "time": fill_time,
                            "price": float(fill.fill_price if fill is not None else order.price or 0),
                        }
                    ],
                },
                "geometry": {"type": "point"},
                "content": {"text": str(order.client_order_id)},
                "style": {},
                "relations": {},
                "lifecycle": {"editable": False},
                "vendor_native": {},
            }
        )
    for position in positions:
        if position.symbol != symbol or str(position.side).lower() == "flat":
            continue
        docs.append(
            {
                "id": f"position:{deployment_id}:{symbol}",
                "source": {
                    "type": "strategy_runtime",
                    "source_id": f"{deployment_id}:{symbol}",
                },
                "scope": scope,
                "semantic": {
                    "kind": "position",
                    "role": "execution",
                    "direction": str(position.side).lower(),
                    "status": "active",
                },
                "tool": {
                    "family": "trading_box",
                    "vendor": "tradingview",
                    "vendor_type": "long_position"
                    if str(position.side).lower() == "long"
                    else "short_position",
                },
                "anchors": {
                    "points": [{"time": int(datetime.now(UTC).timestamp()), "price": float(position.avg_entry_price)}],
                },
                "geometry": {"type": "composite"},
                "content": {},
                "style": {},
                "relations": {},
                "lifecycle": {"editable": False},
                "vendor_native": {},
            }
        )
    return docs


def build_backtest_trade_annotation_documents(
    *,
    market: str,
    symbol: str,
    timeframe: str,
    backtest_id: UUID | str,
    trade: dict[str, Any],
    trade_annotations: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Adapt legacy backtest trade annotations into canonical documents."""
    scope = _base_scope(
        market=market,
        symbol=symbol,
        timeframe=timeframe,
        backtest_id=UUID(str(backtest_id)) if not isinstance(backtest_id, UUID) else backtest_id,
    )
    docs: list[dict[str, Any]] = []
    trade_uid = str(trade.get("trade_uid") or trade.get("uid") or "trade")
    side = str(trade.get("side") or "long").strip().lower()
    entry_price = trade.get("entry_price")
    exit_price = trade.get("exit_price")
    for item in trade_annotations:
        kind = str(item.get("kind") or "").strip().lower()
        if kind in {"trade_entry", "trade_exit", "stop_loss", "take_profit"}:
            raw_time = item.get("time")
            timestamp = None
            if isinstance(raw_time, str):
                try:
                    timestamp = int(datetime.fromisoformat(raw_time.replace("Z", "+00:00")).astimezone(UTC).timestamp())
                except ValueError:
                    timestamp = None
            if timestamp is None:
                continue
            semantic_kind = {
                "trade_entry": "entry",
                "trade_exit": "exit",
                "stop_loss": "stop_loss",
                "take_profit": "take_profit",
            }[kind]
            vendor_type = {
                "trade_entry": "arrow_up" if side != "short" else "arrow_down",
                "trade_exit": "flag",
                "stop_loss": "horizontal_line",
                "take_profit": "horizontal_line",
            }[kind]
            docs.append(
                {
                    "id": f"backtest:{trade_uid}:{kind}",
                    "source": {"type": "backtest", "source_id": trade_uid},
                    "scope": scope,
                    "semantic": {
                        "kind": semantic_kind,
                        "role": "execution",
                        "direction": side,
                        "status": "closed",
                    },
                    "tool": {
                        "family": "marker" if kind.startswith("trade_") else "line",
                        "vendor": "tradingview",
                        "vendor_type": vendor_type,
                    },
                    "anchors": {
                        "points": [{"time": timestamp, "price": item.get("price")}],
                    },
                    "geometry": {"type": "point" if kind.startswith("trade_") else "polyline"},
                    "content": {"text": item.get("label")},
                    "style": {},
                    "relations": {},
                    "lifecycle": {"editable": False},
                    "vendor_native": {},
                }
            )
    if entry_price is not None:
        docs.append(
            {
                "id": f"backtest:{trade_uid}:risk_reward",
                "source": {"type": "backtest", "source_id": trade_uid},
                "scope": scope,
                "semantic": {
                    "kind": "risk_reward",
                    "role": "execution",
                    "direction": side,
                    "status": "closed",
                },
                "tool": {
                    "family": "trading_box",
                    "vendor": "tradingview",
                    "vendor_type": "long_position" if side != "short" else "short_position",
                },
                "anchors": {"points": [{"time": int(datetime.now(UTC).timestamp()), "price": float(entry_price)}]},
                "geometry": {"type": "composite"},
                "content": {},
                "style": {},
                "relations": {},
                "lifecycle": {"editable": False},
                "vendor_native": {
                    "trade": {
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                    }
                },
            }
        )
    return docs
