"""Project trading and backtest facts into canonical annotation documents."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
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


def _positive_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _timeframe_step_seconds(timeframe: str) -> int:
    text = str(timeframe or "").strip().lower()
    if not text:
        return 15 * 60
    if text.isdigit():
        return max(60, int(text) * 60)
    suffix = text[-1]
    amount = text[:-1]
    if not amount.isdigit():
        return 15 * 60
    multiplier = {
        "m": 60,
        "h": 60 * 60,
        "d": 24 * 60 * 60,
        "w": 7 * 24 * 60 * 60,
    }.get(suffix)
    if multiplier is None:
        return 15 * 60
    return max(60, int(amount) * multiplier)


def _timestamps_for_symbol_orders(*, symbol: str, orders: Iterable[Order]) -> list[int]:
    normalized_symbol = str(symbol or "").strip().upper()
    timestamps: list[int] = []
    for order in orders:
        if str(order.symbol or "").strip().upper() != normalized_symbol:
            continue
        timestamp = _iso_seconds(order.submitted_at)
        if timestamp is not None:
            timestamps.append(timestamp)
    return timestamps


def _signed_fill_events_for_symbol(
    *,
    symbol: str,
    orders: Iterable[Order],
    fills: Iterable[Fill],
) -> list[dict[str, Any]]:
    normalized_symbol = str(symbol or "").strip().upper()
    order_by_id = {
        order.id: order
        for order in orders
        if str(order.symbol or "").strip().upper() == normalized_symbol
    }
    events: list[dict[str, Any]] = []
    for fill in fills:
        order = order_by_id.get(fill.order_id)
        if order is None:
            continue
        timestamp = _iso_seconds(fill.filled_at)
        if timestamp is None:
            continue
        qty = _positive_float(fill.fill_qty)
        if qty is None:
            continue
        side = str(order.side or "").strip().lower()
        signed_qty = qty if side == "buy" else -qty
        events.append(
            {
                "time": timestamp,
                "signed_qty": signed_qty,
            }
        )
    events.sort(key=lambda item: (int(item["time"]), float(item["signed_qty"])))
    return events


def _infer_position_window(
    *,
    symbol: str,
    side: str,
    timeframe: str,
    orders: Iterable[Order],
    fills: Iterable[Fill],
) -> tuple[int | None, int | None]:
    normalized_side = str(side or "").strip().lower()
    if normalized_side not in {"long", "short"}:
        return None, None
    step_seconds = _timeframe_step_seconds(timeframe)
    events = _signed_fill_events_for_symbol(symbol=symbol, orders=orders, fills=fills)
    window_start: int | None = None
    latest_time: int | None = None
    running_qty = 0.0
    if events:
        for event in events:
            event_time = int(event["time"])
            latest_time = event_time
            previous_qty = running_qty
            running_qty += float(event["signed_qty"])
            if normalized_side == "long":
                if previous_qty <= 0 < running_qty:
                    window_start = event_time
                elif running_qty <= 0:
                    window_start = None
            else:
                if previous_qty >= 0 > running_qty:
                    window_start = event_time
                elif running_qty >= 0:
                    window_start = None
    if window_start is None:
        order_times = _timestamps_for_symbol_orders(symbol=symbol, orders=orders)
        if order_times:
            window_start = min(order_times)
            latest_time = max(order_times)
    if window_start is None:
        return None, None
    base_end_time = latest_time if latest_time is not None else window_start
    window_end = max(base_end_time, window_start + step_seconds * 12)
    return window_start, window_end


def _managed_exit_prices(
    *,
    managed_exit_state: Mapping[str, Any] | None,
    symbol: str,
    side: str,
) -> tuple[float | None, float | None]:
    payload = managed_exit_state if isinstance(managed_exit_state, Mapping) else {}
    if not payload:
        return None, None
    managed_symbol = str(payload.get("symbol") or "").strip().upper()
    normalized_symbol = str(symbol or "").strip().upper()
    if managed_symbol and managed_symbol != normalized_symbol:
        return None, None
    managed_side = str(payload.get("side") or "").strip().lower()
    normalized_side = str(side or "").strip().lower()
    if managed_side and managed_side != normalized_side:
        return None, None
    return (
        _positive_float(payload.get("stop_price")),
        _positive_float(payload.get("take_price")),
    )


def _line_points(*, start_time: int, end_time: int, price: float) -> list[dict[str, Any]]:
    return [
        {"time": int(start_time), "price": float(price)},
        {"time": int(end_time), "price": float(price)},
    ]


def _bundle_member_ids(stop_loss_id: str | None, take_profit_id: str | None) -> list[str]:
    return [
        member_id
        for member_id in [stop_loss_id, take_profit_id]
        if isinstance(member_id, str) and member_id
    ]


def _trade_annotation_timestamp(raw_value: Any) -> int | None:
    if isinstance(raw_value, datetime):
        return _iso_seconds(raw_value)
    if isinstance(raw_value, (int, float)):
        return int(raw_value)
    if isinstance(raw_value, str):
        try:
            return int(datetime.fromisoformat(raw_value.replace("Z", "+00:00")).astimezone(UTC).timestamp())
        except ValueError:
            return None
    return None


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
    managed_exit_state: Mapping[str, Any] | None = None,
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
    normalized_symbol = str(symbol or "").strip().upper()
    for order in orders:
        if str(order.symbol or "").strip().upper() != normalized_symbol:
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
        if (
            str(position.symbol or "").strip().upper() != normalized_symbol
            or str(position.side).strip().lower() == "flat"
        ):
            continue
        direction = str(position.side).strip().lower()
        entry_price = _positive_float(position.avg_entry_price)
        if entry_price is None:
            continue
        entry_time, exit_time = _infer_position_window(
            symbol=normalized_symbol,
            side=direction,
            timeframe=timeframe,
            orders=orders,
            fills=fills,
        )
        if entry_time is None:
            entry_time = int(datetime.now(UTC).timestamp())
        if exit_time is None or exit_time <= entry_time:
            exit_time = entry_time + _timeframe_step_seconds(timeframe) * 12
        stop_price, take_price = _managed_exit_prices(
            managed_exit_state=managed_exit_state,
            symbol=normalized_symbol,
            side=direction,
        )
        group_id = f"execution:{deployment_id}:{normalized_symbol}:trade_bundle"
        stop_loss_id = (
            f"stop_loss:{deployment_id}:{normalized_symbol}"
            if stop_price is not None
            else None
        )
        take_profit_id = (
            f"take_profit:{deployment_id}:{normalized_symbol}"
            if take_price is not None
            else None
        )
        composite_members = _bundle_member_ids(stop_loss_id, take_profit_id)
        docs.append(
            {
                "id": f"position:{deployment_id}:{normalized_symbol}",
                "source": {
                    "type": "strategy_runtime",
                    "source_id": f"{deployment_id}:{normalized_symbol}",
                },
                "scope": scope,
                "semantic": {
                    "kind": "position",
                    "role": "execution",
                    "direction": direction,
                    "status": "active",
                },
                "tool": {
                    "family": "trading_box",
                    "vendor": "tradingview",
                    "vendor_type": "long_position"
                    if direction == "long"
                    else "short_position",
                },
                "anchors": {
                    "points": [
                        {"time": int(entry_time), "price": float(entry_price)},
                        {"time": int(exit_time), "price": float(entry_price)},
                    ],
                },
                "geometry": {"type": "composite"},
                "content": {},
                "style": {},
                "relations": (
                    {
                        "group_id": group_id,
                        "composite_members": composite_members,
                    }
                    if composite_members
                    else {"group_id": group_id}
                ),
                "lifecycle": {"editable": False},
                "vendor_native": {
                    "trade": {
                        "entry_time": int(entry_time),
                        "exit_time": int(exit_time),
                        "entry_price": float(entry_price),
                        "stop_price": stop_price,
                        "target_price": take_price,
                        "qty": _positive_float(position.qty),
                        "mark_price": _positive_float(position.mark_price),
                    }
                },
            }
        )
        if stop_price is not None:
            docs.append(
                {
                    "id": stop_loss_id,
                    "source": {
                        "type": "strategy_runtime",
                        "source_id": f"{deployment_id}:{normalized_symbol}:stop_loss",
                    },
                    "scope": scope,
                    "semantic": {
                        "kind": "stop_loss",
                        "role": "risk",
                        "direction": direction,
                        "status": "active",
                    },
                    "tool": {
                        "family": "line",
                        "vendor": "tradingview",
                        "vendor_type": "horizontal_line",
                    },
                    "anchors": {
                        "points": _line_points(
                            start_time=entry_time,
                            end_time=exit_time,
                            price=stop_price,
                        ),
                    },
                    "geometry": {"type": "polyline"},
                    "content": {"text": "Stop Loss"},
                    "style": {},
                    "relations": {
                        "group_id": group_id,
                        "parent_id": f"position:{deployment_id}:{normalized_symbol}",
                    },
                    "lifecycle": {"editable": False},
                    "vendor_native": {
                        "trade": {
                            "entry_time": int(entry_time),
                            "exit_time": int(exit_time),
                            "entry_price": float(entry_price),
                            "stop_price": stop_price,
                        }
                    },
                }
            )
        if take_price is not None:
            docs.append(
                {
                    "id": take_profit_id,
                    "source": {
                        "type": "strategy_runtime",
                        "source_id": f"{deployment_id}:{normalized_symbol}:take_profit",
                    },
                    "scope": scope,
                    "semantic": {
                        "kind": "take_profit",
                        "role": "risk",
                        "direction": direction,
                        "status": "active",
                    },
                    "tool": {
                        "family": "line",
                        "vendor": "tradingview",
                        "vendor_type": "horizontal_line",
                    },
                    "anchors": {
                        "points": _line_points(
                            start_time=entry_time,
                            end_time=exit_time,
                            price=take_price,
                        ),
                    },
                    "geometry": {"type": "polyline"},
                    "content": {"text": "Take Profit"},
                    "style": {},
                    "relations": {
                        "group_id": group_id,
                        "parent_id": f"position:{deployment_id}:{normalized_symbol}",
                    },
                    "lifecycle": {"editable": False},
                    "vendor_native": {
                        "trade": {
                            "entry_time": int(entry_time),
                            "exit_time": int(exit_time),
                            "entry_price": float(entry_price),
                            "target_price": take_price,
                        }
                    },
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
    group_id = f"backtest:{trade_uid}:trade_bundle"
    entry_time = _trade_annotation_timestamp(trade.get("entry_time"))
    exit_time = _trade_annotation_timestamp(trade.get("exit_time"))
    stop_price = _positive_float(trade.get("stop_price"))
    take_price = _positive_float(trade.get("take_price"))
    stop_loss_id: str | None = None
    take_profit_id: str | None = None
    normalized_annotations: list[dict[str, Any]] = []
    for item in trade_annotations:
        kind = str(item.get("kind") or "").strip().lower()
        if kind in {"trade_entry", "trade_exit", "stop_loss", "take_profit"}:
            timestamp = _trade_annotation_timestamp(item.get("time"))
            if timestamp is None:
                continue
            if kind == "trade_entry" and entry_time is None:
                entry_time = timestamp
            if kind == "trade_exit":
                exit_time = timestamp
            if kind == "stop_loss":
                stop_price = _positive_float(item.get("price"))
                stop_loss_id = f"backtest:{trade_uid}:stop_loss"
            if kind == "take_profit":
                take_price = _positive_float(item.get("price"))
                take_profit_id = f"backtest:{trade_uid}:take_profit"
            normalized_annotations.append(
                {
                    "kind": kind,
                    "timestamp": timestamp,
                    "price": _positive_float(item.get("price")),
                    "label": item.get("label"),
                }
            )
    effective_entry_time = entry_time
    if effective_entry_time is None and normalized_annotations:
        effective_entry_time = int(normalized_annotations[0]["timestamp"])
    if effective_entry_time is None:
        effective_entry_time = int(datetime.now(UTC).timestamp())
    effective_exit_time = exit_time
    if effective_exit_time is None or effective_exit_time <= effective_entry_time:
        effective_exit_time = effective_entry_time + _timeframe_step_seconds(timeframe) * 12
    for item in normalized_annotations:
        kind = str(item["kind"])
        timestamp = int(item["timestamp"])
        point_price = item["price"]
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
                    "points": (
                        _line_points(
                            start_time=effective_entry_time,
                            end_time=effective_exit_time,
                            price=point_price,
                        )
                        if kind in {"stop_loss", "take_profit"} and point_price is not None
                        else [{"time": timestamp, "price": point_price}]
                    ),
                },
                "geometry": {"type": "point" if kind.startswith("trade_") else "polyline"},
                "content": {"text": item["label"]},
                "style": {},
                "relations": {
                    "group_id": group_id,
                    **(
                        {"parent_id": f"backtest:{trade_uid}:risk_reward"}
                        if kind in {"stop_loss", "take_profit"}
                        else {}
                    ),
                },
                "lifecycle": {"editable": False},
                "vendor_native": {
                    "trade": {
                        "entry_time": effective_entry_time,
                        "exit_time": effective_exit_time,
                        "entry_price": _positive_float(entry_price),
                        "exit_price": _positive_float(exit_price),
                        "stop_price": stop_price,
                        "target_price": take_price,
                    }
                },
            }
        )
    if entry_price is not None:
        normalized_entry_price = _positive_float(entry_price)
        if normalized_entry_price is None:
            return docs
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
                "anchors": {
                    "points": [
                        {"time": int(effective_entry_time), "price": float(normalized_entry_price)},
                        {"time": int(effective_exit_time), "price": float(normalized_entry_price)},
                    ]
                },
                "geometry": {"type": "composite"},
                "content": {},
                "style": {},
                "relations": {
                    "group_id": group_id,
                    "composite_members": _bundle_member_ids(stop_loss_id, take_profit_id),
                },
                "lifecycle": {"editable": False},
                "vendor_native": {
                    "trade": {
                        "entry_time": int(effective_entry_time),
                        "exit_time": int(effective_exit_time),
                        "entry_price": normalized_entry_price,
                        "exit_price": _positive_float(exit_price),
                        "stop_price": stop_price,
                        "target_price": take_price,
                    }
                },
            }
        )
    return docs
