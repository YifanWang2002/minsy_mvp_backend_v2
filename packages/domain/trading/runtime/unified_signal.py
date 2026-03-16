"""Unified trading signal model and compiler helpers."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from decimal import Decimal, InvalidOperation
from typing import Any

from packages.domain.trading.broker_capability_policy import (
    capability_supports_order_type,
    capability_supports_time_in_force,
)
from packages.infra.providers.trading.adapters.base import OrderIntent

_SUPPORTED_ORDER_TYPES = {"market", "limit", "stop", "stop_limit"}
_SUPPORTED_TIME_IN_FORCE = {"day", "gtc", "ioc", "fok"}


def _to_decimal_or_none(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None
    if parsed <= 0:
        return None
    return parsed


def _normalize_order_type(value: Any, *, default: str = "market") -> str:
    normalized = str(value or "").strip().lower()
    if normalized in _SUPPORTED_ORDER_TYPES:
        return normalized
    return default


def _normalize_tif(value: Any, *, default: str = "gtc") -> str:
    normalized = str(value or "").strip().lower()
    if normalized in _SUPPORTED_TIME_IN_FORCE:
        return normalized
    return default


def _resolve_side_trade_payload(
    *,
    strategy_payload: dict[str, Any] | None,
    side_name: str,
) -> dict[str, Any]:
    payload = strategy_payload if isinstance(strategy_payload, dict) else {}
    trade = payload.get("trade") if isinstance(payload.get("trade"), dict) else {}
    side_payload = trade.get(side_name) if isinstance(trade, dict) else {}
    return side_payload if isinstance(side_payload, dict) else {}


def _resolve_open_order_payload(
    *,
    strategy_payload: dict[str, Any] | None,
    signal: str,
) -> tuple[dict[str, Any], str]:
    if signal == "OPEN_SHORT":
        side_payload = _resolve_side_trade_payload(
            strategy_payload=strategy_payload,
            side_name="short",
        )
    else:
        side_payload = _resolve_side_trade_payload(
            strategy_payload=strategy_payload,
            side_name="long",
        )
    entry = side_payload.get("entry") if isinstance(side_payload.get("entry"), dict) else {}
    order_payload = entry.get("order") if isinstance(entry.get("order"), dict) else {}
    source = "strategy.entry.order" if isinstance(order_payload, dict) and order_payload else "default.market"
    return (order_payload if isinstance(order_payload, dict) else {}), source


def _resolve_close_order_payload(
    *,
    strategy_payload: dict[str, Any] | None,
    position_side: str,
    reason: str,
) -> tuple[dict[str, Any], str]:
    side_name = "long" if str(position_side).strip().lower() == "long" else "short"
    side_payload = _resolve_side_trade_payload(
        strategy_payload=strategy_payload,
        side_name=side_name,
    )
    exits = side_payload.get("exits")
    if not isinstance(exits, list):
        return {}, "default.market"

    normalized_reason = str(reason or "").strip().lower()
    for item in exits:
        if not isinstance(item, dict):
            continue
        rule_name = str(item.get("name") or "").strip().lower()
        rule_type = str(item.get("type") or "").strip().lower()
        if normalized_reason and normalized_reason not in {rule_name, rule_type}:
            continue
        order_payload = item.get("order")
        if isinstance(order_payload, dict) and order_payload:
            return order_payload, "strategy.exit.order"
    return {}, "default.market"


def _resolve_manual_order_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], str]:
    order = payload.get("order")
    if isinstance(order, dict) and order:
        return order, "manual.payload.order"
    fallback = {
        "type": payload.get("order_type"),
        "time_in_force": payload.get("time_in_force"),
        "limit_price": payload.get("limit_price"),
        "stop_price": payload.get("stop_price"),
    }
    if any(value is not None for value in fallback.values()):
        return fallback, "manual.payload.flat_fields"
    return {}, "default.market"


def _apply_price_fallbacks(
    *,
    order_type: str,
    limit_price: Decimal | None,
    stop_price: Decimal | None,
    mark_price: Decimal,
) -> tuple[str, Decimal | None, Decimal | None, dict[str, Any]]:
    metadata: dict[str, Any] = {}
    resolved_type = order_type
    resolved_limit = limit_price
    resolved_stop = stop_price
    if order_type == "limit" and resolved_limit is None and mark_price > 0:
        resolved_limit = mark_price
        metadata["limit_price_fallback"] = "mark_price"
    elif order_type == "stop" and resolved_stop is None and mark_price > 0:
        resolved_stop = mark_price
        metadata["stop_price_fallback"] = "mark_price"
    elif order_type == "stop_limit":
        if resolved_limit is None and mark_price > 0:
            resolved_limit = mark_price
            metadata["limit_price_fallback"] = "mark_price"
        if resolved_stop is None and mark_price > 0:
            resolved_stop = mark_price
            metadata["stop_price_fallback"] = "mark_price"

    if resolved_type in {"limit", "stop", "stop_limit"}:
        if (resolved_type == "limit" and resolved_limit is None) or (
            resolved_type == "stop" and resolved_stop is None
        ) or (resolved_type == "stop_limit" and (resolved_limit is None or resolved_stop is None)):
            resolved_type = "market"
            resolved_limit = None
            resolved_stop = None
            metadata["order_type_downgraded"] = "missing_price_reference"
    return resolved_type, resolved_limit, resolved_stop, metadata


@dataclass(frozen=True, slots=True)
class UnifiedTradingSignal:
    """Normalized order-intent payload shared by runtime/manual paths."""

    signal: str
    action: str
    direction: str
    symbol: str
    market: str
    timeframe: str
    side: str
    qty: Decimal
    reason: str
    order_type: str = "market"
    time_in_force: str = "gtc"
    limit_price: Decimal | None = None
    stop_price: Decimal | None = None
    bracket_stop_price: Decimal | None = None
    bracket_take_price: Decimal | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def build_runtime_unified_signal(
    *,
    signal: str,
    reason: str,
    symbol: str,
    market: str,
    timeframe: str,
    qty: Decimal,
    position_side: str,
    strategy_payload: dict[str, Any] | None,
    mark_price: Decimal,
    base_metadata: dict[str, Any] | None = None,
) -> UnifiedTradingSignal:
    normalized_signal = str(signal or "").strip().upper()
    if normalized_signal == "OPEN_SHORT":
        action = "open"
        direction = "short"
        side = "sell"
        order_payload, source = _resolve_open_order_payload(
            strategy_payload=strategy_payload,
            signal=normalized_signal,
        )
    elif normalized_signal == "CLOSE":
        action = "close"
        direction = "flat"
        side = "sell" if str(position_side).strip().lower() == "long" else "buy"
        order_payload, source = _resolve_close_order_payload(
            strategy_payload=strategy_payload,
            position_side=position_side,
            reason=reason,
        )
    else:
        action = "open"
        direction = "long"
        side = "buy"
        order_payload, source = _resolve_open_order_payload(
            strategy_payload=strategy_payload,
            signal=normalized_signal,
        )

    order_type = _normalize_order_type(order_payload.get("type"), default="market")
    time_in_force = _normalize_tif(order_payload.get("time_in_force"), default="gtc")
    limit_price = _to_decimal_or_none(order_payload.get("limit_price"))
    stop_price = _to_decimal_or_none(order_payload.get("stop_price"))
    order_type, limit_price, stop_price, fallback_metadata = _apply_price_fallbacks(
        order_type=order_type,
        limit_price=limit_price,
        stop_price=stop_price,
        mark_price=mark_price,
    )

    metadata = dict(base_metadata) if isinstance(base_metadata, dict) else {}
    metadata.update(
        {
            "order_spec_source": source,
            "requested_order_type": _normalize_order_type(order_payload.get("type"), default="market"),
            "requested_time_in_force": _normalize_tif(order_payload.get("time_in_force"), default="gtc"),
            **fallback_metadata,
        }
    )

    return UnifiedTradingSignal(
        signal=normalized_signal,
        action=action,
        direction=direction,
        symbol=str(symbol).strip().upper(),
        market=str(market).strip().lower(),
        timeframe=str(timeframe).strip().lower(),
        side=side,
        qty=qty,
        reason=str(reason or "").strip() or "runtime_signal",
        order_type=order_type,
        time_in_force=time_in_force,
        limit_price=limit_price,
        stop_price=stop_price,
        metadata=metadata,
    )


def build_manual_unified_signal(
    *,
    action: str,
    signal: str,
    reason: str,
    symbol: str,
    market: str,
    timeframe: str,
    qty: Decimal,
    current_position_side: str,
    mark_price: Decimal,
    payload: dict[str, Any] | None,
    base_metadata: dict[str, Any] | None = None,
) -> UnifiedTradingSignal:
    action_key = str(action or "").strip().lower()
    if signal == "OPEN_SHORT":
        direction = "short"
        side = "sell"
    elif signal == "CLOSE":
        direction = "flat"
        side = "sell" if str(current_position_side).strip().lower() == "long" else "buy"
    else:
        direction = "long"
        side = "buy"

    order_payload, source = _resolve_manual_order_payload(payload or {})
    order_type = _normalize_order_type(order_payload.get("type"), default="market")
    time_in_force = _normalize_tif(order_payload.get("time_in_force"), default="gtc")
    limit_price = _to_decimal_or_none(order_payload.get("limit_price"))
    stop_price = _to_decimal_or_none(order_payload.get("stop_price"))
    order_type, limit_price, stop_price, fallback_metadata = _apply_price_fallbacks(
        order_type=order_type,
        limit_price=limit_price,
        stop_price=stop_price,
        mark_price=mark_price,
    )

    metadata = dict(base_metadata) if isinstance(base_metadata, dict) else {}
    metadata.update(
        {
            "order_spec_source": source,
            "requested_order_type": _normalize_order_type(order_payload.get("type"), default="market"),
            "requested_time_in_force": _normalize_tif(order_payload.get("time_in_force"), default="gtc"),
            **fallback_metadata,
        }
    )

    return UnifiedTradingSignal(
        signal=signal,
        action=action_key,
        direction=direction,
        symbol=str(symbol).strip().upper(),
        market=str(market).strip().lower(),
        timeframe=str(timeframe).strip().lower(),
        side=side,
        qty=qty,
        reason=str(reason or "").strip() or action_key or "manual_action",
        order_type=order_type,
        time_in_force=time_in_force,
        limit_price=limit_price,
        stop_price=stop_price,
        metadata=metadata,
    )


def _fallback_time_in_force(capabilities: dict[str, Any] | None) -> str:
    if not isinstance(capabilities, dict):
        return "gtc"
    raw = capabilities.get("time_in_force")
    if not isinstance(raw, list):
        return "gtc"
    for candidate in raw:
        normalized = _normalize_tif(candidate, default="")
        if normalized:
            return normalized
    return "gtc"


def reconcile_unified_signal_capabilities(
    *,
    unified_signal: UnifiedTradingSignal,
    capabilities: dict[str, Any] | None,
    strict: bool = False,
) -> tuple[UnifiedTradingSignal | None, str | None]:
    """Reconcile unified signal order semantics with broker capabilities."""

    if not isinstance(capabilities, dict) or not capabilities:
        return unified_signal, None

    signal = unified_signal
    metadata = dict(signal.metadata)
    if not capability_supports_order_type(
        capabilities=capabilities,
        order_type=signal.order_type,
    ):
        if strict:
            return None, f"unsupported_order_type:{signal.order_type}"
        metadata["order_type_downgraded_from"] = signal.order_type
        signal = replace(
            signal,
            order_type="market",
            limit_price=None,
            stop_price=None,
            metadata=metadata,
        )
    if not capability_supports_time_in_force(
        capabilities=capabilities,
        time_in_force=signal.time_in_force,
    ):
        if strict:
            return None, f"unsupported_time_in_force:{signal.time_in_force}"
        fallback_tif = _fallback_time_in_force(capabilities)
        metadata = dict(signal.metadata)
        metadata["time_in_force_downgraded_from"] = signal.time_in_force
        metadata["time_in_force_downgraded_to"] = fallback_tif
        signal = replace(
            signal,
            time_in_force=fallback_tif,
            metadata=metadata,
        )
    return signal, None


def unified_signal_to_order_intent(
    *,
    unified_signal: UnifiedTradingSignal,
    client_order_id: str,
) -> OrderIntent:
    """Compile unified signal into adapter-facing OrderIntent."""

    metadata = dict(unified_signal.metadata)
    metadata.update(
        {
            "signal": unified_signal.signal,
            "reason": unified_signal.reason,
            "market": unified_signal.market,
            "timeframe": unified_signal.timeframe,
            "unified_action": unified_signal.action,
            "unified_direction": unified_signal.direction,
            "unified_order_type": unified_signal.order_type,
            "unified_time_in_force": unified_signal.time_in_force,
            "unified_limit_price": (
                str(unified_signal.limit_price) if unified_signal.limit_price is not None else None
            ),
            "unified_stop_price": (
                str(unified_signal.stop_price) if unified_signal.stop_price is not None else None
            ),
        }
    )
    return OrderIntent(
        client_order_id=client_order_id,
        symbol=unified_signal.symbol,
        side=unified_signal.side,
        qty=unified_signal.qty,
        order_type=unified_signal.order_type,
        limit_price=unified_signal.limit_price,
        stop_price=unified_signal.stop_price,
        time_in_force=unified_signal.time_in_force,
        metadata=metadata,
    )

