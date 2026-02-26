"""Telegram templates for production notification events."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from packages.core.events.notification_events import (
    EVENT_BACKTEST_COMPLETED,
    EVENT_DEPLOYMENT_STARTED,
    EVENT_EXECUTION_ANOMALY,
    EVENT_POSITION_CLOSED,
    EVENT_POSITION_OPENED,
    EVENT_RISK_TRIGGERED,
    EVENT_TRADE_APPROVAL_APPROVED,
    EVENT_TRADE_APPROVAL_EXPIRED,
    EVENT_TRADE_APPROVAL_REJECTED,
    EVENT_TRADE_APPROVAL_REQUESTED,
)


def render_telegram_notification(
    *,
    event_type: str,
    payload: dict[str, Any],
    locale: str,
) -> str:
    """Render one notification event into Telegram plain-text message."""
    lang = "zh" if str(locale).strip().lower().startswith("zh") else "en"
    if event_type == EVENT_BACKTEST_COMPLETED:
        return _render_backtest_completed(payload=payload, locale=lang)
    if event_type == EVENT_DEPLOYMENT_STARTED:
        return _render_deployment_started(payload=payload, locale=lang)
    if event_type == EVENT_POSITION_OPENED:
        return _render_position_opened(payload=payload, locale=lang)
    if event_type == EVENT_POSITION_CLOSED:
        return _render_position_closed(payload=payload, locale=lang)
    if event_type == EVENT_TRADE_APPROVAL_REQUESTED:
        return _render_trade_approval_requested(payload=payload, locale=lang)
    if event_type == EVENT_TRADE_APPROVAL_APPROVED:
        return _render_trade_approval_approved(payload=payload, locale=lang)
    if event_type == EVENT_TRADE_APPROVAL_REJECTED:
        return _render_trade_approval_rejected(payload=payload, locale=lang)
    if event_type == EVENT_TRADE_APPROVAL_EXPIRED:
        return _render_trade_approval_expired(payload=payload, locale=lang)
    if event_type == EVENT_RISK_TRIGGERED:
        return _render_risk_triggered(payload=payload, locale=lang)
    if event_type == EVENT_EXECUTION_ANOMALY:
        return _render_execution_anomaly(payload=payload, locale=lang)

    title = "交易通知" if lang == "zh" else "Trading Notification"
    return f"🔔 {title}\n{event_type}\n{_render_compact_json(payload)}"


def _render_backtest_completed(*, payload: dict[str, Any], locale: str) -> str:
    job_id = _str(payload.get("job_id"), default="unknown")
    strategy_id = _str(payload.get("strategy_id"), default="unknown")
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    total_return_pct = _num(summary.get("total_return_pct"))
    max_drawdown_pct = _num(summary.get("max_drawdown_pct"))
    sharpe = _num(summary.get("sharpe_ratio"))
    completed_at = _fmt_ts(payload.get("completed_at"))
    if locale == "zh":
        return (
            "✅ 回测完成\n"
            f"job_id: {job_id}\n"
            f"strategy_id: {strategy_id}\n"
            f"收益率: {_fmt_num(total_return_pct, suffix='%')}\n"
            f"最大回撤: {_fmt_num(max_drawdown_pct, suffix='%')}\n"
            f"Sharpe: {_fmt_num(sharpe)}\n"
            f"完成时间: {completed_at}"
        )
    return (
        "✅ Backtest Completed\n"
        f"job_id: {job_id}\n"
        f"strategy_id: {strategy_id}\n"
        f"Total return: {_fmt_num(total_return_pct, suffix='%')}\n"
        f"Max drawdown: {_fmt_num(max_drawdown_pct, suffix='%')}\n"
        f"Sharpe: {_fmt_num(sharpe)}\n"
        f"Completed at: {completed_at}"
    )


def _render_deployment_started(*, payload: dict[str, Any], locale: str) -> str:
    deployment_id = _str(payload.get("deployment_id"), default="unknown")
    mode = _str(payload.get("mode"), default="paper")
    timeframe = _str(payload.get("timeframe"), default="n/a")
    symbols = payload.get("symbols")
    if isinstance(symbols, list):
        symbol_text = ",".join(str(item) for item in symbols[:5]) or "n/a"
    else:
        symbol_text = _str(payload.get("symbol"), default="n/a")
    deployed_at = _fmt_ts(payload.get("deployed_at"))
    if locale == "zh":
        return (
            "🚀 策略部署成功\n"
            f"deployment_id: {deployment_id}\n"
            f"mode: {mode}\n"
            f"symbols: {symbol_text}\n"
            f"timeframe: {timeframe}\n"
            f"时间: {deployed_at}"
        )
    return (
        "🚀 Deployment Started\n"
        f"deployment_id: {deployment_id}\n"
        f"mode: {mode}\n"
        f"symbols: {symbol_text}\n"
        f"timeframe: {timeframe}\n"
        f"Time: {deployed_at}"
    )


def _render_position_opened(*, payload: dict[str, Any], locale: str) -> str:
    symbol = _str(payload.get("symbol"), default="n/a")
    side = _str(payload.get("side"), default="n/a")
    qty = _num(payload.get("qty"))
    price = _num(payload.get("price"))
    order_id = _str(payload.get("order_id"), default="unknown")
    reason = _str(payload.get("reason"), default="n/a")
    timestamp = _fmt_ts(payload.get("occurred_at"))
    if locale == "zh":
        return (
            "📈 开仓成交\n"
            f"symbol: {symbol}\n"
            f"side: {side}\n"
            f"qty: {_fmt_num(qty)}\n"
            f"price: {_fmt_num(price)}\n"
            f"order_id: {order_id}\n"
            f"reason: {reason}\n"
            f"时间: {timestamp}"
        )
    return (
        "📈 Position Opened\n"
        f"symbol: {symbol}\n"
        f"side: {side}\n"
        f"qty: {_fmt_num(qty)}\n"
        f"price: {_fmt_num(price)}\n"
        f"order_id: {order_id}\n"
        f"reason: {reason}\n"
        f"Time: {timestamp}"
    )


def _render_position_closed(*, payload: dict[str, Any], locale: str) -> str:
    symbol = _str(payload.get("symbol"), default="n/a")
    qty = _num(payload.get("qty"))
    exit_price = _num(payload.get("exit_price"))
    pnl_delta = _num(payload.get("realized_pnl_delta"))
    remaining_qty = _num(payload.get("remaining_qty"))
    order_id = _str(payload.get("order_id"), default="unknown")
    reason = _str(payload.get("reason"), default="n/a")
    timestamp = _fmt_ts(payload.get("occurred_at"))
    if locale == "zh":
        return (
            "📉 平仓成交\n"
            f"symbol: {symbol}\n"
            f"qty: {_fmt_num(qty)}\n"
            f"exit_price: {_fmt_num(exit_price)}\n"
            f"realized_pnl_delta: {_fmt_num(pnl_delta)}\n"
            f"remaining_qty: {_fmt_num(remaining_qty)}\n"
            f"order_id: {order_id}\n"
            f"reason: {reason}\n"
            f"时间: {timestamp}"
        )
    return (
        "📉 Position Closed\n"
        f"symbol: {symbol}\n"
        f"qty: {_fmt_num(qty)}\n"
        f"exit_price: {_fmt_num(exit_price)}\n"
        f"realized_pnl_delta: {_fmt_num(pnl_delta)}\n"
        f"remaining_qty: {_fmt_num(remaining_qty)}\n"
        f"order_id: {order_id}\n"
        f"reason: {reason}\n"
        f"Time: {timestamp}"
    )


def _render_trade_approval_requested(*, payload: dict[str, Any], locale: str) -> str:
    request_id = _str(payload.get("approval_request_id"), default="unknown")
    symbol = _str(payload.get("symbol"), default="n/a")
    side = _str(payload.get("side"), default="n/a")
    qty = _num(payload.get("qty"))
    mark_price = _num(payload.get("mark_price"))
    reason = _str(payload.get("reason"), default="n/a")
    expires_at = _fmt_ts(payload.get("expires_at"))
    if locale == "zh":
        return (
            "🟡 开仓待审批\n"
            f"request_id: {request_id}\n"
            f"symbol: {symbol}\n"
            f"side: {side}\n"
            f"qty: {_fmt_num(qty)}\n"
            f"mark_price: {_fmt_num(mark_price)}\n"
            f"reason: {reason}\n"
            f"过期时间: {expires_at}\n"
            "请点击下方按钮执行审批。"
        )
    return (
        "🟡 Open Trade Approval Needed\n"
        f"request_id: {request_id}\n"
        f"symbol: {symbol}\n"
        f"side: {side}\n"
        f"qty: {_fmt_num(qty)}\n"
        f"mark_price: {_fmt_num(mark_price)}\n"
        f"reason: {reason}\n"
        f"Expires at: {expires_at}\n"
        "Tap a button below to decide."
    )


def _render_trade_approval_approved(*, payload: dict[str, Any], locale: str) -> str:
    request_id = _str(payload.get("approval_request_id"), default="unknown")
    symbol = _str(payload.get("symbol"), default="n/a")
    if locale == "zh":
        return (
            "✅ 审批通过\n"
            f"request_id: {request_id}\n"
            f"symbol: {symbol}\n"
            "系统将开始执行开仓。"
        )
    return (
        "✅ Approval Accepted\n"
        f"request_id: {request_id}\n"
        f"symbol: {symbol}\n"
        "Order execution has been queued."
    )


def _render_trade_approval_rejected(*, payload: dict[str, Any], locale: str) -> str:
    request_id = _str(payload.get("approval_request_id"), default="unknown")
    symbol = _str(payload.get("symbol"), default="n/a")
    if locale == "zh":
        return (
            "⛔ 审批已拒绝\n"
            f"request_id: {request_id}\n"
            f"symbol: {symbol}"
        )
    return (
        "⛔ Approval Rejected\n"
        f"request_id: {request_id}\n"
        f"symbol: {symbol}"
    )


def _render_trade_approval_expired(*, payload: dict[str, Any], locale: str) -> str:
    request_id = _str(payload.get("approval_request_id"), default="unknown")
    symbol = _str(payload.get("symbol"), default="n/a")
    if locale == "zh":
        return (
            "⌛ 审批已超时\n"
            f"request_id: {request_id}\n"
            f"symbol: {symbol}"
        )
    return (
        "⌛ Approval Expired\n"
        f"request_id: {request_id}\n"
        f"symbol: {symbol}"
    )


def _render_risk_triggered(*, payload: dict[str, Any], locale: str) -> str:
    deployment_id = _str(payload.get("deployment_id"), default="unknown")
    reason = _str(payload.get("reason"), default="risk_guard")
    timestamp = _fmt_ts(payload.get("occurred_at"))
    if locale == "zh":
        return (
            "⚠️ 风控触发\n"
            f"deployment_id: {deployment_id}\n"
            f"reason: {reason}\n"
            f"时间: {timestamp}"
        )
    return (
        "⚠️ Risk Triggered\n"
        f"deployment_id: {deployment_id}\n"
        f"reason: {reason}\n"
        f"Time: {timestamp}"
    )


def _render_execution_anomaly(*, payload: dict[str, Any], locale: str) -> str:
    deployment_id = _str(payload.get("deployment_id"), default="unknown")
    reason = _str(payload.get("reason"), default="execution_anomaly")
    timestamp = _fmt_ts(payload.get("occurred_at"))
    if locale == "zh":
        return (
            "🚨 执行异常\n"
            f"deployment_id: {deployment_id}\n"
            f"reason: {reason}\n"
            f"时间: {timestamp}"
        )
    return (
        "🚨 Execution Anomaly\n"
        f"deployment_id: {deployment_id}\n"
        f"reason: {reason}\n"
        f"Time: {timestamp}"
    )


def _str(value: object, *, default: str) -> str:
    text = str(value).strip() if value is not None else ""
    return text or default


def _num(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt_num(value: float | None, *, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    if abs(value) >= 1000:
        text = f"{value:,.2f}"
    else:
        text = f"{value:.4f}".rstrip("0").rstrip(".")
    return f"{text}{suffix}"


def _fmt_ts(value: object) -> str:
    if isinstance(value, datetime):
        dt = value.astimezone(UTC) if value.tzinfo is not None else value.replace(tzinfo=UTC)
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return _fmt_ts(datetime.now(UTC))
        normalized = text.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return text
        return _fmt_ts(parsed)
    return _fmt_ts(datetime.now(UTC))


def _render_compact_json(payload: dict[str, Any]) -> str:
    if not payload:
        return "{}"
    lines: list[str] = []
    for key in sorted(payload):
        value = payload[key]
        lines.append(f"{key}={value}")
        if len(lines) >= 8:
            break
    return ", ".join(lines)
