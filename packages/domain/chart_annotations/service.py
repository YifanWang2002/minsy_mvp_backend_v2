"""Persistence and query services for canonical chart annotations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import Select, and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.routes import chart_datafeed as chart_datafeed_route
from packages.domain.chart_annotations.projector import (
    build_backtest_trade_annotation_documents,
    build_execution_annotation_documents,
)
from packages.domain.chart_annotations.pubsub import (
    ChartAnnotationRealtimeEvent,
    publish_chart_annotation_event,
)
from packages.infra.db.models.backtest import BacktestJob
from packages.infra.db.models.chart_annotation import ChartAnnotation
from packages.infra.db.models.chart_annotation_outbox import ChartAnnotationOutbox
from packages.infra.db.models.chart_annotation_revision import ChartAnnotationRevision
from packages.infra.db.models.deployment import Deployment
from packages.infra.db.models.deployment_run import DeploymentRun
from packages.infra.db.models.fill import Fill
from packages.infra.db.models.order import Order
from packages.infra.db.models.position import Position
from packages.infra.db.models.signal_event import SignalEvent

_SOURCE_TYPES = {"strategy_runtime", "ai_agent", "user_manual", "backtest", "system"}
_ANCHOR_SPACES = {"time_price", "time_only", "price_only", "viewport_percent"}
_SEMANTIC_KINDS = {
    "signal",
    "entry",
    "exit",
    "stop_loss",
    "take_profit",
    "risk_reward",
    "position",
    "trendline",
    "support_resistance",
    "channel",
    "fib",
    "gann",
    "pattern",
    "cycle",
    "forecast",
    "measurement",
    "note",
    "media",
    "table",
    "profile",
    "anchored_vwap",
}
_SEMANTIC_ROLES = {"execution", "analysis", "risk", "markup", "demo"}
_SEMANTIC_DIRECTIONS = {"long", "short", "neutral"}
_SEMANTIC_STATUSES = {
    "planned",
    "active",
    "filled",
    "closed",
    "canceled",
    "invalidated",
}
_TOOL_FAMILIES = {
    "marker",
    "text",
    "line",
    "channel",
    "fork",
    "fib",
    "gann",
    "pattern",
    "cycle",
    "forecast",
    "measurement",
    "shape",
    "brush",
    "trading_box",
    "profile",
    "media",
    "table",
    "anchored",
}
_FIB_VENDOR_TYPES = {
    "fib_retracement",
    "fib_trend_ext",
    "fib_speed_resist_fan",
    "fib_timezone",
    "fib_trend_time",
    "fib_circles",
    "fib_spiral",
    "fib_speed_resist_arcs",
    "fib_channel",
    "fib_wedge",
}
_GANN_VENDOR_TYPES = {
    "gannbox",
    "gannbox_square",
    "gannbox_fixed",
    "gannbox_fan",
}
_PROFILE_VENDOR_TYPES = {
    "anchored_vwap",
    "fixed_range_volume_profile",
}
_PATTERN_VENDOR_TYPES = {
    "xabcd_pattern",
    "cypher_pattern",
    "abcd_pattern",
    "triangle_pattern",
    "3divers_pattern",
    "head_and_shoulders",
    "elliott_impulse_wave",
    "elliott_triangle_wave",
    "elliott_triple_combo",
    "elliott_correction",
    "elliott_double_combo",
}
_FORK_VENDOR_TYPES = {
    "pitchfork",
    "schiff_pitchfork_modified",
    "schiff_pitchfork",
    "inside_pitchfork",
    "pitchfan",
}
_MEASUREMENT_VENDOR_TYPES = {
    "price_range",
    "date_range",
    "date_and_price_range",
}
_CYCLE_VENDOR_TYPES = {
    "cyclic_lines",
    "time_cycles",
    "sine_line",
}
_FORECAST_VENDOR_TYPES = {
    "forecast",
    "bars_pattern",
    "ghost_feed",
    "projection",
}
_TEXT_VENDOR_TYPES = {
    "text",
    "note",
    "text_note",
    "callout",
    "comment",
    "balloon",
}
_LINE_VENDOR_TYPES = {
    "trend_line",
    "ray",
    "horizontal_line",
    "vertical_line",
    "horizontal_ray",
    "cross_line",
    "info_line",
    "trend_angle",
    "arrow",
    "extended",
    "regression_trend",
}
_CHANNEL_VENDOR_TYPES = {
    "parallel_channel",
    "disjoint_angle",
    "flat_bottom",
}
_SHAPE_VENDOR_TYPES = {
    "rectangle",
    "rotated_rectangle",
    "circle",
    "ellipse",
    "triangle",
    "polyline",
    "path",
    "curve",
    "double_curve",
    "arc",
}
_BRUSH_VENDOR_TYPES = {
    "brush",
    "highlighter",
}
_TRADING_BOX_VENDOR_TYPES = {
    "long_position",
    "short_position",
}
_MEDIA_VENDOR_TYPES = {
    "icon",
    "emoji",
    "sticker",
}
_TABLE_VENDOR_TYPES = {
    "table",
}
_STRICT_VENDOR_TYPES_BY_FAMILY = {
    "text": _TEXT_VENDOR_TYPES,
    "line": _LINE_VENDOR_TYPES,
    "channel": _CHANNEL_VENDOR_TYPES,
    "shape": _SHAPE_VENDOR_TYPES,
    "brush": _BRUSH_VENDOR_TYPES,
    "fib": _FIB_VENDOR_TYPES,
    "gann": _GANN_VENDOR_TYPES,
    "profile": _PROFILE_VENDOR_TYPES,
    "pattern": _PATTERN_VENDOR_TYPES,
    "fork": _FORK_VENDOR_TYPES,
    "measurement": _MEASUREMENT_VENDOR_TYPES,
    "cycle": _CYCLE_VENDOR_TYPES,
    "forecast": _FORECAST_VENDOR_TYPES,
    "trading_box": _TRADING_BOX_VENDOR_TYPES,
    "media": _MEDIA_VENDOR_TYPES,
    "table": _TABLE_VENDOR_TYPES,
}
_GEOMETRY_TYPES = {
    "point",
    "polyline",
    "polygon",
    "range",
    "path",
    "anchored",
    "composite",
}
_EDITABLE_SOURCE_TYPES = {"ai_agent", "user_manual", "system"}
_OUTBOX_RETENTION_PER_OWNER = 4000
_TABLE_CONTENT_PREVIEW_ROWS = 8
_TABLE_CONTENT_PREVIEW_COLS = 8


class ChartAnnotationConflictError(RuntimeError):
    """Raised when annotation version preconditions fail."""

    def __init__(self, *, latest: dict[str, Any]) -> None:
        super().__init__("chart_annotation_version_conflict")
        self.latest = latest


@dataclass(frozen=True, slots=True)
class ChartAnnotationFilter:
    """Query filter for chart annotations."""

    market: str | None = None
    symbol: str | None = None
    timeframe: str | None = None
    deployment_id: UUID | None = None
    strategy_id: UUID | None = None
    backtest_id: UUID | None = None
    chart_layout_id: str | None = None
    from_dt: datetime | None = None
    to_dt: datetime | None = None
    include_deleted: bool = False


def _string(value: Any, *, default: str = "") -> str:
    text = str(value or "").strip()
    return text or default


def _optional_string(value: Any) -> str | None:
    text = _string(value)
    return text or None


def _normalize_enum(value: Any, *, allowed: set[str], default: str) -> str:
    candidate = _string(value, default=default).lower()
    if candidate not in allowed:
        return default
    return candidate


def _normalize_uuid_string(value: Any) -> str | None:
    text = _optional_string(value)
    if text is None:
        return None
    try:
        return str(UUID(text))
    except ValueError:
        return None


def _normalize_json_map(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): _normalize_json_value(item) for key, item in value.items()}
    return {}


def _normalize_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _normalize_json_map(value)
    if isinstance(value, list):
        return [_normalize_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_json_value(item) for item in value]
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    if isinstance(value, Decimal):
        return float(value)
    return value


def _parse_time(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo is not None else value.replace(tzinfo=UTC)
    if isinstance(value, (int, float)):
        numeric = float(value)
        if numeric > 9999999999:
            numeric = numeric / 1000.0
        try:
            return datetime.fromtimestamp(numeric, tz=UTC)
        except (OSError, OverflowError, ValueError):
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return _parse_time(float(text))
        except ValueError:
            pass
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    return None


def _collect_times(value: Any) -> list[datetime]:
    if isinstance(value, Mapping):
        results: list[datetime] = []
        for key, item in value.items():
            if str(key).lower() in {"time", "start_time", "end_time", "from_time", "to_time"}:
                parsed = _parse_time(item)
                if parsed is not None:
                    results.append(parsed)
            else:
                results.extend(_collect_times(item))
        return results
    if isinstance(value, list):
        results: list[datetime] = []
        for item in value:
            results.extend(_collect_times(item))
        return results
    return []


def _derive_time_window(
    *,
    anchor_space: str,
    anchors: dict[str, Any],
    geometry: dict[str, Any],
) -> tuple[datetime | None, datetime | None]:
    if anchor_space == "viewport_percent":
        return None, None
    times = _collect_times(anchors)
    times.extend(_collect_times(geometry))
    if not times:
        return None, None
    ordered = sorted(times)
    return ordered[0], ordered[-1]


def _default_geometry_type(
    *,
    anchor_space: str,
    tool_family: str,
    tool_vendor_type: str | None,
    anchors: dict[str, Any],
) -> str:
    if anchor_space == "viewport_percent":
        return "anchored"
    points = anchors.get("points")
    if tool_family in {"channel", "fork", "fib", "gann", "trading_box", "profile"}:
        return "composite"
    if tool_family == "measurement":
        return "range"
    if tool_family == "brush":
        return "path"
    if isinstance(points, list):
        if len(points) <= 1:
            return "point"
        if tool_family == "shape":
            normalized_vendor_type = _optional_string(tool_vendor_type)
            if normalized_vendor_type == "polyline":
                return "polyline"
            if normalized_vendor_type in {"path", "curve", "double_curve", "arc"}:
                return "path"
            return "polygon"
        return "polyline"
    return "point"


def _validate_native_line_tool_payload(
    *,
    tool_family: str,
    tool_vendor_type: str | None,
    vendor_native: Mapping[str, Any],
) -> None:
    allowed_vendor_types = _STRICT_VENDOR_TYPES_BY_FAMILY.get(tool_family)
    if not allowed_vendor_types:
        return
    if tool_vendor_type is None or tool_vendor_type not in allowed_vendor_types:
        raise ValueError(
            f"tool.vendor_type '{tool_vendor_type or ''}' is not supported for family '{tool_family}'"
        )
    if "state" in vendor_native and not isinstance(vendor_native.get("state"), Mapping):
        raise ValueError(
            f"vendor_native.state must be an object for family '{tool_family}'"
        )


def _normalize_table_cells_preview(value: Any) -> list[list[str]]:
    if not isinstance(value, list):
        return []
    preview: list[list[str]] = []
    for row in value[:_TABLE_CONTENT_PREVIEW_ROWS]:
        if not isinstance(row, list):
            continue
        preview.append(
            [
                _string(cell)
                for cell in row[:_TABLE_CONTENT_PREVIEW_COLS]
            ]
        )
    return preview


def _coerce_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float, Decimal)):
        return float(value)
    text = _optional_string(value)
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _coerce_epoch_seconds(value: Any) -> int | None:
    numeric = _coerce_number(value)
    if numeric is None:
        return None
    if numeric > 9999999999:
        numeric = numeric / 1000.0
    return int(round(numeric))


def _derive_trading_box_summary(
    *,
    tool_vendor_type: str | None,
    anchors: dict[str, Any],
    vendor_native: Mapping[str, Any],
) -> dict[str, Any]:
    summary = _normalize_json_map(vendor_native.get("trade"))
    line_tool_state = _normalize_json_map(vendor_native.get("state"))
    root_state = _normalize_json_map(line_tool_state.get("state"))
    nested_state = _normalize_json_map(root_state.get("state"))
    state_payload = nested_state or root_state
    properties = _normalize_json_map(vendor_native.get("properties"))
    anchor_points = anchors.get("points")
    normalized_anchor_points = (
        [item for item in anchor_points if isinstance(item, Mapping)]
        if isinstance(anchor_points, list)
        else []
    )
    entry_point = normalized_anchor_points[0] if normalized_anchor_points else {}
    second_point = (
        normalized_anchor_points[1]
        if len(normalized_anchor_points) > 1
        else entry_point
    )
    entry_time = (
        _coerce_epoch_seconds(summary.get("entry_time"))
        or _coerce_epoch_seconds(entry_point.get("time"))
    )
    exit_time = (
        _coerce_epoch_seconds(summary.get("exit_time"))
        or _coerce_epoch_seconds(second_point.get("time"))
    )
    entry_price = (
        _coerce_number(summary.get("entry_price"))
        or _coerce_number(entry_point.get("price"))
    )
    stop_price = (
        _coerce_number(summary.get("stop_price"))
        or _coerce_number(state_payload.get("stopLevel"))
        or _coerce_number(properties.get("stopLevel"))
    )
    target_price = (
        _coerce_number(summary.get("target_price"))
        or _coerce_number(state_payload.get("profitLevel"))
        or _coerce_number(properties.get("profitLevel"))
    )

    direction = "short" if tool_vendor_type == "short_position" else "long"
    normalized_summary = dict(summary)
    normalized_summary["direction"] = direction
    if entry_time is not None:
        normalized_summary["entry_time"] = entry_time
    if exit_time is not None:
        normalized_summary["exit_time"] = exit_time
    if entry_price is not None:
        normalized_summary["entry_price"] = entry_price
    if stop_price is not None:
        normalized_summary["stop_price"] = stop_price
    if target_price is not None:
        normalized_summary["target_price"] = target_price

    numeric_mapping = {
        "qty": state_payload.get("qty"),
        "risk": state_payload.get("risk"),
        "account_size": state_payload.get("accountSize"),
        "lot_size": state_payload.get("lotSize"),
        "leverage": state_payload.get("leverage"),
        "amount_stop": state_payload.get("amountStop"),
        "amount_target": state_payload.get("amountTarget"),
    }
    property_mapping = {
        "qty": properties.get("qty"),
        "risk": properties.get("risk"),
        "account_size": properties.get("accountSize"),
        "lot_size": properties.get("lotSize"),
        "leverage": properties.get("leverage"),
        "amount_stop": properties.get("amountStop"),
        "amount_target": properties.get("amountTarget"),
    }
    for key, nested_value in numeric_mapping.items():
        number = (
            _coerce_number(normalized_summary.get(key))
            or _coerce_number(nested_value)
            or _coerce_number(property_mapping.get(key))
        )
        if number is not None:
            normalized_summary[key] = number

    string_mapping = {
        "currency": state_payload.get("currency") or properties.get("currency"),
        "risk_display_mode": state_payload.get("riskDisplayMode")
        or properties.get("riskDisplayMode"),
        "line_color": state_payload.get("linecolor") or properties.get("linecolor"),
        "text_color": state_payload.get("textcolor") or properties.get("textcolor"),
    }
    for key, fallback in string_mapping.items():
        text = _optional_string(normalized_summary.get(key)) or _optional_string(fallback)
        if text is not None:
            normalized_summary[key] = text

    boolean_mapping = {
        "compact": state_payload.get("compact")
        if isinstance(state_payload.get("compact"), bool)
        else properties.get("compact"),
        "always_show_stats": state_payload.get("alwaysShowStats")
        if isinstance(state_payload.get("alwaysShowStats"), bool)
        else properties.get("alwaysShowStats"),
        "show_price_labels": state_payload.get("showPriceLabels")
        if isinstance(state_payload.get("showPriceLabels"), bool)
        else properties.get("showPriceLabels"),
    }
    for key, fallback in boolean_mapping.items():
        existing = normalized_summary.get(key)
        if isinstance(existing, bool):
            continue
        if isinstance(fallback, bool):
            normalized_summary[key] = fallback

    if (
        entry_price is not None
        and stop_price is not None
        and target_price is not None
        and abs(entry_price - stop_price) > 0
    ):
        normalized_summary["risk_reward_ratio"] = round(
            abs(target_price - entry_price) / abs(entry_price - stop_price),
            4,
        )
    return normalized_summary


def _derive_content_summary(
    *,
    tool_family: str,
    tool_vendor_type: str | None,
    content: dict[str, Any],
    anchors: dict[str, Any],
    vendor_native: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = dict(content)
    line_tool_state = _normalize_json_map(vendor_native.get("state"))
    nested_state = _normalize_json_map(line_tool_state.get("state"))
    if tool_family == "table":
        table_payload = _normalize_json_map(normalized.get("table"))
        rows_count = int(nested_state.get("rowsCount") or 0)
        cols_count = int(nested_state.get("colsCount") or 0)
        if rows_count <= 0 or cols_count <= 0:
            preview = _normalize_table_cells_preview(nested_state.get("cells"))
            if rows_count <= 0:
                rows_count = len(preview)
            if cols_count <= 0:
                cols_count = len(preview[0]) if preview else 0
        else:
            preview = _normalize_table_cells_preview(nested_state.get("cells"))
        table_payload["rows"] = rows_count
        table_payload["cols"] = cols_count
        if preview:
            table_payload["cells_preview"] = preview
        title = _optional_string(nested_state.get("title"))
        if title is not None:
            table_payload["title"] = title
        normalized["table"] = table_payload
    elif tool_family == "brush":
        stroke_payload = _normalize_json_map(normalized.get("stroke"))
        line_tool_points = line_tool_state.get("points")
        if isinstance(line_tool_points, list):
            point_count = len(line_tool_points)
        else:
            anchor_points = anchors.get("points")
            point_count = len(anchor_points) if isinstance(anchor_points, list) else 0
        stroke_payload["point_count"] = point_count
        smooth = nested_state.get("smooth")
        if isinstance(smooth, (int, float)):
            stroke_payload["smooth"] = float(smooth)
        normalized["stroke"] = stroke_payload
    elif tool_family == "media":
        media_payload = _normalize_json_map(normalized.get("media"))
        size = nested_state.get("size")
        if isinstance(size, (int, float)):
            media_payload["size"] = float(size)
        angle = nested_state.get("angle")
        if isinstance(angle, (int, float)):
            media_payload["angle"] = float(angle)
        if media_payload:
            normalized["media"] = media_payload
        emoji = _optional_string(nested_state.get("emoji"))
        if emoji is not None:
            normalized["emoji"] = emoji
        sticker = _optional_string(nested_state.get("sticker"))
        if sticker is not None:
            normalized["sticker"] = sticker
        icon_code = nested_state.get("icon")
        if isinstance(icon_code, (int, float)):
            normalized["icon_code"] = int(icon_code)
    elif tool_family == "trading_box":
        trade_payload = _normalize_json_map(normalized.get("trade"))
        trade_payload.update(
            _derive_trading_box_summary(
                tool_vendor_type=tool_vendor_type,
                anchors=anchors,
                vendor_native=vendor_native,
            )
        )
        if trade_payload:
            normalized["trade"] = trade_payload
    return normalized


def _normalize_annotation_payload(
    payload: Mapping[str, Any],
    *,
    owner_user_id: UUID,
    existing: ChartAnnotation | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    document = _normalize_json_map(payload)
    annotation_id = _normalize_uuid_string(document.get("id")) or (
        str(existing.id) if existing is not None else str(uuid4())
    )

    source = _normalize_json_map(document.get("source"))
    source_type = _normalize_enum(
        source.get("type"),
        allowed=_SOURCE_TYPES,
        default="user_manual",
    )
    source_id = _optional_string(source.get("source_id"))
    actor_user_id = _normalize_uuid_string(source.get("actor_user_id")) or str(owner_user_id)
    normalized_source: dict[str, Any] = {
        "type": source_type,
        "source_id": source_id,
        "actor_user_id": actor_user_id,
    }

    scope = _normalize_json_map(document.get("scope"))
    market = _string(scope.get("market") or getattr(existing, "market", None), default="stocks").lower()
    symbol = _string(scope.get("symbol") or getattr(existing, "symbol", None), default="").upper()
    timeframe = _string(scope.get("timeframe") or getattr(existing, "timeframe", None), default="1m").lower()
    anchor_space = _normalize_enum(
        scope.get("anchor_space") or getattr(existing, "anchor_space", None),
        allowed=_ANCHOR_SPACES,
        default="time_price",
    )
    deployment_id = _normalize_uuid_string(scope.get("deployment_id") or getattr(existing, "deployment_id", None))
    strategy_id = _normalize_uuid_string(scope.get("strategy_id") or getattr(existing, "strategy_id", None))
    backtest_id = _normalize_uuid_string(scope.get("backtest_id") or getattr(existing, "backtest_id", None))
    chart_layout_id = _optional_string(scope.get("chart_layout_id") or getattr(existing, "chart_layout_id", None))
    normalized_scope: dict[str, Any] = {
        "market": market,
        "symbol": symbol,
        "timeframe": timeframe,
        "anchor_space": anchor_space,
    }
    if deployment_id is not None:
        normalized_scope["deployment_id"] = deployment_id
    if strategy_id is not None:
        normalized_scope["strategy_id"] = strategy_id
    if backtest_id is not None:
        normalized_scope["backtest_id"] = backtest_id
    if chart_layout_id is not None:
        normalized_scope["chart_layout_id"] = chart_layout_id

    semantic = _normalize_json_map(document.get("semantic"))
    semantic_kind = _normalize_enum(
        semantic.get("kind") or getattr(existing, "semantic_kind", None),
        allowed=_SEMANTIC_KINDS,
        default="note",
    )
    semantic_role = _normalize_enum(
        semantic.get("role") or getattr(existing, "semantic_role", None),
        allowed=_SEMANTIC_ROLES,
        default="markup",
    )
    semantic_intent = _optional_string(semantic.get("intent"))
    semantic_direction = _optional_string(
        _normalize_enum(
            semantic.get("direction"),
            allowed=_SEMANTIC_DIRECTIONS,
            default="neutral",
        )
        if semantic.get("direction") is not None
        else None
    )
    semantic_status = _optional_string(
        _normalize_enum(
            semantic.get("status"),
            allowed=_SEMANTIC_STATUSES,
            default="active",
        )
        if semantic.get("status") is not None
        else None
    )
    normalized_semantic: dict[str, Any] = {
        "kind": semantic_kind,
        "role": semantic_role,
    }
    if semantic_intent is not None:
        normalized_semantic["intent"] = semantic_intent
    if semantic_direction is not None:
        normalized_semantic["direction"] = semantic_direction
    if semantic_status is not None:
        normalized_semantic["status"] = semantic_status

    tool = _normalize_json_map(document.get("tool"))
    tool_family = _normalize_enum(
        tool.get("family") or getattr(existing, "tool_family", None),
        allowed=_TOOL_FAMILIES,
        default="text",
    )
    tool_vendor = _string(tool.get("vendor") or getattr(existing, "tool_vendor", None), default="tradingview").lower()
    tool_vendor_type = _optional_string(tool.get("vendor_type") or getattr(existing, "tool_vendor_type", None))
    if tool_vendor_type is not None:
        tool_vendor_type = tool_vendor_type.lower()
    normalized_tool: dict[str, Any] = {
        "family": tool_family,
        "vendor": tool_vendor,
    }
    if tool_vendor_type is not None:
        normalized_tool["vendor_type"] = tool_vendor_type

    anchors = _normalize_json_map(document.get("anchors"))
    geometry = _normalize_json_map(document.get("geometry"))
    geometry_type = _normalize_enum(
        geometry.get("type"),
        allowed=_GEOMETRY_TYPES,
        default=_default_geometry_type(
            anchor_space=anchor_space,
            tool_family=tool_family,
            tool_vendor_type=tool_vendor_type,
            anchors=anchors,
        ),
    )
    normalized_geometry = {"type": geometry_type, **geometry}

    content = _normalize_json_map(document.get("content"))
    style = _normalize_json_map(document.get("style"))
    relations = _normalize_json_map(document.get("relations"))
    lifecycle = _normalize_json_map(document.get("lifecycle"))
    vendor_native = _normalize_json_map(document.get("vendor_native"))
    _validate_native_line_tool_payload(
        tool_family=tool_family,
        tool_vendor_type=tool_vendor_type,
        vendor_native=vendor_native,
    )
    if tool_family == "trading_box":
        trade_summary = _derive_trading_box_summary(
            tool_vendor_type=tool_vendor_type,
            anchors=anchors,
            vendor_native=vendor_native,
        )
        if trade_summary:
            vendor_native["trade"] = trade_summary
    content = _derive_content_summary(
        tool_family=tool_family,
        tool_vendor_type=tool_vendor_type,
        content=content,
        anchors=anchors,
        vendor_native=vendor_native,
    )

    is_editable = bool(
        lifecycle.get("editable")
        if lifecycle.get("editable") is not None
        else source_type in _EDITABLE_SOURCE_TYPES
    )
    lifecycle["editable"] = is_editable
    group_id = _optional_string(relations.get("group_id"))
    parent_id = _normalize_uuid_string(relations.get("parent_id"))

    time_start, time_end = _derive_time_window(
        anchor_space=anchor_space,
        anchors=anchors,
        geometry=normalized_geometry,
    )
    normalized_document: dict[str, Any] = {
        "id": annotation_id,
        "version": int(existing.version if existing is not None else 1),
        "source": normalized_source,
        "scope": normalized_scope,
        "semantic": normalized_semantic,
        "tool": normalized_tool,
        "anchors": anchors,
        "geometry": normalized_geometry,
        "content": content,
        "style": style,
        "relations": relations,
        "lifecycle": lifecycle,
        "vendor_native": vendor_native,
        "created_at": (
            existing.created_at.astimezone(UTC).isoformat()
            if existing is not None
            else datetime.now(UTC).isoformat()
        ),
    }
    metadata = {
        "annotation_id": UUID(annotation_id),
        "owner_user_id": owner_user_id,
        "actor_user_id": UUID(actor_user_id),
        "source_type": source_type,
        "source_id": source_id,
        "market": market,
        "symbol": symbol,
        "timeframe": timeframe,
        "chart_layout_id": chart_layout_id,
        "deployment_id": UUID(deployment_id) if deployment_id is not None else None,
        "strategy_id": UUID(strategy_id) if strategy_id is not None else None,
        "backtest_id": UUID(backtest_id) if backtest_id is not None else None,
        "anchor_space": anchor_space,
        "semantic_kind": semantic_kind,
        "semantic_role": semantic_role,
        "semantic_intent": semantic_intent,
        "semantic_direction": semantic_direction,
        "semantic_status": semantic_status,
        "tool_family": tool_family,
        "tool_vendor": tool_vendor,
        "tool_vendor_type": tool_vendor_type,
        "geometry_type": geometry_type,
        "group_id": group_id,
        "parent_id": UUID(parent_id) if parent_id is not None else None,
        "time_start": time_start,
        "time_end": time_end,
        "is_editable": is_editable,
    }
    return normalized_document, metadata


def _serialize_row(row: ChartAnnotation) -> dict[str, Any]:
    payload = dict(row.data) if isinstance(row.data, dict) else {}
    payload["id"] = str(row.id)
    payload["version"] = int(row.version)
    payload["created_at"] = row.created_at.astimezone(UTC).isoformat()
    payload["updated_at"] = row.updated_at.astimezone(UTC).isoformat()
    if row.deleted_at is not None:
        lifecycle = (
            payload.get("lifecycle")
            if isinstance(payload.get("lifecycle"), dict)
            else {}
        )
        lifecycle["deleted_at"] = row.deleted_at.astimezone(UTC).isoformat()
        payload["lifecycle"] = lifecycle
    return payload


def _apply_scope_filters(stmt: Select[tuple[ChartAnnotation]], filters: ChartAnnotationFilter) -> Select[tuple[ChartAnnotation]]:
    if filters.market:
        stmt = stmt.where(ChartAnnotation.market == filters.market.lower())
    if filters.symbol:
        stmt = stmt.where(ChartAnnotation.symbol == filters.symbol.upper())
    if filters.timeframe:
        stmt = stmt.where(ChartAnnotation.timeframe == filters.timeframe.lower())
    if filters.deployment_id is not None:
        stmt = stmt.where(ChartAnnotation.deployment_id == filters.deployment_id)
    if filters.strategy_id is not None:
        stmt = stmt.where(ChartAnnotation.strategy_id == filters.strategy_id)
    if filters.backtest_id is not None:
        stmt = stmt.where(ChartAnnotation.backtest_id == filters.backtest_id)
    if filters.chart_layout_id:
        stmt = stmt.where(ChartAnnotation.chart_layout_id == filters.chart_layout_id)
    if not filters.include_deleted:
        stmt = stmt.where(ChartAnnotation.is_deleted.is_(False))
    if filters.from_dt is not None or filters.to_dt is not None:
        upper_clause = (
            ChartAnnotation.time_start.is_(None)
            if filters.to_dt is None
            else or_(
                ChartAnnotation.time_start.is_(None),
                ChartAnnotation.time_start <= filters.to_dt,
            )
        )
        lower_clause = (
            ChartAnnotation.time_end.is_(None)
            if filters.from_dt is None
            else or_(
                ChartAnnotation.time_end.is_(None),
                ChartAnnotation.time_end >= filters.from_dt,
            )
        )
        stmt = stmt.where(
            or_(
                ChartAnnotation.anchor_space == "viewport_percent",
                and_(
                    upper_clause,
                    lower_clause,
                ),
            )
        )
    return stmt


async def _append_revision(
    db: AsyncSession,
    *,
    annotation_id: UUID,
    version: int,
    snapshot: dict[str, Any],
) -> None:
    db.add(
        ChartAnnotationRevision(
            annotation_id=annotation_id,
            version=version,
            snapshot=_normalize_json_map(snapshot),
        )
    )


async def _append_outbox(
    db: AsyncSession,
    *,
    row: ChartAnnotation,
    event_type: str,
    payload: dict[str, Any],
) -> ChartAnnotationOutbox:
    outbox = ChartAnnotationOutbox(
        owner_user_id=row.owner_user_id,
        annotation_id=row.id,
        event_type=event_type,
        market=row.market,
        symbol=row.symbol,
        timeframe=row.timeframe,
        chart_layout_id=row.chart_layout_id,
        payload=_normalize_json_map(payload),
    )
    db.add(outbox)
    await db.flush()
    return outbox


async def _publish_outbox(outbox: ChartAnnotationOutbox) -> None:
    payload = outbox.payload if isinstance(outbox.payload, dict) else {}
    await publish_chart_annotation_event(
        ChartAnnotationRealtimeEvent(
            owner_user_id=str(outbox.owner_user_id),
            event_type=str(outbox.event_type),
            event_seq=int(outbox.event_seq),
            payload=dict(payload),
        )
    )


async def _trim_outbox(db: AsyncSession, *, owner_user_id: UUID) -> None:
    cutoff = await db.scalar(
        select(ChartAnnotationOutbox.event_seq)
        .where(ChartAnnotationOutbox.owner_user_id == owner_user_id)
        .order_by(ChartAnnotationOutbox.event_seq.desc())
        .offset(_OUTBOX_RETENTION_PER_OWNER)
        .limit(1)
    )
    if cutoff is None:
        return
    await db.execute(
        ChartAnnotationOutbox.__table__.delete().where(
            ChartAnnotationOutbox.owner_user_id == owner_user_id,
            ChartAnnotationOutbox.event_seq < int(cutoff),
        )
    )


async def list_chart_annotations(
    db: AsyncSession,
    *,
    owner_user_id: UUID,
    filters: ChartAnnotationFilter,
) -> list[dict[str, Any]]:
    stmt = select(ChartAnnotation).where(ChartAnnotation.owner_user_id == owner_user_id)
    stmt = _apply_scope_filters(stmt, filters)
    stmt = stmt.order_by(
        ChartAnnotation.updated_at.asc(),
        ChartAnnotation.created_at.asc(),
    )
    rows = (await db.scalars(stmt)).all()
    return [_serialize_row(row) for row in rows]


async def list_chart_annotation_events(
    db: AsyncSession,
    *,
    owner_user_id: UUID,
    filters: ChartAnnotationFilter,
    cursor: int | None = None,
    limit: int = 200,
) -> list[ChartAnnotationOutbox]:
    stmt = select(ChartAnnotationOutbox).where(
        ChartAnnotationOutbox.owner_user_id == owner_user_id
    )
    if filters.market:
        stmt = stmt.where(ChartAnnotationOutbox.market == filters.market.lower())
    if filters.symbol:
        stmt = stmt.where(ChartAnnotationOutbox.symbol == filters.symbol.upper())
    if filters.timeframe:
        stmt = stmt.where(ChartAnnotationOutbox.timeframe == filters.timeframe.lower())
    if filters.chart_layout_id:
        stmt = stmt.where(ChartAnnotationOutbox.chart_layout_id == filters.chart_layout_id)
    if cursor is not None and cursor > 0:
        stmt = stmt.where(ChartAnnotationOutbox.event_seq > cursor)
    stmt = stmt.order_by(ChartAnnotationOutbox.event_seq.asc()).limit(max(1, min(limit, 1000)))
    return (await db.scalars(stmt)).all()


async def create_chart_annotation(
    db: AsyncSession,
    *,
    owner_user_id: UUID,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    document, metadata = _normalize_annotation_payload(payload, owner_user_id=owner_user_id)
    row = ChartAnnotation(
        id=metadata["annotation_id"],
        owner_user_id=owner_user_id,
        actor_user_id=metadata["actor_user_id"],
        version=1,
        source_type=metadata["source_type"],
        source_id=metadata["source_id"],
        market=metadata["market"],
        symbol=metadata["symbol"],
        timeframe=metadata["timeframe"],
        chart_layout_id=metadata["chart_layout_id"],
        deployment_id=metadata["deployment_id"],
        strategy_id=metadata["strategy_id"],
        backtest_id=metadata["backtest_id"],
        anchor_space=metadata["anchor_space"],
        semantic_kind=metadata["semantic_kind"],
        semantic_role=metadata["semantic_role"],
        semantic_intent=metadata["semantic_intent"],
        semantic_direction=metadata["semantic_direction"],
        semantic_status=metadata["semantic_status"],
        tool_family=metadata["tool_family"],
        tool_vendor=metadata["tool_vendor"],
        tool_vendor_type=metadata["tool_vendor_type"],
        geometry_type=metadata["geometry_type"],
        group_id=metadata["group_id"],
        parent_id=metadata["parent_id"],
        time_start=metadata["time_start"],
        time_end=metadata["time_end"],
        is_editable=metadata["is_editable"],
        is_deleted=False,
        data=document,
    )
    db.add(row)
    await db.flush()
    await db.refresh(row)
    document["version"] = 1
    document["created_at"] = row.created_at.astimezone(UTC).isoformat()
    document["updated_at"] = row.updated_at.astimezone(UTC).isoformat()
    row.data = document
    await _append_revision(
        db,
        annotation_id=row.id,
        version=1,
        snapshot=document,
    )
    outbox = await _append_outbox(
        db,
        row=row,
        event_type="annotation_upsert",
        payload={
            "annotation": document,
            "annotation_id": str(row.id),
            "scope": document.get("scope", {}),
        },
    )
    await _trim_outbox(db, owner_user_id=owner_user_id)
    await db.commit()
    await db.refresh(row)
    await _publish_outbox(outbox)
    return _serialize_row(row)


async def update_chart_annotation(
    db: AsyncSession,
    *,
    owner_user_id: UUID,
    annotation_id: UUID,
    base_version: int,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    row = await db.get(ChartAnnotation, annotation_id)
    if row is None or row.owner_user_id != owner_user_id:
        raise KeyError("chart_annotation_not_found")
    if int(base_version) != int(row.version):
        raise ChartAnnotationConflictError(latest=_serialize_row(row))
    if not row.is_editable:
        raise PermissionError("chart_annotation_locked")

    document, metadata = _normalize_annotation_payload(
        payload,
        owner_user_id=owner_user_id,
        existing=row,
    )
    row.version = int(row.version) + 1
    document["version"] = int(row.version)
    row.actor_user_id = metadata["actor_user_id"]
    row.source_type = metadata["source_type"]
    row.source_id = metadata["source_id"]
    row.market = metadata["market"]
    row.symbol = metadata["symbol"]
    row.timeframe = metadata["timeframe"]
    row.chart_layout_id = metadata["chart_layout_id"]
    row.deployment_id = metadata["deployment_id"]
    row.strategy_id = metadata["strategy_id"]
    row.backtest_id = metadata["backtest_id"]
    row.anchor_space = metadata["anchor_space"]
    row.semantic_kind = metadata["semantic_kind"]
    row.semantic_role = metadata["semantic_role"]
    row.semantic_intent = metadata["semantic_intent"]
    row.semantic_direction = metadata["semantic_direction"]
    row.semantic_status = metadata["semantic_status"]
    row.tool_family = metadata["tool_family"]
    row.tool_vendor = metadata["tool_vendor"]
    row.tool_vendor_type = metadata["tool_vendor_type"]
    row.geometry_type = metadata["geometry_type"]
    row.group_id = metadata["group_id"]
    row.parent_id = metadata["parent_id"]
    row.time_start = metadata["time_start"]
    row.time_end = metadata["time_end"]
    row.is_editable = metadata["is_editable"]
    row.is_deleted = False
    row.deleted_at = None
    row.data = document
    await db.flush()
    await db.refresh(row)
    document["created_at"] = row.created_at.astimezone(UTC).isoformat()
    document["updated_at"] = row.updated_at.astimezone(UTC).isoformat()
    row.data = document
    await _append_revision(
        db,
        annotation_id=row.id,
        version=int(row.version),
        snapshot=document,
    )
    outbox = await _append_outbox(
        db,
        row=row,
        event_type="annotation_upsert",
        payload={
            "annotation": document,
            "annotation_id": str(row.id),
            "scope": document.get("scope", {}),
        },
    )
    await _trim_outbox(db, owner_user_id=owner_user_id)
    await db.commit()
    await db.refresh(row)
    await _publish_outbox(outbox)
    return _serialize_row(row)


async def delete_chart_annotation(
    db: AsyncSession,
    *,
    owner_user_id: UUID,
    annotation_id: UUID,
    base_version: int,
) -> dict[str, Any]:
    row = await db.get(ChartAnnotation, annotation_id)
    if row is None or row.owner_user_id != owner_user_id:
        raise KeyError("chart_annotation_not_found")
    if int(base_version) != int(row.version):
        raise ChartAnnotationConflictError(latest=_serialize_row(row))
    if not row.is_editable:
        raise PermissionError("chart_annotation_locked")

    row.version = int(row.version) + 1
    row.is_deleted = True
    row.deleted_at = datetime.now(UTC)
    payload = dict(row.data) if isinstance(row.data, dict) else {}
    payload["version"] = int(row.version)
    lifecycle = payload.get("lifecycle") if isinstance(payload.get("lifecycle"), dict) else {}
    lifecycle["deleted_at"] = row.deleted_at.astimezone(UTC).isoformat()
    lifecycle["editable"] = bool(row.is_editable)
    payload["lifecycle"] = lifecycle
    row.data = payload
    await db.flush()
    await db.refresh(row)
    payload["created_at"] = row.created_at.astimezone(UTC).isoformat()
    payload["updated_at"] = row.updated_at.astimezone(UTC).isoformat()
    row.data = payload
    await _append_revision(
        db,
        annotation_id=row.id,
        version=int(row.version),
        snapshot=payload,
    )
    outbox = await _append_outbox(
        db,
        row=row,
        event_type="annotation_remove",
        payload={
            "annotation_id": str(row.id),
            "version": int(row.version),
            "scope": payload.get("scope", {}),
        },
    )
    await _trim_outbox(db, owner_user_id=owner_user_id)
    await db.commit()
    await db.refresh(row)
    await _publish_outbox(outbox)
    return _serialize_row(row)


async def batch_upsert_chart_annotations(
    db: AsyncSession,
    *,
    owner_user_id: UUID,
    annotations: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for item in annotations:
        candidate = _normalize_json_map(item)
        annotation_id = _normalize_uuid_string(candidate.get("id"))
        existing = await db.get(ChartAnnotation, UUID(annotation_id)) if annotation_id else None
        if existing is not None and existing.owner_user_id == owner_user_id:
            results.append(
                await update_chart_annotation(
                    db,
                    owner_user_id=owner_user_id,
                    annotation_id=existing.id,
                    base_version=int(existing.version),
                    payload=candidate,
                )
            )
        else:
            results.append(
                await create_chart_annotation(
                    db,
                    owner_user_id=owner_user_id,
                    payload=candidate,
                )
            )
    return results


def _round_price(value: float, *, pricescale: int) -> float:
    tick = 1.0 / max(pricescale, 1)
    return round(round(value / tick) * tick, 8)


def _atr_from_bars(bars: list[Any], *, period: int = 14) -> float:
    if len(bars) < 2:
        return 0.0
    trs: list[float] = []
    previous_close = float(bars[0].close)
    for bar in bars[1:]:
        high = float(bar.high)
        low = float(bar.low)
        close = float(bar.close)
        trs.append(max(high - low, abs(high - previous_close), abs(low - previous_close)))
        previous_close = close
    window = trs[-period:] if len(trs) >= period else trs
    return sum(window) / max(len(window), 1)


async def generate_demo_annotations(
    db: AsyncSession,
    *,
    owner_user_id: UUID,
    market: str,
    symbol: str,
    timeframe: str,
    scenario: str,
    reference: str,
    persist: bool,
    deployment_id: UUID | None = None,
    strategy_id: UUID | None = None,
    backtest_id: UUID | None = None,
    chart_layout_id: str | None = None,
) -> list[dict[str, Any]]:
    normalized_market = (
        chart_datafeed_route._market_prefix_to_market(str(market or "").strip())  # noqa: SLF001
        or str(market or "").strip().lower()
    )
    descriptor = await chart_datafeed_route._resolve_symbol_descriptor(  # noqa: SLF001
        db=db,
        symbol_name=symbol,
        market_hint=normalized_market,
    )
    if descriptor is None:
        return []
    to_dt = datetime.now(UTC)
    step = chart_datafeed_route.market_data_route._timeframe_step(timeframe)  # noqa: SLF001
    from_dt = to_dt - (step * 240 if step is not None else timedelta(days=14))
    bars = await chart_datafeed_route._load_chart_history(  # noqa: SLF001
        market=normalized_market,
        symbol=symbol,
        timeframe=timeframe,
        from_dt=from_dt,
        to_dt=to_dt,
        count_back=240,
    )
    if not bars:
        return []
    ref_index = len(bars) - 2 if reference == "latest_closed_bar" and len(bars) >= 2 else len(bars) // 2
    ref_bar = bars[ref_index]
    entry_price = float(ref_bar.close)
    atr = _atr_from_bars(list(bars), period=14)
    if atr <= 0:
        atr = max(entry_price * 0.015, 0.01)
    is_long = str(scenario).strip().lower() != "short"
    stop_price = entry_price - (atr * 1.2 if is_long else -atr * 1.2)
    take_price = entry_price + (atr * 2.0 if is_long else -atr * 2.0)
    entry_time = int(ref_bar.timestamp.astimezone(UTC).timestamp())
    exit_index = min(len(bars) - 1, ref_index + 24)
    exit_bar = bars[exit_index]
    exit_time = int(exit_bar.timestamp.astimezone(UTC).timestamp())
    pricescale = int(descriptor.pricescale or 100)
    entry_price = _round_price(entry_price, pricescale=pricescale)
    stop_price = _round_price(stop_price, pricescale=pricescale)
    take_price = _round_price(take_price, pricescale=pricescale)
    direction = "long" if is_long else "short"
    vendor_type = "long_position" if is_long else "short_position"
    scope: dict[str, Any] = {
        "market": descriptor.market,
        "symbol": descriptor.symbol,
        "timeframe": timeframe,
        "anchor_space": "time_price",
    }
    if deployment_id is not None:
        scope["deployment_id"] = str(deployment_id)
    if strategy_id is not None:
        scope["strategy_id"] = str(strategy_id)
    if backtest_id is not None:
        scope["backtest_id"] = str(backtest_id)
    if chart_layout_id:
        scope["chart_layout_id"] = chart_layout_id
    documents = [
        {
            "id": str(uuid4()),
            "source": {"type": "system", "source_id": "demo-seed"},
            "scope": scope,
            "semantic": {
                "kind": "signal",
                "role": "demo",
                "direction": direction,
                "status": "active",
            },
            "tool": {
                "family": "marker",
                "vendor": "tradingview",
                "vendor_type": "arrow_up" if is_long else "arrow_down",
            },
            "anchors": {"points": [{"time": entry_time, "price": entry_price}]},
            "geometry": {"type": "point"},
            "content": {"text": "DEMO SIGNAL"},
            "style": {"overrides": {"color": "#2563EB"}},
            "relations": {},
            "lifecycle": {"editable": True},
            "vendor_native": {},
        },
        {
            "id": str(uuid4()),
            "source": {"type": "system", "source_id": "demo-seed"},
            "scope": scope,
            "semantic": {
                "kind": "risk_reward",
                "role": "demo",
                "direction": direction,
                "status": "active",
            },
            "tool": {
                "family": "trading_box",
                "vendor": "tradingview",
                "vendor_type": vendor_type,
            },
            "anchors": {"points": [{"time": entry_time, "price": entry_price}]},
            "geometry": {"type": "composite"},
            "content": {},
            "style": {},
            "relations": {},
            "lifecycle": {"editable": True},
            "vendor_native": {
                "trade": {
                    "entry_price": entry_price,
                    "stop_price": stop_price,
                    "target_price": take_price,
                    "exit_time": exit_time,
                }
            },
        },
    ]
    if not persist:
        return [
            _normalize_annotation_payload(item, owner_user_id=owner_user_id)[0]
            for item in documents
        ]
    return await batch_upsert_chart_annotations(
        db,
        owner_user_id=owner_user_id,
        annotations=documents,
    )


async def append_chart_annotation_snapshot(
    db: AsyncSession,
    *,
    owner_user_id: UUID,
    filters: ChartAnnotationFilter,
) -> dict[str, Any]:
    annotations = await list_chart_annotations(
        db,
        owner_user_id=owner_user_id,
        filters=filters,
    )
    cursor = await db.scalar(
        select(func.max(ChartAnnotationOutbox.event_seq)).where(
            ChartAnnotationOutbox.owner_user_id == owner_user_id
        )
    )
    return {
        "annotations": annotations,
        "cursor": int(cursor or 0),
    }


async def project_execution_annotations_for_scope(
    db: AsyncSession,
    *,
    owner_user_id: UUID,
    filters: ChartAnnotationFilter,
) -> list[dict[str, Any]]:
    if filters.deployment_id is None or not filters.symbol or not filters.timeframe:
        return []
    deployment = await db.scalar(
        select(Deployment).where(
            Deployment.id == filters.deployment_id,
            Deployment.user_id == owner_user_id,
        )
    )
    if deployment is None:
        return []
    signal_events = (
        await db.scalars(
            select(SignalEvent).where(
                SignalEvent.deployment_id == deployment.id,
                SignalEvent.symbol == filters.symbol.upper(),
                SignalEvent.timeframe == filters.timeframe.lower(),
            )
        )
    ).all()
    orders = (
        await db.scalars(
            select(Order).where(
                Order.deployment_id == deployment.id,
                Order.symbol == filters.symbol.upper(),
            )
        )
    ).all()
    fills = (
        await db.scalars(
            select(Fill).join(Order, Fill.order_id == Order.id).where(
                Order.deployment_id == deployment.id,
                Order.symbol == filters.symbol.upper(),
            )
        )
    ).all()
    positions = (
        await db.scalars(
            select(Position).where(
                Position.deployment_id == deployment.id,
                Position.symbol == filters.symbol.upper(),
            )
        )
    ).all()
    latest_run = await db.scalar(
        select(DeploymentRun)
        .where(DeploymentRun.deployment_id == deployment.id)
        .order_by(DeploymentRun.updated_at.desc(), DeploymentRun.created_at.desc())
        .limit(1)
    )
    managed_exit_state = (
        latest_run.runtime_state.get("managed_exit")
        if latest_run is not None
        and isinstance(latest_run.runtime_state, dict)
        and isinstance(latest_run.runtime_state.get("managed_exit"), dict)
        else None
    )
    market = filters.market or "stocks"
    return build_execution_annotation_documents(
        market=market,
        symbol=filters.symbol.upper(),
        timeframe=filters.timeframe.lower(),
        deployment_id=deployment.id,
        signal_events=signal_events,
        orders=orders,
        fills=fills,
        positions=positions,
        managed_exit_state=managed_exit_state,
    )


async def project_backtest_annotations_for_scope(
    db: AsyncSession,
    *,
    filters: ChartAnnotationFilter,
) -> list[dict[str, Any]]:
    if filters.backtest_id is None or not filters.symbol or not filters.timeframe:
        return []
    job = await db.get(BacktestJob, filters.backtest_id)
    if job is None or not isinstance(job.results, dict):
        return []
    snapshots = job.results.get("snapshots")
    if not isinstance(snapshots, list):
        return []
    market = filters.market or "stocks"
    docs: list[dict[str, Any]] = []
    for item in snapshots:
        if not isinstance(item, dict):
            continue
        docs.extend(
            build_backtest_trade_annotation_documents(
                market=market,
                symbol=filters.symbol.upper(),
                timeframe=filters.timeframe.lower(),
                backtest_id=filters.backtest_id,
                trade=item.get("trade") if isinstance(item.get("trade"), dict) else {},
                trade_annotations=item.get("trade_annotations")
                if isinstance(item.get("trade_annotations"), list)
                else [],
            )
        )
    return docs
