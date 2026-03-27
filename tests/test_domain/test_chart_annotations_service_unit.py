from __future__ import annotations

from uuid import uuid4

import pytest

from packages.domain.chart_annotations.service import _normalize_annotation_payload


def _base_payload(*, family: str, vendor_type: str) -> dict:
    return {
        "source": {"type": "user_manual"},
        "scope": {
            "market": "crypto",
            "symbol": "BTCUSD",
            "timeframe": "15m",
            "chart_layout_id": "migration-lab",
        },
        "semantic": {"kind": "note", "role": "markup"},
        "tool": {
            "family": family,
            "vendor": "tradingview",
            "vendor_type": vendor_type,
        },
        "anchors": {
            "points": [
                {"time": 1717000000, "price": 71234.56},
                {"time": 1717001800, "price": 70234.12},
            ]
        },
        "geometry": {"type": "composite"},
        "vendor_native": {},
    }


def test_normalize_annotation_payload_accepts_fib_vendor_native_state_round_trip():
    payload = _base_payload(family="fib", vendor_type="Fib_Retracement")
    payload["vendor_native"] = {
        "state": {
            "id": "line-tool-fib-1",
            "type": "LineToolFibRetracement",
            "state": {"levelsStyle": {"linewidth": 2}},
        },
        "properties": {"linewidth": 2},
    }

    document, metadata = _normalize_annotation_payload(
        payload,
        owner_user_id=uuid4(),
    )

    assert metadata["tool_family"] == "fib"
    assert metadata["tool_vendor_type"] == "fib_retracement"
    assert document["vendor_native"]["state"]["id"] == "line-tool-fib-1"
    assert document["vendor_native"]["state"]["state"]["levelsStyle"]["linewidth"] == 2


@pytest.mark.parametrize(
    ("family", "vendor_type"),
    [
        ("fib", "gannbox"),
        ("gann", "fib_retracement"),
    ],
)
def test_normalize_annotation_payload_rejects_family_vendor_type_mismatch(
    family: str,
    vendor_type: str,
):
    payload = _base_payload(family=family, vendor_type=vendor_type)

    with pytest.raises(ValueError, match="tool\\.vendor_type"):
        _normalize_annotation_payload(payload, owner_user_id=uuid4())


def test_normalize_annotation_payload_rejects_non_object_vendor_native_state():
    payload = _base_payload(family="gann", vendor_type="gannbox")
    payload["vendor_native"] = {"state": ["not", "an", "object"]}

    with pytest.raises(ValueError, match="vendor_native\\.state"):
        _normalize_annotation_payload(payload, owner_user_id=uuid4())


def test_normalize_annotation_payload_accepts_legacy_fib_without_native_state():
    payload = _base_payload(family="fib", vendor_type="fib_channel")

    document, metadata = _normalize_annotation_payload(
        payload,
        owner_user_id=uuid4(),
    )

    assert metadata["tool_family"] == "fib"
    assert metadata["tool_vendor_type"] == "fib_channel"
    assert document["vendor_native"] == {}


def test_normalize_annotation_payload_accepts_pattern_vendor_native_state_round_trip():
    payload = _base_payload(
        family="pattern",
        vendor_type="XABCD_Pattern",
    )
    payload["anchors"]["points"].extend(
        [
            {"time": 1717003600, "price": 70654.44},
            {"time": 1717005400, "price": 69980.11},
            {"time": 1717007200, "price": 70420.32},
        ]
    )
    payload["vendor_native"] = {
        "state": {
            "id": "line-tool-pattern-1",
            "type": "LineToolXABCDPattern",
            "state": {"linewidth": 2},
        },
        "properties": {"linewidth": 2},
    }

    document, metadata = _normalize_annotation_payload(
        payload,
        owner_user_id=uuid4(),
    )

    assert metadata["tool_family"] == "pattern"
    assert metadata["tool_vendor_type"] == "xabcd_pattern"
    assert document["vendor_native"]["state"]["type"] == "LineToolXABCDPattern"


def test_normalize_annotation_payload_rejects_non_object_pattern_vendor_native_state():
    payload = _base_payload(family="pattern", vendor_type="triangle_pattern")
    payload["vendor_native"] = {"state": "bad"}

    with pytest.raises(ValueError, match="vendor_native\\.state"):
        _normalize_annotation_payload(payload, owner_user_id=uuid4())


def test_normalize_annotation_payload_accepts_legacy_pattern_without_native_state():
    payload = _base_payload(
        family="pattern",
        vendor_type="head_and_shoulders",
    )
    payload["anchors"]["points"].extend(
        [
            {"time": 1717003600, "price": 70654.44},
            {"time": 1717005400, "price": 69980.11},
            {"time": 1717007200, "price": 70420.32},
        ]
    )

    document, metadata = _normalize_annotation_payload(
        payload,
        owner_user_id=uuid4(),
    )

    assert metadata["tool_family"] == "pattern"
    assert metadata["tool_vendor_type"] == "head_and_shoulders"
    assert document["vendor_native"] == {}


def test_normalize_annotation_payload_accepts_elliott_vendor_native_state_round_trip():
    payload = _base_payload(
        family="pattern",
        vendor_type="Elliott_Impulse_Wave",
    )
    payload["anchors"]["points"].extend(
        [
            {"time": 1717003600, "price": 70654.44},
            {"time": 1717005400, "price": 69980.11},
            {"time": 1717007200, "price": 70420.32},
        ]
    )
    payload["vendor_native"] = {
        "state": {
            "id": "line-tool-elliott-1",
            "type": "LineToolElliottImpulseWave",
            "state": {"linewidth": 2},
        },
        "properties": {"linewidth": 2},
    }

    document, metadata = _normalize_annotation_payload(
        payload,
        owner_user_id=uuid4(),
    )

    assert metadata["tool_family"] == "pattern"
    assert metadata["tool_vendor_type"] == "elliott_impulse_wave"
    assert document["vendor_native"]["state"]["type"] == "LineToolElliottImpulseWave"


def test_normalize_annotation_payload_accepts_profile_vendor_native_state_round_trip():
    payload = _base_payload(family="profile", vendor_type="Anchored_VWAP")
    payload["anchors"] = {
        "points": [
            {"time": 1717000000, "price": 71234.56},
        ]
    }
    payload["vendor_native"] = {
        "state": {
            "id": "line-tool-profile-1",
            "type": "LineToolAnchoredVWAP",
            "state": {"bandLineColor": "#2563EB"},
        },
        "properties": {"linecolor": "#2563EB"},
    }

    document, metadata = _normalize_annotation_payload(
        payload,
        owner_user_id=uuid4(),
    )

    assert metadata["tool_family"] == "profile"
    assert metadata["tool_vendor_type"] == "anchored_vwap"
    assert document["vendor_native"]["state"]["id"] == "line-tool-profile-1"
    assert document["vendor_native"]["state"]["type"] == "LineToolAnchoredVWAP"


@pytest.mark.parametrize(
    ("family", "vendor_type"),
    [
        ("profile", "fib_retracement"),
        ("profile", "gannbox"),
    ],
)
def test_normalize_annotation_payload_rejects_profile_vendor_type_mismatch(
    family: str,
    vendor_type: str,
):
    payload = _base_payload(family=family, vendor_type=vendor_type)

    with pytest.raises(ValueError, match="tool\\.vendor_type"):
        _normalize_annotation_payload(payload, owner_user_id=uuid4())


def test_normalize_annotation_payload_rejects_non_object_profile_vendor_native_state():
    payload = _base_payload(family="profile", vendor_type="anchored_vwap")
    payload["anchors"] = {
        "points": [
            {"time": 1717000000, "price": 71234.56},
        ]
    }
    payload["vendor_native"] = {"state": "bad"}

    with pytest.raises(ValueError, match="vendor_native\\.state"):
        _normalize_annotation_payload(payload, owner_user_id=uuid4())


def test_normalize_annotation_payload_accepts_legacy_profile_without_native_state():
    payload = _base_payload(
        family="profile",
        vendor_type="fixed_range_volume_profile",
    )

    document, metadata = _normalize_annotation_payload(
        payload,
        owner_user_id=uuid4(),
    )

    assert metadata["tool_family"] == "profile"
    assert metadata["tool_vendor_type"] == "fixed_range_volume_profile"
    assert document["vendor_native"] == {}


def test_normalize_annotation_payload_accepts_fork_vendor_native_state_round_trip():
    payload = _base_payload(
        family="fork",
        vendor_type="Schiff_Pitchfork_Modified",
    )
    payload["anchors"]["points"].append(
        {"time": 1717003600, "price": 70654.44},
    )
    payload["vendor_native"] = {
        "state": {
            "id": "line-tool-fork-1",
            "type": "LineToolSchiffPitchfork2",
            "state": {"style": 2},
        },
        "properties": {"linewidth": 2},
    }

    document, metadata = _normalize_annotation_payload(
        payload,
        owner_user_id=uuid4(),
    )

    assert metadata["tool_family"] == "fork"
    assert metadata["tool_vendor_type"] == "schiff_pitchfork_modified"
    assert document["vendor_native"]["state"]["type"] == "LineToolSchiffPitchfork2"


@pytest.mark.parametrize(
    ("family", "vendor_type"),
    [
        ("text", "pitchfork"),
        ("line", "callout"),
        ("channel", "pitchfork"),
        ("shape", "pitchfork"),
        ("brush", "pitchfork"),
        ("pattern", "pitchfork"),
        ("fork", "anchored_vwap"),
        ("measurement", "pitchfork"),
        ("cycle", "price_range"),
        ("forecast", "pitchfork"),
        ("trading_box", "pitchfork"),
        ("media", "pitchfork"),
        ("table", "pitchfork"),
    ],
)
def test_normalize_annotation_payload_rejects_additional_family_vendor_type_mismatch(
    family: str,
    vendor_type: str,
):
    payload = _base_payload(family=family, vendor_type=vendor_type)

    with pytest.raises(ValueError, match="tool\\.vendor_type"):
        _normalize_annotation_payload(payload, owner_user_id=uuid4())


def test_normalize_annotation_payload_rejects_non_object_cycle_vendor_native_state():
    payload = _base_payload(family="cycle", vendor_type="time_cycles")
    payload["vendor_native"] = {"state": ["bad"]}

    with pytest.raises(ValueError, match="vendor_native\\.state"):
        _normalize_annotation_payload(payload, owner_user_id=uuid4())


def test_normalize_annotation_payload_accepts_brush_vendor_native_state_round_trip():
    payload = _base_payload(family="brush", vendor_type="Brush")
    payload["geometry"] = {}
    payload["anchors"]["points"].append(
        {"time": 1717003600, "price": 70654.44},
    )
    payload["vendor_native"] = {
        "state": {
            "id": "line-tool-brush-1",
            "type": "LineToolBrush",
            "state": {"color": "#F59E0B"},
        },
        "properties": {"linewidth": 3},
    }

    document, metadata = _normalize_annotation_payload(
        payload,
        owner_user_id=uuid4(),
    )

    assert metadata["tool_family"] == "brush"
    assert metadata["tool_vendor_type"] == "brush"
    assert document["geometry"]["type"] == "path"
    assert document["vendor_native"]["state"]["type"] == "LineToolBrush"
    assert document["content"]["stroke"] == {"point_count": 3}


def test_normalize_annotation_payload_rejects_non_object_brush_vendor_native_state():
    payload = _base_payload(family="brush", vendor_type="highlighter")
    payload["vendor_native"] = {"state": "bad"}

    with pytest.raises(ValueError, match="vendor_native\\.state"):
        _normalize_annotation_payload(payload, owner_user_id=uuid4())


def test_normalize_annotation_payload_accepts_legacy_measurement_without_native_state():
    payload = _base_payload(
        family="measurement",
        vendor_type="date_range",
    )
    payload["geometry"] = {"type": "range"}

    document, metadata = _normalize_annotation_payload(
        payload,
        owner_user_id=uuid4(),
    )

    assert metadata["tool_family"] == "measurement"
    assert metadata["tool_vendor_type"] == "date_range"
    assert document["vendor_native"] == {}


def test_normalize_annotation_payload_accepts_forecast_vendor_native_state_round_trip():
    payload = _base_payload(
        family="forecast",
        vendor_type="Projection",
    )
    payload["vendor_native"] = {
        "state": {
            "id": "line-tool-forecast-1",
            "type": "LineToolProjection",
            "state": {"linewidth": 2},
        },
        "properties": {"linewidth": 2},
    }

    document, metadata = _normalize_annotation_payload(
        payload,
        owner_user_id=uuid4(),
    )

    assert metadata["tool_family"] == "forecast"
    assert metadata["tool_vendor_type"] == "projection"
    assert document["vendor_native"]["state"]["type"] == "LineToolProjection"


def test_normalize_annotation_payload_accepts_text_extension_without_native_state():
    payload = _base_payload(
        family="text",
        vendor_type="callout",
    )
    payload["anchors"] = {
        "points": [
            {"time": 1717000000, "price": 71234.56},
            {"time": 1717001800, "price": 70654.44},
        ]
    }
    payload["content"] = {"text": "Callout 1"}

    document, metadata = _normalize_annotation_payload(
        payload,
        owner_user_id=uuid4(),
    )

    assert metadata["tool_family"] == "text"
    assert metadata["tool_vendor_type"] == "callout"
    assert document["content"]["text"] == "Callout 1"
    assert document["vendor_native"] == {}


def test_normalize_annotation_payload_accepts_advanced_line_without_native_state():
    payload = _base_payload(
        family="line",
        vendor_type="regression_trend",
    )
    payload["geometry"] = {"type": "polyline"}

    document, metadata = _normalize_annotation_payload(
        payload,
        owner_user_id=uuid4(),
    )

    assert metadata["tool_family"] == "line"
    assert metadata["tool_vendor_type"] == "regression_trend"
    assert document["geometry"]["type"] == "polyline"


def test_normalize_annotation_payload_accepts_channel_vendor_native_state_round_trip():
    payload = _base_payload(
        family="channel",
        vendor_type="Parallel_Channel",
    )
    payload["anchors"]["points"].append(
        {"time": 1717003600, "price": 70654.44},
    )
    payload["vendor_native"] = {
        "state": {
            "id": "line-tool-channel-1",
            "type": "LineToolParallelChannel",
            "state": {"showMidline": True},
        },
        "properties": {"showMidline": True},
    }

    document, metadata = _normalize_annotation_payload(
        payload,
        owner_user_id=uuid4(),
    )

    assert metadata["tool_family"] == "channel"
    assert metadata["tool_vendor_type"] == "parallel_channel"
    assert document["vendor_native"]["state"]["type"] == "LineToolParallelChannel"


def test_normalize_annotation_payload_accepts_shape_path_geometry_without_native_state():
    payload = _base_payload(
        family="shape",
        vendor_type="arc",
    )
    payload["anchors"]["points"].append(
        {"time": 1717003600, "price": 70654.44},
    )
    payload["geometry"] = {}

    document, metadata = _normalize_annotation_payload(
        payload,
        owner_user_id=uuid4(),
    )

    assert metadata["tool_family"] == "shape"
    assert metadata["tool_vendor_type"] == "arc"
    assert document["geometry"]["type"] == "path"
    assert document["vendor_native"] == {}


def test_normalize_annotation_payload_accepts_polygon_shape_without_native_state():
    payload = _base_payload(
        family="shape",
        vendor_type="rotated_rectangle",
    )
    payload["geometry"] = {}

    document, metadata = _normalize_annotation_payload(
        payload,
        owner_user_id=uuid4(),
    )

    assert metadata["tool_family"] == "shape"
    assert metadata["tool_vendor_type"] == "rotated_rectangle"
    assert document["geometry"]["type"] == "polygon"


def test_normalize_annotation_payload_accepts_trading_box_vendor_native_state_round_trip():
    payload = _base_payload(
        family="trading_box",
        vendor_type="Long_Position",
    )
    payload["vendor_native"] = {
        "state": {
            "id": "line-tool-trading-1",
            "type": "LineToolRiskRewardLong",
            "state": {"risk": 1200},
        },
        "properties": {"linecolor": "#16A34A"},
    }

    document, metadata = _normalize_annotation_payload(
        payload,
        owner_user_id=uuid4(),
    )

    assert metadata["tool_family"] == "trading_box"
    assert metadata["tool_vendor_type"] == "long_position"
    assert document["vendor_native"]["state"]["type"] == "LineToolRiskRewardLong"
    assert document["content"]["trade"]["direction"] == "long"
    assert document["content"]["trade"]["risk"] == 1200.0


def test_normalize_annotation_payload_derives_trading_box_trade_summary_from_native_state():
    payload = _base_payload(
        family="trading_box",
        vendor_type="long_position",
    )
    payload["anchors"]["points"] = [
        {"time": 1717000000, "price": 71234.56},
        {"time": 1717003600, "price": 71234.56},
    ]
    payload["vendor_native"] = {
        "state": {
            "id": "line-tool-trading-2",
            "state": {
                "type": "LineToolRiskRewardLong",
                "state": {
                    "qty": 1.5,
                    "risk": 800,
                    "lotSize": 2,
                    "accountSize": 25000,
                    "currency": "USD",
                    "riskDisplayMode": "currency",
                    "linecolor": "#16A34A",
                    "textcolor": "#0F172A",
                    "stopLevel": 70500.0,
                    "profitLevel": 72600.0,
                    "amountStop": -800.0,
                    "amountTarget": 1600.0,
                    "compact": True,
                    "alwaysShowStats": True,
                    "showPriceLabels": True,
                },
            },
        },
        "properties": {"linewidth": 2},
    }

    document, _ = _normalize_annotation_payload(
        payload,
        owner_user_id=uuid4(),
    )

    assert document["vendor_native"]["trade"] == {
        "direction": "long",
        "entry_time": 1717000000,
        "exit_time": 1717003600,
        "entry_price": 71234.56,
        "stop_price": 70500.0,
        "target_price": 72600.0,
        "qty": 1.5,
        "risk": 800.0,
        "account_size": 25000.0,
        "lot_size": 2.0,
        "amount_stop": -800.0,
        "amount_target": 1600.0,
        "currency": "USD",
        "risk_display_mode": "currency",
        "line_color": "#16A34A",
        "text_color": "#0F172A",
        "compact": True,
        "always_show_stats": True,
        "show_price_labels": True,
        "risk_reward_ratio": 1.8589,
    }
    assert document["content"]["trade"] == document["vendor_native"]["trade"]


def test_normalize_annotation_payload_rejects_non_object_forecast_vendor_native_state():
    payload = _base_payload(family="forecast", vendor_type="forecast")
    payload["vendor_native"] = {"state": ["bad"]}

    with pytest.raises(ValueError, match="vendor_native\\.state"):
        _normalize_annotation_payload(payload, owner_user_id=uuid4())


def test_normalize_annotation_payload_rejects_non_object_channel_vendor_native_state():
    payload = _base_payload(family="channel", vendor_type="flat_bottom")
    payload["anchors"]["points"].append(
        {"time": 1717003600, "price": 70654.44},
    )
    payload["vendor_native"] = {"state": "bad"}

    with pytest.raises(ValueError, match="vendor_native\\.state"):
        _normalize_annotation_payload(payload, owner_user_id=uuid4())


def test_normalize_annotation_payload_rejects_non_object_trading_box_vendor_native_state():
    payload = _base_payload(family="trading_box", vendor_type="short_position")
    payload["vendor_native"] = {"state": "bad"}

    with pytest.raises(ValueError, match="vendor_native\\.state"):
        _normalize_annotation_payload(payload, owner_user_id=uuid4())


def test_normalize_annotation_payload_accepts_legacy_forecast_without_native_state():
    payload = _base_payload(
        family="forecast",
        vendor_type="bars_pattern",
    )

    document, metadata = _normalize_annotation_payload(
        payload,
        owner_user_id=uuid4(),
    )

    assert metadata["tool_family"] == "forecast"
    assert metadata["tool_vendor_type"] == "bars_pattern"
    assert document["vendor_native"] == {}


def test_normalize_annotation_payload_accepts_legacy_channel_without_native_state():
    payload = _base_payload(
        family="channel",
        vendor_type="disjoint_angle",
    )
    payload["anchors"]["points"].append(
        {"time": 1717003600, "price": 70654.44},
    )

    document, metadata = _normalize_annotation_payload(
        payload,
        owner_user_id=uuid4(),
    )

    assert metadata["tool_family"] == "channel"
    assert metadata["tool_vendor_type"] == "disjoint_angle"
    assert document["vendor_native"] == {}


def test_normalize_annotation_payload_accepts_legacy_trading_box_without_native_state():
    payload = _base_payload(
        family="trading_box",
        vendor_type="short_position",
    )
    payload["anchors"]["points"] = [
        {"time": 1717000000, "price": 71234.56},
    ]
    payload["vendor_native"] = {
        "trade": {
            "entry_time": 1717000000,
            "exit_time": 1717003600,
            "entry_price": 71234.56,
            "stop_price": 71999.0,
            "target_price": 70123.0,
            "qty": 2,
        }
    }

    document, metadata = _normalize_annotation_payload(
        payload,
        owner_user_id=uuid4(),
    )

    assert metadata["tool_family"] == "trading_box"
    assert metadata["tool_vendor_type"] == "short_position"
    assert document["vendor_native"]["trade"]["direction"] == "short"
    assert document["content"]["trade"]["qty"] == 2.0
    assert document["anchors"]["points"] == [
        {"time": 1717000000, "price": 71234.56},
        {"time": 1717003600, "price": 71234.56},
    ]
    assert document["vendor_native"]["properties"]["stopLevel"] == 71999.0
    assert document["vendor_native"]["properties"]["profitLevel"] == 70123.0
    assert document["vendor_native"]["properties"]["qty"] == 2.0
    assert document["vendor_native"]["properties"]["linecolor"] == "#DC2626"


def test_normalize_annotation_payload_normalizes_legacy_risk_reward_vendor_type():
    payload = _base_payload(
        family="trading_box",
        vendor_type="risk_reward",
    )
    payload["semantic"]["direction"] = "short"
    payload["vendor_native"] = {
        "trade": {
            "entry_time": 1717000000,
            "exit_time": 1717003600,
            "entry_price": 71234.56,
            "stop_price": 71999.0,
            "target_price": 70123.0,
        }
    }

    document, metadata = _normalize_annotation_payload(
        payload,
        owner_user_id=uuid4(),
    )

    assert metadata["tool_family"] == "trading_box"
    assert metadata["tool_vendor_type"] == "short_position"
    assert document["tool"]["vendor_type"] == "short_position"
    assert document["vendor_native"]["trade"]["direction"] == "short"
    assert document["vendor_native"]["properties"]["linecolor"] == "#DC2626"


def test_normalize_annotation_payload_accepts_media_without_native_state():
    payload = _base_payload(
        family="media",
        vendor_type="emoji",
    )
    payload["geometry"] = {}
    payload["anchors"] = {
        "points": [
            {"time": 1717000000, "price": 71234.56},
        ]
    }
    payload["content"] = {"emoji": "🚀"}

    document, metadata = _normalize_annotation_payload(
        payload,
        owner_user_id=uuid4(),
    )

    assert metadata["tool_family"] == "media"
    assert metadata["tool_vendor_type"] == "emoji"
    assert document["geometry"]["type"] == "point"
    assert document["content"]["emoji"] == "🚀"
    assert document["vendor_native"] == {}


def test_normalize_annotation_payload_derives_media_content_summary_from_native_state():
    payload = _base_payload(
        family="media",
        vendor_type="emoji",
    )
    payload["geometry"] = {}
    payload["anchors"] = {
        "points": [
            {"time": 1717000000, "price": 71234.56},
        ]
    }
    payload["content"] = {"emoji": "🚀"}
    payload["vendor_native"] = {
        "state": {
            "id": "line-tool-emoji-1",
            "type": "LineToolEmoji",
            "state": {
                "emoji": "😀",
                "size": 72,
                "angle": 1.57,
            },
        }
    }

    document, metadata = _normalize_annotation_payload(
        payload,
        owner_user_id=uuid4(),
    )

    assert metadata["tool_family"] == "media"
    assert metadata["tool_vendor_type"] == "emoji"
    assert document["content"]["emoji"] == "😀"
    assert document["content"]["media"] == {"size": 72.0, "angle": 1.57}


def test_normalize_annotation_payload_accepts_table_without_native_state():
    payload = _base_payload(
        family="table",
        vendor_type="table",
    )
    payload["geometry"] = {}
    payload["anchors"] = {
        "points": [
            {"time": 1717000000, "price": 71234.56},
        ]
    }

    document, metadata = _normalize_annotation_payload(
        payload,
        owner_user_id=uuid4(),
    )

    assert metadata["tool_family"] == "table"
    assert metadata["tool_vendor_type"] == "table"
    assert document["geometry"]["type"] == "point"
    assert document["vendor_native"] == {}


def test_normalize_annotation_payload_derives_table_content_summary_from_native_state():
    payload = _base_payload(
        family="table",
        vendor_type="table",
    )
    payload["geometry"] = {}
    payload["anchors"] = {
        "points": [
            {"time": 1717000000, "price": 71234.56},
        ]
    }
    payload["vendor_native"] = {
        "state": {
            "id": "line-tool-table-1",
            "type": "LineToolTable",
            "state": {
                "title": "Trade Plan",
                "rowsCount": 2,
                "colsCount": 3,
                "cells": [
                    ["Entry", "Stop", "Target"],
                    ["100", "95", "110"],
                ],
            },
        }
    }

    document, metadata = _normalize_annotation_payload(
        payload,
        owner_user_id=uuid4(),
    )

    assert metadata["tool_family"] == "table"
    assert metadata["tool_vendor_type"] == "table"
    assert document["content"]["table"] == {
        "rows": 2,
        "cols": 3,
        "title": "Trade Plan",
        "cells_preview": [
            ["Entry", "Stop", "Target"],
            ["100", "95", "110"],
        ],
    }
