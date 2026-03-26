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
        ("fork", "anchored_vwap"),
        ("measurement", "pitchfork"),
        ("cycle", "price_range"),
        ("forecast", "pitchfork"),
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


def test_normalize_annotation_payload_rejects_non_object_forecast_vendor_native_state():
    payload = _base_payload(family="forecast", vendor_type="forecast")
    payload["vendor_native"] = {"state": ["bad"]}

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
