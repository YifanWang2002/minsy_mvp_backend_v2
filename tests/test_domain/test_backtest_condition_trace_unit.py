"""Unit tests for condition trace evaluation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd

from packages.domain.backtest.condition import evaluate_condition_trace_at


def _build_frame() -> pd.DataFrame:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    index = [start + timedelta(minutes=i) for i in range(5)]
    return pd.DataFrame(
        {
            "close": [100.0, 101.0, 102.0, 101.5, 103.0],
            "ema_fast": [99.5, 100.5, 101.5, 101.7, 102.0],
            "flag": [0, 1, 1, 0, 1],
        },
        index=index,
    )


def test_condition_trace_cmp_and_unmet_nodes() -> None:
    frame = _build_frame()
    condition = {
        "cmp": {
            "left": {"ref": "close"},
            "op": "lt",
            "right": {"ref": "ema_fast"},
        }
    }

    traced = evaluate_condition_trace_at(condition, frame=frame, bar_index=2)

    assert traced["hit"] is False
    assert traced["tree"]["node_type"] == "cmp"
    assert traced["tree"]["reason_code"] == "cmp_false"
    assert traced["unmet_nodes"][0]["node_type"] == "cmp"


def test_condition_trace_cross_and_not_nodes() -> None:
    frame = _build_frame()
    condition = {
        "not": {
            "cross": {
                "a": {"ref": "close"},
                "op": "cross_above",
                "b": {"ref": "ema_fast"},
            }
        }
    }

    traced = evaluate_condition_trace_at(condition, frame=frame, bar_index=2)

    assert traced["tree"]["node_type"] == "not"
    child = traced["tree"]["child"]
    assert child["node_type"] == "cross"
    assert child["path"].endswith(".not")


def test_condition_trace_all_any_ref_paths() -> None:
    frame = _build_frame()
    condition = {
        "all": [
            {
                "cmp": {
                    "left": {"ref": "close"},
                    "op": "gte",
                    "right": 101,
                }
            },
            {
                "any": [
                    {"ref": "flag"},
                    {
                        "cmp": {
                            "left": {"ref": "ema_fast"},
                            "op": "gt",
                            "right": 200,
                        }
                    },
                ]
            },
        ]
    }

    traced = evaluate_condition_trace_at(condition, frame=frame, bar_index=1)

    assert traced["hit"] is True
    assert traced["tree"]["node_type"] == "all"
    assert len(traced["tree"]["children"]) == 2
