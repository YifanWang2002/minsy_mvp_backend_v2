from __future__ import annotations

import pandas as pd
import pytest

from src.engine.backtest.condition import (
    compile_condition,
    evaluate_compiled_condition_at,
    evaluate_condition_at,
    evaluate_condition_series,
)


@pytest.fixture()
def condition_frame() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=12, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "open": [100, 101, 102, 103, 102, 101, 100, 99, 100, 101, 102, 103],
            "close": [101, 102, 101, 104, 103, 100, 99, 100, 102, 100, 103, 104],
            "fast": [1.0, 1.2, 1.4, 1.1, 0.9, 1.0, 1.3, 1.5, 1.1, 0.8, 1.0, 1.4],
            "slow": [1.1, 1.1, 1.2, 1.2, 1.0, 0.9, 1.0, 1.2, 1.2, 1.0, 0.9, 1.0],
            "signal_flag": [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
            "volume_zero": [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        },
        index=index,
    )


def _assert_series_matches_scalar(
    condition: dict[str, object],
    *,
    frame: pd.DataFrame,
) -> None:
    series = evaluate_condition_series(condition, frame=frame)
    compiled = compile_condition(condition)

    assert len(series) == len(frame)
    for bar_index in range(len(frame)):
        scalar = evaluate_condition_at(condition, frame=frame, bar_index=bar_index)
        compiled_scalar = evaluate_compiled_condition_at(
            compiled,
            frame=frame,
            bar_index=bar_index,
        )
        assert bool(series[bar_index]) == scalar
        assert compiled_scalar == scalar


@pytest.mark.parametrize(
    "condition",
    [
        {"cmp": {"left": {"ref": "close"}, "op": "gt", "right": {"ref": "open"}}},
        {"cmp": {"left": {"ref": "close", "offset": 1}, "op": "gt", "right": 100}},
        {"cross": {"a": {"ref": "fast"}, "op": "cross_above", "b": {"ref": "slow"}}},
        {
            "all": [
                {"cmp": {"left": {"ref": "close"}, "op": "gt", "right": 100}},
                {"not": {"ref": "volume_zero"}},
            ]
        },
        {
            "any": [
                {"ref": "signal_flag"},
                {"cross": {"a": {"ref": "fast"}, "op": "cross_below", "b": {"ref": "slow"}}},
            ]
        },
        {"not": {"cmp": {"left": {"ref": "open"}, "op": "lt", "right": 100}}},
        {"not": "invalid_child"},
    ],
)
def test_vectorized_condition_matches_scalar_semantics(
    condition: dict[str, object],
    condition_frame: pd.DataFrame,
) -> None:
    _assert_series_matches_scalar(condition, frame=condition_frame)
