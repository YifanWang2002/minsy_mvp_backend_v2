from __future__ import annotations

from src.engine.optimization.pareto import build_metric_points, pareto_front_indices


ROWS = [
    {
        "trial_no": 1,
        "objective_values": {"max_return": 10.0, "min_drawdown": -8.0},
        "metrics": {"total_return_pct": 10.0, "max_drawdown_pct": 8.0},
        "params": {"a": 1},
    },
    {
        "trial_no": 2,
        "objective_values": {"max_return": 9.0, "min_drawdown": -5.0},
        "metrics": {"total_return_pct": 9.0, "max_drawdown_pct": 5.0},
        "params": {"a": 2},
    },
    {
        "trial_no": 3,
        "objective_values": {"max_return": 6.0, "min_drawdown": -12.0},
        "metrics": {"total_return_pct": 6.0, "max_drawdown_pct": 12.0},
        "params": {"a": 3},
    },
]


def test_pareto_front_indices_excludes_dominated_points() -> None:
    front = pareto_front_indices(rows=ROWS, objective_keys=["max_return", "min_drawdown"])
    assert front == {0, 1}


def test_build_metric_points_shapes() -> None:
    points = build_metric_points(rows=ROWS, x_metric="total_return_pct", y_metric="max_drawdown_pct")
    assert len(points) == 3
    assert points[0]["trial_no"] == 1
