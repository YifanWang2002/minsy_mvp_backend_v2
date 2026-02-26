from __future__ import annotations

from packages.domain.stress.optimization.pareto import build_metric_points, pareto_front_indices
from packages.domain.stress.param_sensitivity import scan_param_sensitivity
from packages.domain.strategy import EXAMPLE_PATH, load_strategy_payload
from packages.domain.strategy.param_mutation import list_tunable_params


def test_000_accessibility_pareto_front_indices() -> None:
    rows = [
        {"objective_values": {"ret": 1.0, "dd": 0.8}},
        {"objective_values": {"ret": 0.5, "dd": 0.9}},
        {"objective_values": {"ret": 1.2, "dd": 0.7}},
    ]
    front = pareto_front_indices(rows=rows, objective_keys=["ret", "dd"])
    assert front == {0, 1, 2}


def test_010_build_metric_points_shapes_output() -> None:
    rows = [
        {"trial_no": 1, "metrics": {"return": 10.0, "sharpe": 1.2}, "params": {"a": 1}},
        {"trial_no": 2, "metrics": {"return": 8.0, "sharpe": 1.4}, "params": {"a": 2}},
    ]
    points = build_metric_points(rows=rows, x_metric="return", y_metric="sharpe")
    assert len(points) == 2
    assert points[0]["trial_no"] == 1
    assert "x" in points[0] and "y" in points[0]


def test_020_param_sensitivity_scan_returns_variants_and_rank() -> None:
    payload = load_strategy_payload(EXAMPLE_PATH)
    params = list_tunable_params(payload)[:2]
    assert params

    def _evaluator(mutated_payload: dict[str, object]) -> dict[str, float]:
        period = float(mutated_payload["factors"]["ema_9"]["params"]["period"])
        return {"total_return_pct": 100.0 - abs(period - 9.0)}

    variants, fragile_rank, stability_score = scan_param_sensitivity(
        base_payload=payload,
        params=params,
        scan_pct=20.0,
        steps_per_side=2,
        evaluator=_evaluator,
    )

    assert variants
    assert fragile_rank
    assert 0.0 <= stability_score <= 100.0
