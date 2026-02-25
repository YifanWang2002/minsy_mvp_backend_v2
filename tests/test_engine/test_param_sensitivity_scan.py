from __future__ import annotations

from src.engine.strategy.param_mutation import list_tunable_params
from src.engine.stress.param_sensitivity import scan_param_sensitivity


PAYLOAD = {
    "factors": {
        "ema_fast": {"type": "ema", "params": {"length": 10}},
        "ema_slow": {"type": "ema", "params": {"length": 30}},
    }
}


def _evaluator(payload: dict) -> dict:
    fast = payload["factors"]["ema_fast"]["params"]["length"]
    slow = payload["factors"]["ema_slow"]["params"]["length"]
    diff = slow - fast
    return {
        "total_return_pct": float(diff),
        "max_drawdown_pct": float(abs(diff) / 2.0),
        "stability_score": float(100.0 - abs(diff)),
    }


def test_scan_param_sensitivity_outputs_variants_and_rank() -> None:
    params = list_tunable_params(PAYLOAD)
    variants, fragile_rank, stability_score = scan_param_sensitivity(
        base_payload=PAYLOAD,
        params=params,
        scan_pct=10.0,
        steps_per_side=2,
        evaluator=_evaluator,
    )

    assert variants
    assert fragile_rank
    assert stability_score >= 0.0
