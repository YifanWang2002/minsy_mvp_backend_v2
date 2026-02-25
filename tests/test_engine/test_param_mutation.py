from __future__ import annotations

from src.engine.strategy.param_mutation import apply_param_values, list_tunable_params


def _payload() -> dict:
    return {
        "factors": {
            "ema_fast": {"type": "ema", "params": {"length": 12}},
            "ema_slow": {"type": "ema", "params": {"length": 26}},
        }
    }


def test_list_tunable_params_extracts_numeric_fields() -> None:
    params = list_tunable_params(_payload())
    keys = [item.key for item in params]
    assert "ema_fast.length" in keys
    assert "ema_slow.length" in keys


def test_apply_param_values_supports_dot_and_json_path_keys() -> None:
    mutated = apply_param_values(
        _payload(),
        values={
            "ema_fast.length": 15,
            "/factors/ema_slow/params/length": 30,
        },
    )
    assert mutated["factors"]["ema_fast"]["params"]["length"] == 15
    assert mutated["factors"]["ema_slow"]["params"]["length"] == 30
