from __future__ import annotations

from copy import deepcopy

from packages.domain.strategy import EXAMPLE_PATH, load_strategy_payload
from packages.domain.strategy.param_mutation import apply_param_values, list_tunable_params


def test_000_accessibility_list_tunable_params() -> None:
    payload = load_strategy_payload(EXAMPLE_PATH)
    params = list_tunable_params(payload)
    assert params
    assert all(item.key for item in params)


def test_010_apply_param_values_supports_dot_and_json_pointer_keys() -> None:
    payload = load_strategy_payload(EXAMPLE_PATH)
    base_period = int(payload["factors"]["ema_9"]["params"]["period"])

    mutated_dot = apply_param_values(
        payload,
        values={"ema_9.period": float(base_period + 3)},
    )
    assert mutated_dot["factors"]["ema_9"]["params"]["period"] == base_period + 3

    mutated_pointer = apply_param_values(
        payload,
        values={"/factors/ema_9/params/period": float(base_period + 5)},
    )
    assert mutated_pointer["factors"]["ema_9"]["params"]["period"] == base_period + 5


def test_020_apply_param_values_keeps_original_payload_immutable() -> None:
    payload = load_strategy_payload(EXAMPLE_PATH)
    original = deepcopy(payload)
    _ = apply_param_values(payload, values={"ema_9.period": 99.0})
    assert payload == original
