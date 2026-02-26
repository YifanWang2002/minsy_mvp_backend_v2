"""Search-space normalization for optimization jobs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from packages.domain.strategy.param_mutation import TunableParam


@dataclass(frozen=True, slots=True)
class SearchDimension:
    """One tunable optimization dimension."""

    key: str
    dtype: str
    min_value: float
    max_value: float
    step: float


def build_search_space(
    *,
    raw_search_space: dict[str, Any] | None,
    tunable_params: list[TunableParam],
) -> list[SearchDimension]:
    """Build a validated search space from request or defaults."""

    tunable_map = {item.key: item for item in tunable_params}
    if not tunable_map:
        raise ValueError("No tunable params found in strategy")

    raw = raw_search_space if isinstance(raw_search_space, dict) else {}
    output: list[SearchDimension] = []

    if not raw:
        for item in tunable_params:
            center = float(item.current_value)
            span = abs(center) * 0.2 or 1.0
            min_value = center - span
            max_value = center + span
            step = span / 4.0
            output.append(
                SearchDimension(
                    key=item.key,
                    dtype="float",
                    min_value=min_value,
                    max_value=max_value,
                    step=max(step, 1e-6),
                )
            )
        return output

    for key, spec in raw.items():
        if key not in tunable_map:
            continue
        if not isinstance(spec, dict):
            continue

        min_value = float(spec.get("min", tunable_map[key].current_value))
        max_value = float(spec.get("max", tunable_map[key].current_value))
        if max_value < min_value:
            min_value, max_value = max_value, min_value

        maybe_step = spec.get("step")
        if isinstance(maybe_step, int | float) and float(maybe_step) > 0:
            step = float(maybe_step)
        else:
            step = max((max_value - min_value) / 10.0, 1e-6)

        output.append(
            SearchDimension(
                key=key,
                dtype="float",
                min_value=min_value,
                max_value=max_value,
                step=step,
            )
        )

    if not output:
        raise ValueError("search_space did not contain valid tunable keys")
    output.sort(key=lambda item: item.key)
    return output
