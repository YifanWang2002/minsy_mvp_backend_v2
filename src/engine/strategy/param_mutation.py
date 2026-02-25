"""Helpers for extracting and mutating tunable strategy parameters."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TunableParam:
    """One numeric parameter candidate for scans/optimization."""

    factor_id: str
    param_name: str
    key: str
    current_value: float


def list_tunable_params(
    payload: dict[str, Any],
    *,
    target_params: list[str] | None = None,
) -> list[TunableParam]:
    """Extract numeric params from DSL factors."""

    factors = payload.get("factors") if isinstance(payload, dict) else None
    if not isinstance(factors, dict):
        return []

    filters = {item.strip() for item in (target_params or []) if isinstance(item, str) and item.strip()}
    output: list[TunableParam] = []

    for factor_id, factor in factors.items():
        if not isinstance(factor_id, str) or not isinstance(factor, dict):
            continue
        params = factor.get("params")
        if not isinstance(params, dict):
            continue
        for param_name, value in params.items():
            if not isinstance(param_name, str):
                continue
            if not isinstance(value, int | float) or isinstance(value, bool):
                continue
            key = f"{factor_id}.{param_name}"
            json_key = f"/factors/{factor_id}/params/{param_name}"
            if filters and key not in filters and json_key not in filters and param_name not in filters:
                continue
            output.append(
                TunableParam(
                    factor_id=factor_id,
                    param_name=param_name,
                    key=key,
                    current_value=float(value),
                )
            )

    output.sort(key=lambda item: (item.factor_id, item.param_name))
    return output


def apply_param_values(
    payload: dict[str, Any],
    *,
    values: dict[str, float],
) -> dict[str, Any]:
    """Return a deep-copied payload with parameter overrides applied."""

    next_payload = deepcopy(payload)
    factors = next_payload.get("factors")
    if not isinstance(factors, dict):
        return next_payload

    for raw_key, raw_value in values.items():
        factor_id, param_name = _resolve_param_key(raw_key)
        if factor_id is None or param_name is None:
            continue
        factor = factors.get(factor_id)
        if not isinstance(factor, dict):
            continue
        params = factor.get("params")
        if not isinstance(params, dict):
            continue
        if param_name not in params:
            continue

        source_value = params.get(param_name)
        if isinstance(source_value, int) and not isinstance(source_value, bool):
            params[param_name] = int(round(float(raw_value)))
        else:
            params[param_name] = float(raw_value)
    return next_payload


def _resolve_param_key(key: str) -> tuple[str | None, str | None]:
    text = str(key).strip()
    if not text:
        return None, None
    if text.startswith("/factors/"):
        parts = text.strip("/").split("/")
        if len(parts) == 4 and parts[0] == "factors" and parts[2] == "params":
            return parts[1], parts[3]
        return None, None

    if "." not in text:
        return None, None
    factor_id, param_name = text.split(".", 1)
    factor_id = factor_id.strip()
    param_name = param_name.strip()
    if not factor_id or not param_name:
        return None, None
    return factor_id, param_name
