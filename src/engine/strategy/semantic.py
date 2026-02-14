"""Semantic validation for strategy DSL payloads."""

from __future__ import annotations

import re
from typing import Any

from src.engine.feature.indicators import IndicatorRegistry
from src.engine.strategy.errors import StrategyDslError

_FACTOR_ID_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")

_FACTOR_PARAM_ORDER: dict[str, tuple[str, ...]] = {
    "ema": ("period",),
    "sma": ("period",),
    "wma": ("period",),
    "dema": ("period",),
    "tema": ("period",),
    "kama": ("period",),
    "rsi": ("period",),
    "atr": ("period",),
    "macd": ("fast", "slow", "signal"),
    "bbands": ("period", "std_dev"),
    "stoch": ("k_period", "k_smooth", "d_period"),
}

_MULTI_OUTPUT_DEFAULTS: dict[str, set[str]] = {
    "macd": {"macd_line", "signal", "histogram"},
    "bbands": {"upper", "middle", "lower"},
    "stoch": {"k", "d"},
    "ichimoku": {"tenkan", "kijun", "senkou_a", "senkou_b", "chikou"},
}

_RESERVED_PRICE_REFS: set[str] = {
    "price.open",
    "price.high",
    "price.low",
    "price.close",
    "price.hl2",
    "price.hlc3",
    "price.ohlc4",
    "price.typical",
}


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _normalize_numeric_fragment(value: Any) -> str:
    if isinstance(value, bool) or value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        normalized = f"{value:.10f}".rstrip("0").rstrip(".")
        return normalized.replace("-", "n").replace(".", "_")
    return ""


def _expected_factor_id(*, factor_type: str, params: dict[str, Any]) -> str | None:
    order = _FACTOR_PARAM_ORDER.get(factor_type)
    if order is None:
        return None

    fragments: list[str] = [factor_type]
    for param_name in order:
        if param_name not in params:
            return None
        fragment = _normalize_numeric_fragment(params[param_name])
        if not fragment:
            return None
        fragments.append(fragment)

    source = params.get("source", "close")
    if isinstance(source, str):
        source_value = source.strip().lower()
        if source_value and source_value != "close":
            fragments.append(source_value)

    return "_".join(fragments)


def _major_version(value: str) -> int:
    left = value.split(".", 1)[0].strip()
    if left.isdigit():
        return int(left)
    return 0


def _allowed_outputs_for_factor(
    *,
    factor_type: str,
    explicit_outputs: list[str] | None,
) -> set[str]:
    if explicit_outputs:
        return {item for item in explicit_outputs if isinstance(item, str) and item.strip()}
    return set(_MULTI_OUTPUT_DEFAULTS.get(factor_type, set()))


def _to_indicator_params(
    *,
    factor_type: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    mapped: dict[str, Any] = {}
    for key, value in params.items():
        if key == "source":
            continue
        mapped[key] = value

    if "period" in mapped and factor_type in {
        "ema",
        "sma",
        "wma",
        "dema",
        "tema",
        "kama",
        "rsi",
        "atr",
        "bbands",
    }:
        mapped["length"] = mapped.pop("period")
    if factor_type == "bbands" and "std_dev" in mapped:
        mapped["std"] = mapped.pop("std_dev")
    if factor_type == "stoch":
        if "k_period" in mapped:
            mapped["fastk_period"] = mapped.pop("k_period")
        if "k_smooth" in mapped:
            mapped["slowk_period"] = mapped.pop("k_smooth")
        if "d_period" in mapped:
            mapped["slowd_period"] = mapped.pop("d_period")

    return mapped


def _is_valid_param_type(param_type: str, value: Any) -> bool:
    kind = param_type.strip().lower()
    if kind == "int":
        if isinstance(value, bool):
            return False
        if isinstance(value, int):
            return True
        return isinstance(value, float) and value.is_integer()
    if kind == "float":
        return _is_number(value)
    if kind == "str":
        return isinstance(value, str)
    if kind == "bool":
        return isinstance(value, bool)
    return True


def _validate_factor_indicator_contract(
    *,
    factor_id: str,
    factor_type: str,
    params: dict[str, Any],
    errors: list[StrategyDslError],
) -> None:
    metadata = IndicatorRegistry.get(factor_type)
    if metadata is None:
        errors.append(
            StrategyDslError(
                code="UNSUPPORTED_FACTOR_TYPE",
                message=f"Unsupported factor type '{factor_type}'.",
                path=f"$.factors.{factor_id}.type",
                value=factor_type,
            )
        )
        return

    mapped_params = _to_indicator_params(factor_type=factor_type, params=params)
    known_param_names = {param.name for param in metadata.params}
    unknown_params = sorted(name for name in mapped_params if name not in known_param_names)
    if unknown_params:
        errors.append(
            StrategyDslError(
                code="UNSUPPORTED_FACTOR_PARAM",
                message=f"Unsupported params for factor '{factor_id}': {unknown_params}",
                path=f"$.factors.{factor_id}.params",
                value=unknown_params,
            )
        )

    for param in metadata.params:
        value = mapped_params.get(param.name, param.default)
        if value is None:
            errors.append(
                StrategyDslError(
                    code="INVALID_FACTOR_PARAM_VALUE",
                    message=f"Missing required parameter '{param.name}' for factor '{factor_id}'.",
                    path=f"$.factors.{factor_id}.params.{param.name}",
                    value=value,
                )
            )
            continue

        if not _is_valid_param_type(param.type, value):
            errors.append(
                StrategyDslError(
                    code="INVALID_FACTOR_PARAM_VALUE",
                    message=(
                        f"Parameter '{param.name}' for factor '{factor_id}' "
                        f"must be type '{param.type}'."
                    ),
                    path=f"$.factors.{factor_id}.params.{param.name}",
                    value=value,
                )
            )
            continue

        if _is_number(value):
            numeric_value = float(value)
            if param.min_value is not None and numeric_value < float(param.min_value):
                errors.append(
                    StrategyDslError(
                        code="INVALID_FACTOR_PARAM_VALUE",
                        message=(
                            f"Parameter '{param.name}' for factor '{factor_id}' "
                            f"must be >= {param.min_value}."
                        ),
                        path=f"$.factors.{factor_id}.params.{param.name}",
                        value=value,
                    )
                )
            if param.max_value is not None and numeric_value > float(param.max_value):
                errors.append(
                    StrategyDslError(
                        code="INVALID_FACTOR_PARAM_VALUE",
                        message=(
                            f"Parameter '{param.name}' for factor '{factor_id}' "
                            f"must be <= {param.max_value}."
                        ),
                        path=f"$.factors.{factor_id}.params.{param.name}",
                        value=value,
                    )
                )

        if param.choices and value not in param.choices:
            errors.append(
                StrategyDslError(
                    code="INVALID_FACTOR_PARAM_VALUE",
                    message=(
                        f"Parameter '{param.name}' for factor '{factor_id}' "
                        f"must be one of {param.choices}."
                    ),
                    path=f"$.factors.{factor_id}.params.{param.name}",
                    value=value,
                )
            )


def _extract_factor_ref(ref: str) -> tuple[str, str | None]:
    if ref in _RESERVED_PRICE_REFS or ref == "volume":
        return ref, None

    base, sep, output = ref.partition(".")
    if not sep:
        return base, None
    return base, output


def _validate_ref(
    *,
    ref: str,
    path: str,
    factors: dict[str, dict[str, Any]],
    errors: list[StrategyDslError],
) -> None:
    base, output = _extract_factor_ref(ref)

    if base in _RESERVED_PRICE_REFS or base == "volume":
        return

    factor = factors.get(base)
    if factor is None:
        errors.append(
            StrategyDslError(
                code="UNKNOWN_FACTOR_REF",
                message=f"Unknown factor reference '{base}'",
                path=path,
                value=ref,
            )
        )
        return

    outputs = _allowed_outputs_for_factor(
        factor_type=str(factor.get("type", "")).strip().lower(),
        explicit_outputs=factor.get("outputs"),
    )
    factor_type = str(factor.get("type", "")).strip().lower()

    if output is None:
        if factor_type in _MULTI_OUTPUT_DEFAULTS:
            errors.append(
                StrategyDslError(
                    code="INVALID_OUTPUT_NAME",
                    message=(
                        f"Factor '{base}' is multi-output; "
                        "use explicit dot notation such as factor.output"
                    ),
                    path=path,
                    value=ref,
                )
            )
            return
        if len(outputs) > 1:
            errors.append(
                StrategyDslError(
                    code="INVALID_OUTPUT_NAME",
                    message=(
                        f"Factor '{base}' has multiple outputs; "
                        "use explicit dot notation such as factor.output"
                    ),
                    path=path,
                    value=ref,
                )
            )
        return

    if output not in outputs:
        errors.append(
            StrategyDslError(
                code="INVALID_OUTPUT_NAME",
                message=f"Invalid output '{output}' for factor '{base}'",
                path=path,
                value=ref,
            )
        )


def _collect_operand_refs(
    *,
    operand: Any,
    path: str,
    factors: dict[str, dict[str, Any]],
    errors: list[StrategyDslError],
) -> None:
    if not isinstance(operand, dict):
        return

    ref = operand.get("ref")
    if isinstance(ref, str):
        _validate_ref(ref=ref, path=f"{path}.ref", factors=factors, errors=errors)

    offset = operand.get("offset")
    if isinstance(offset, int) and offset > 0:
        errors.append(
            StrategyDslError(
                code="FUTURE_LOOK",
                message="Operand offset must be <= 0.",
                path=f"{path}.offset",
                value=offset,
            )
        )


def _collect_condition_refs(
    *,
    condition: Any,
    path: str,
    factors: dict[str, dict[str, Any]],
    errors: list[StrategyDslError],
    allow_temporal: bool,
) -> None:
    if not isinstance(condition, dict):
        return

    if "all" in condition and isinstance(condition["all"], list):
        for idx, child in enumerate(condition["all"]):
            _collect_condition_refs(
                condition=child,
                path=f"{path}.all[{idx}]",
                factors=factors,
                errors=errors,
                allow_temporal=allow_temporal,
            )
        return

    if "any" in condition and isinstance(condition["any"], list):
        for idx, child in enumerate(condition["any"]):
            _collect_condition_refs(
                condition=child,
                path=f"{path}.any[{idx}]",
                factors=factors,
                errors=errors,
                allow_temporal=allow_temporal,
            )
        return

    if "not" in condition:
        _collect_condition_refs(
            condition=condition.get("not"),
            path=f"{path}.not",
            factors=factors,
            errors=errors,
            allow_temporal=allow_temporal,
        )
        return

    if "cmp" in condition and isinstance(condition["cmp"], dict):
        cmp = condition["cmp"]
        _collect_operand_refs(
            operand=cmp.get("left"),
            path=f"{path}.cmp.left",
            factors=factors,
            errors=errors,
        )
        _collect_operand_refs(
            operand=cmp.get("right"),
            path=f"{path}.cmp.right",
            factors=factors,
            errors=errors,
        )
        return

    if "cross" in condition and isinstance(condition["cross"], dict):
        cross = condition["cross"]
        _collect_operand_refs(
            operand=cross.get("a"),
            path=f"{path}.cross.a",
            factors=factors,
            errors=errors,
        )
        _collect_operand_refs(
            operand=cross.get("b"),
            path=f"{path}.cross.b",
            factors=factors,
            errors=errors,
        )
        return

    if "ref" in condition and isinstance(condition["ref"], str):
        _validate_ref(
            ref=condition["ref"],
            path=f"{path}.ref",
            factors=factors,
            errors=errors,
        )
        return

    if "temporal" in condition:
        if not allow_temporal:
            errors.append(
                StrategyDslError(
                    code="TEMPORAL_NOT_SUPPORTED",
                    message="Temporal conditions are reserved and not supported in v1 runtime.",
                    path=f"{path}.temporal",
                    value=condition.get("temporal"),
                )
            )
            return
        temporal = condition.get("temporal")
        if isinstance(temporal, dict):
            for key in ("condition", "first", "then"):
                if key in temporal:
                    _collect_condition_refs(
                        condition=temporal.get(key),
                        path=f"{path}.temporal.{key}",
                        factors=factors,
                        errors=errors,
                        allow_temporal=allow_temporal,
                    )


def _validate_stop_specs(
    *,
    exits: list[dict[str, Any]],
    path: str,
    factors: dict[str, dict[str, Any]],
    errors: list[StrategyDslError],
) -> None:
    for idx, exit_rule in enumerate(exits):
        if not isinstance(exit_rule, dict):
            continue

        exit_type = str(exit_rule.get("type", ""))
        exit_path = f"{path}[{idx}]"

        if exit_type == "bracket_rr":
            has_stop = "stop" in exit_rule
            has_take = "take" in exit_rule
            if has_stop == has_take:
                errors.append(
                    StrategyDslError(
                        code="BRACKET_RR_CONFLICT",
                        message="bracket_rr requires exactly one of stop/take.",
                        path=exit_path,
                        value=exit_rule,
                    )
                )

        for stop_key in ("stop", "take"):
            stop_spec = exit_rule.get(stop_key)
            if not isinstance(stop_spec, dict):
                continue

            kind = stop_spec.get("kind")
            if kind != "atr_multiple":
                continue

            atr_ref = stop_spec.get("atr_ref")
            if not isinstance(atr_ref, str) or not atr_ref.strip():
                errors.append(
                    StrategyDslError(
                        code="MISSING_ATR_REF",
                        message="atr_multiple stop requires atr_ref pointing to an ATR factor.",
                        path=f"{exit_path}.{stop_key}.atr_ref",
                        value=atr_ref,
                    )
                )
                continue

            factor_name, output = _extract_factor_ref(atr_ref)
            factor = factors.get(factor_name)
            if output is not None or factor is None or str(factor.get("type", "")).lower() != "atr":
                errors.append(
                    StrategyDslError(
                        code="MISSING_ATR_REF",
                        message="atr_ref must reference a factor with type='atr'.",
                        path=f"{exit_path}.{stop_key}.atr_ref",
                        value=atr_ref,
                    )
                )


def validate_strategy_semantics(
    payload: dict[str, Any],
    *,
    allow_temporal: bool = False,
) -> list[StrategyDslError]:
    errors: list[StrategyDslError] = []

    dsl_version = str(payload.get("dsl_version", "")).strip()
    if dsl_version and _major_version(dsl_version) > 1:
        errors.append(
            StrategyDslError(
                code="UNSUPPORTED_DSL_VERSION",
                message=f"Unsupported DSL major version: {dsl_version}",
                path="$.dsl_version",
                value=dsl_version,
            )
        )

    raw_factors = payload.get("factors")
    factors: dict[str, dict[str, Any]] = {}
    if isinstance(raw_factors, dict):
        for factor_id, factor_def in raw_factors.items():
            if not isinstance(factor_id, str) or factor_id.startswith("x-"):
                continue
            if not isinstance(factor_def, dict):
                continue
            factors[factor_id] = factor_def

            if not _FACTOR_ID_PATTERN.match(factor_id):
                errors.append(
                    StrategyDslError(
                        code="FACTOR_ID_FORMAT_ERROR",
                        message=f"Invalid factor id format: {factor_id}",
                        path=f"$.factors.{factor_id}",
                        value=factor_id,
                    )
                )
                continue

            factor_type = str(factor_def.get("type", "")).strip().lower()
            params = factor_def.get("params")
            if not isinstance(params, dict):
                continue
            expected = _expected_factor_id(factor_type=factor_type, params=params)
            if expected and expected != factor_id:
                errors.append(
                    StrategyDslError(
                        code="FACTOR_ID_MISMATCH",
                        message=f"Factor id '{factor_id}' should be '{expected}'.",
                        path=f"$.factors.{factor_id}",
                        value={"actual": factor_id, "expected": expected},
                    )
                )
            _validate_factor_indicator_contract(
                factor_id=factor_id,
                factor_type=factor_type,
                params=params,
                errors=errors,
            )

    trade = payload.get("trade")
    if not isinstance(trade, dict):
        return errors

    if "long" not in trade and "short" not in trade:
        errors.append(
            StrategyDslError(
                code="NO_TRADE_SIDE",
                message="At least one of trade.long / trade.short must be defined.",
                path="$.trade",
                value=trade,
            )
        )

    for side_name in ("long", "short"):
        side = trade.get(side_name)
        if not isinstance(side, dict):
            continue

        entry = side.get("entry")
        if isinstance(entry, dict):
            condition = entry.get("condition")
            _collect_condition_refs(
                condition=condition,
                path=f"$.trade.{side_name}.entry.condition",
                factors=factors,
                errors=errors,
                allow_temporal=allow_temporal,
            )

        exits = side.get("exits")
        if isinstance(exits, list):
            _validate_stop_specs(
                exits=exits,
                path=f"$.trade.{side_name}.exits",
                factors=factors,
                errors=errors,
            )
            for idx, exit_rule in enumerate(exits):
                if not isinstance(exit_rule, dict):
                    continue
                condition = exit_rule.get("condition")
                if condition is None:
                    continue
                _collect_condition_refs(
                    condition=condition,
                    path=f"$.trade.{side_name}.exits[{idx}].condition",
                    factors=factors,
                    errors=errors,
                    allow_temporal=allow_temporal,
                )

    return errors
