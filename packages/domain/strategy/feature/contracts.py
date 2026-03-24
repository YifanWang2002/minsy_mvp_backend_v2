"""Shared indicator DSL contracts.

This module centralizes indicator parameter mappings and output alias behavior
that previously lived in semantic validation, backtest runtime, MCP tools,
and trade snapshot rendering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class IndicatorContract:
    """Contract mapping between DSL-facing names and engine-facing names."""

    name: str
    dsl_to_indicator_params: dict[str, str]
    indicator_to_dsl_params: dict[str, str]
    default_outputs: tuple[str, ...]
    dsl_output_alias_map: dict[str, str]


_PERIOD_BASED_FACTORS: set[str] = {
    "ema",
    "sma",
    "wma",
    "dema",
    "tema",
    "kama",
    "rsi",
    "atr",
    "bbands",
}

_DEFAULT_MULTI_OUTPUTS: dict[str, tuple[str, ...]] = {
    "macd": ("macd_line", "signal", "histogram"),
    "bbands": ("upper", "middle", "lower"),
    "stoch": ("k", "d"),
    "adx": ("adx", "dmp", "dmn"),
    "ichimoku": ("tenkan", "kijun", "senkou_a", "senkou_b", "chikou"),
}

_DSL_OUTPUT_ALIAS_MAP: dict[str, dict[str, str]] = {
    "macd": {
        "MACD": "macd_line",
        "MACDs": "signal",
        "MACDh": "histogram",
    },
    "bbands": {
        "BBU": "upper",
        "BBM": "middle",
        "BBL": "lower",
    },
    "stoch": {
        "STOCHk": "k",
        "STOCHd": "d",
    },
    "adx": {
        "ADX": "adx",
        "DMP": "dmp",
        "DMN": "dmn",
    },
}

_FALLBACK_INDICATOR_TO_DSL_PARAM: dict[str, str] = {
    "length": "period",
    "std": "std_dev",
    "fastk_period": "k_period",
    "slowk_period": "k_smooth",
    "slowd_period": "d_period",
}


def _dsl_to_indicator_param_map(indicator: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if indicator in _PERIOD_BASED_FACTORS:
        mapping["period"] = "length"
    if indicator == "bbands":
        mapping["std_dev"] = "std"
    if indicator == "stoch":
        mapping["k_period"] = "fastk_period"
        mapping["k_smooth"] = "slowk_period"
        mapping["d_period"] = "slowd_period"
    return mapping


def _build_contract(indicator: str) -> IndicatorContract:
    dsl_to_indicator = _dsl_to_indicator_param_map(indicator)
    indicator_to_dsl = {value: key for key, value in dsl_to_indicator.items()}
    return IndicatorContract(
        name=indicator,
        dsl_to_indicator_params=dsl_to_indicator,
        indicator_to_dsl_params=indicator_to_dsl,
        default_outputs=_DEFAULT_MULTI_OUTPUTS.get(indicator, tuple()),
        dsl_output_alias_map=dict(_DSL_OUTPUT_ALIAS_MAP.get(indicator, {})),
    )


_CONTRACT_NAMES: set[str] = (
    set(_PERIOD_BASED_FACTORS)
    | set(_DEFAULT_MULTI_OUTPUTS)
    | set(_DSL_OUTPUT_ALIAS_MAP)
    | {"stoch"}
)
_CONTRACTS: dict[str, IndicatorContract] = {
    name: _build_contract(name)
    for name in _CONTRACT_NAMES
}


def _metadata_default_outputs(indicator: str) -> tuple[str, ...]:
    from packages.domain.strategy.feature.indicators import IndicatorRegistry

    metadata = IndicatorRegistry.get(indicator)
    if metadata is None or len(metadata.outputs) < 2:
        return tuple()
    outputs: list[str] = []
    seen: set[str] = set()
    for output in metadata.outputs:
        name = str(output.name).strip()
        if not name:
            continue
        lowered = name.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        outputs.append(name)
    if len(outputs) < 2:
        return tuple()
    return tuple(outputs)


def get_contract(indicator: str) -> IndicatorContract | None:
    """Return contract for one indicator, if present."""
    key = str(indicator).strip().lower()
    if not key:
        return None
    return _CONTRACTS.get(key)


def resolve_indicator_params_from_dsl(
    indicator: str,
    params: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    """Map DSL factor params to indicator wrapper params."""
    key = str(indicator).strip().lower()
    source = str(params.get("source", "close")).strip().lower() or "close"
    contract = get_contract(key)
    mapping = contract.dsl_to_indicator_params if contract else {}

    mapped: dict[str, Any] = {}
    for raw_key, value in params.items():
        if raw_key == "source":
            continue
        target_key = mapping.get(raw_key, raw_key)
        mapped[target_key] = value
    return mapped, source


def to_dsl_param_name(indicator: str, indicator_param_name: str) -> str:
    """Map indicator metadata param names back to DSL naming."""
    key = str(indicator).strip().lower()
    name = str(indicator_param_name).strip()
    if not name:
        return name
    contract = get_contract(key)
    if contract is not None and name in contract.indicator_to_dsl_params:
        return contract.indicator_to_dsl_params[name]
    return _FALLBACK_INDICATOR_TO_DSL_PARAM.get(name, name)


def default_outputs_for_indicator(indicator: str) -> tuple[str, ...]:
    contract = get_contract(indicator)
    if contract is not None and contract.default_outputs:
        return tuple(contract.default_outputs)
    return _metadata_default_outputs(indicator)


def is_multi_output_indicator(indicator: str) -> bool:
    return len(default_outputs_for_indicator(indicator)) > 1


def dsl_output_alias_map(indicator: str) -> dict[str, str]:
    contract = get_contract(indicator)
    if contract is None:
        return {}
    return dict(contract.dsl_output_alias_map)


def canonicalize_output_name(indicator: str, output_name: str | None) -> str | None:
    if output_name is None:
        return None
    text = str(output_name).strip()
    if not text:
        return None

    defaults = set(default_outputs_for_indicator(indicator))
    if text in defaults:
        return text

    alias_map = dsl_output_alias_map(indicator)
    mapped = alias_map.get(text)
    if mapped is not None:
        return mapped

    lowered = text.lower()
    for alias, canonical in alias_map.items():
        if alias.lower() == lowered:
            return canonical
    for default_output in defaults:
        if default_output.lower() == lowered:
            return default_output
    return text


def deprecated_output_alias(indicator: str, output_name: str | None) -> str | None:
    """Return canonical output name when input token is a deprecated alias."""
    if output_name is None:
        return None
    text = str(output_name).strip()
    if not text:
        return None
    alias_map = dsl_output_alias_map(indicator)
    direct = alias_map.get(text)
    if direct is not None and direct != text:
        return direct
    lowered = text.lower()
    for alias, canonical in alias_map.items():
        if alias.lower() == lowered and canonical != text:
            return canonical
    return None


def to_dsl_output_alias(indicator: str, output_name: str) -> str:
    """Return DSL-canonical output name for one metadata/legacy output token."""
    resolved = canonicalize_output_name(indicator, output_name)
    if resolved is None:
        return str(output_name)
    return resolved


def aliases_for_output_name(indicator: str, output_name: str) -> tuple[str, ...]:
    """Return alternate aliases for one canonical or legacy output token."""
    text = str(output_name).strip()
    if not text:
        return tuple()
    canonical = canonicalize_output_name(indicator, text)
    if canonical is None:
        return tuple()

    aliases: set[str] = set()
    if canonical != text:
        aliases.add(canonical)

    alias_map = dsl_output_alias_map(indicator)
    for alias, canonical_name in alias_map.items():
        if canonical_name == canonical and alias != text:
            aliases.add(alias)
    aliases.discard(text)
    return tuple(sorted(aliases))


def allowed_outputs_for_factor(
    *,
    factor_type: str,
    explicit_outputs: list[str] | None,
) -> set[str]:
    """Resolve allowed output tokens for semantic validation."""
    indicator = str(factor_type).strip().lower()
    alias_map = dsl_output_alias_map(indicator)
    alias_keys = set(alias_map.keys())
    alias_values = set(alias_map.values())

    if explicit_outputs:
        resolved: set[str] = set()
        for item in explicit_outputs:
            if not isinstance(item, str):
                continue
            name = item.strip()
            if not name:
                continue
            resolved.add(name)
            canonical = canonicalize_output_name(indicator, name)
            if canonical is not None:
                resolved.add(canonical)
            lowered = name.lower()
            if lowered and lowered != name:
                resolved.add(lowered)
        return resolved

    defaults = set(default_outputs_for_indicator(indicator))
    return defaults | alias_keys | alias_values


def _normalize_tokens(values: list[str]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for item in values:
        token = "".join(ch for ch in item.lower() if ch.isalnum())
        normalized[token] = item
    return normalized


def _map_macd_columns(columns: list[str]) -> dict[str, str]:
    normalized = _normalize_tokens(columns)
    mapping: dict[str, str] = {}
    for key, original in normalized.items():
        if "hist" in key:
            mapping["histogram"] = original
        elif key.endswith("s") or "signal" in key:
            mapping["signal"] = original
        elif "macd" in key:
            mapping["macd_line"] = original
    return mapping


def _map_bbands_columns(columns: list[str]) -> dict[str, str]:
    normalized = _normalize_tokens(columns)
    mapping: dict[str, str] = {}
    for key, original in normalized.items():
        if "upper" in key or key.startswith("bbu"):
            mapping["upper"] = original
        elif "middle" in key or key.startswith("bbm") or key.startswith("mid"):
            mapping["middle"] = original
        elif "lower" in key or key.startswith("bbl"):
            mapping["lower"] = original
    return mapping


def _map_stoch_columns(columns: list[str]) -> dict[str, str]:
    normalized = _normalize_tokens(columns)
    mapping: dict[str, str] = {}
    for key, original in normalized.items():
        if key.endswith("k") or "stochk" in key:
            mapping["k"] = original
        elif key.endswith("d") or "stochd" in key:
            mapping["d"] = original
    return mapping


def _map_adx_columns(columns: list[str]) -> dict[str, str]:
    normalized = _normalize_tokens(columns)
    mapping: dict[str, str] = {}
    for key, original in normalized.items():
        if key == "adx":
            mapping["adx"] = original
        elif key in {"dmp", "plusdi", "pdi"}:
            mapping["dmp"] = original
        elif key in {"dmn", "minusdi", "mdi"}:
            mapping["dmn"] = original
    return mapping


def _column_mapping_by_indicator(indicator: str, columns: list[str]) -> dict[str, str]:
    name = str(indicator).strip().lower()
    if name == "macd":
        return _map_macd_columns(columns)
    if name == "bbands":
        return _map_bbands_columns(columns)
    if name == "stoch":
        return _map_stoch_columns(columns)
    if name == "adx":
        return _map_adx_columns(columns)
    return {}


def resolve_multi_output_assignments(
    *,
    indicator: str,
    requested_outputs: tuple[str, ...],
    result_columns: list[str],
) -> list[tuple[str, str]]:
    """Resolve result-column to output-name assignments for DataFrame outputs."""
    if not result_columns:
        return []

    defaults = default_outputs_for_indicator(indicator)
    if requested_outputs:
        target_outputs = tuple(
            canonicalize_output_name(indicator, output) or output
            for output in requested_outputs
        )
    elif defaults:
        target_outputs = defaults
    else:
        target_outputs = tuple(result_columns)

    column_map = _column_mapping_by_indicator(indicator, result_columns)
    assignments: list[tuple[str, str]] = []
    used_columns: set[str] = set()
    for output in target_outputs:
        source_column = column_map.get(output)
        if source_column is None and output in result_columns:
            source_column = output
        if source_column and source_column not in used_columns:
            used_columns.add(source_column)
            assignments.append((source_column, output))
            continue

        fallback = next((col for col in result_columns if col not in used_columns), None)
        if fallback is None:
            break
        used_columns.add(fallback)
        assignments.append((fallback, output))

    return assignments


def missing_multi_output_contracts() -> list[str]:
    """Return multi-output indicators that still resolve no default outputs."""
    from packages.domain.strategy.feature.indicators import IndicatorRegistry

    missing: list[str] = []
    for indicator in IndicatorRegistry.list_all():
        metadata = IndicatorRegistry.get(indicator)
        if metadata is None:
            continue
        if len(metadata.outputs) < 2:
            continue
        if len(default_outputs_for_indicator(indicator)) < 2:
            missing.append(indicator)
    return sorted(set(missing))
