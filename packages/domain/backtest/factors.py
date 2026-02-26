"""Factor runtime for strategy DSL backtests."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from packages.domain.strategy.feature.indicators import IndicatorWrapper
from packages.domain.strategy.models import ParsedStrategyDsl

_DEFAULT_MULTI_OUTPUTS: dict[str, tuple[str, ...]] = {
    "macd": ("macd_line", "signal", "histogram"),
    "bbands": ("upper", "middle", "lower"),
    "stoch": ("k", "d"),
}


def prepare_backtest_frame(
    data: pd.DataFrame,
    *,
    strategy: ParsedStrategyDsl,
    indicator_wrapper: IndicatorWrapper | None = None,
) -> pd.DataFrame:
    """Return a DataFrame enriched with all factor columns required by DSL."""

    wrapper = indicator_wrapper or IndicatorWrapper()
    frame = data.copy()
    frame.columns = [str(col).strip().lower() for col in frame.columns]
    frame = _ensure_price_columns(frame)

    for factor in strategy.factors.values():
        indicator_params, source = _to_indicator_params(factor.factor_type, factor.params)
        result = wrapper.calculate(
            factor.factor_type,
            frame,
            source=source,
            **indicator_params,
        )
        _attach_factor_result(
            frame=frame,
            factor_id=factor.factor_id,
            factor_type=factor.factor_type,
            requested_outputs=factor.outputs,
            result=result,
        )

    return frame


def _ensure_price_columns(frame: pd.DataFrame) -> pd.DataFrame:
    required = ("open", "high", "low", "close", "volume")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {missing}")

    frame["price.open"] = frame["open"]
    frame["price.high"] = frame["high"]
    frame["price.low"] = frame["low"]
    frame["price.close"] = frame["close"]
    frame["price.hl2"] = (frame["high"] + frame["low"]) / 2.0
    frame["price.hlc3"] = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    frame["price.ohlc4"] = (frame["open"] + frame["high"] + frame["low"] + frame["close"]) / 4.0
    frame["price.typical"] = frame["price.hlc3"]

    frame["hl2"] = frame["price.hl2"]
    frame["hlc3"] = frame["price.hlc3"]
    frame["ohlc4"] = frame["price.ohlc4"]
    frame["typical"] = frame["price.typical"]
    return frame


def _to_indicator_params(
    factor_type: str,
    params: dict[str, object],
) -> tuple[dict[str, object], str]:
    source = str(params.get("source", "close")).strip().lower() or "close"
    mapped: dict[str, object] = {}
    for key, value in params.items():
        if key == "source":
            continue
        mapped[key] = value

    # DSL naming -> indicator wrapper naming.
    if "period" in mapped and factor_type in {"ema", "sma", "wma", "dema", "tema", "kama", "rsi", "atr", "bbands"}:
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

    return mapped, source


def _attach_factor_result(
    *,
    frame: pd.DataFrame,
    factor_id: str,
    factor_type: str,
    requested_outputs: tuple[str, ...],
    result: pd.Series | pd.DataFrame,
) -> None:
    if isinstance(result, pd.Series):
        numeric_series = pd.to_numeric(result, errors="coerce")
        frame[factor_id] = numeric_series
        for output in requested_outputs:
            alias = output.strip()
            if not alias:
                continue
            frame[f"{factor_id}.{alias}"] = numeric_series
        return

    columns = list(result.columns)
    output_names = _resolve_output_names(
        factor_type=factor_type,
        requested_outputs=requested_outputs,
        result_columns=columns,
    )

    for source_column, output_name in output_names:
        frame[f"{factor_id}.{output_name}"] = pd.to_numeric(
            result[source_column],
            errors="coerce",
        )


def _resolve_output_names(
    *,
    factor_type: str,
    requested_outputs: tuple[str, ...],
    result_columns: list[str],
) -> list[tuple[str, str]]:
    if not result_columns:
        return []

    target_outputs: tuple[str, ...]
    if requested_outputs:
        target_outputs = requested_outputs
    else:
        target_outputs = _DEFAULT_MULTI_OUTPUTS.get(factor_type, tuple(result_columns))

    if factor_type == "macd":
        mapped = _map_macd_columns(result_columns)
    elif factor_type == "bbands":
        mapped = _map_bbands_columns(result_columns)
    elif factor_type == "stoch":
        mapped = _map_stoch_columns(result_columns)
    else:
        mapped = {}

    assignments: list[tuple[str, str]] = []
    used_columns: set[str] = set()
    for output in target_outputs:
        column = mapped.get(output)
        if column and column not in used_columns:
            used_columns.add(column)
            assignments.append((column, output))
            continue

        fallback = next((col for col in result_columns if col not in used_columns), None)
        if fallback is None:
            break
        used_columns.add(fallback)
        assignments.append((fallback, output))

    return assignments


def _normalize_tokens(values: Iterable[str]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for item in values:
        key = "".join(ch for ch in item.lower() if ch.isalnum())
        normalized[key] = item
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
