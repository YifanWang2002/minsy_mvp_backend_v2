"""Factor runtime for strategy DSL backtests."""

from __future__ import annotations

import pandas as pd

from packages.domain.strategy.feature.contracts import (
    aliases_for_output_name,
    deprecated_output_alias,
    resolve_indicator_params_from_dsl,
    resolve_multi_output_assignments,
)
from packages.domain.strategy.feature.indicators import IndicatorWrapper
from packages.domain.strategy.models import ParsedStrategyDsl
from packages.infra.observability.logger import logger


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
    mapped, source = resolve_indicator_params_from_dsl(factor_type, params)
    return mapped, source


def _attach_factor_result(
    *,
    frame: pd.DataFrame,
    factor_id: str,
    factor_type: str,
    requested_outputs: tuple[str, ...],
    result: pd.Series | pd.DataFrame,
) -> None:
    for output in requested_outputs:
        canonical = deprecated_output_alias(factor_type, output)
        if canonical is not None:
            logger.warning(
                "backtest uses deprecated output alias factor_id=%s indicator=%s alias=%s canonical=%s",
                factor_id,
                factor_type,
                output,
                canonical,
            )

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
        numeric_series = pd.to_numeric(
            result[source_column],
            errors="coerce",
        )
        frame[f"{factor_id}.{output_name}"] = numeric_series
        for alias in _alias_output_names(
            factor_type=factor_type,
            output_name=output_name,
        ):
            frame[f"{factor_id}.{alias}"] = numeric_series


def _resolve_output_names(
    *,
    factor_type: str,
    requested_outputs: tuple[str, ...],
    result_columns: list[str],
) -> list[tuple[str, str]]:
    return resolve_multi_output_assignments(
        indicator=factor_type,
        requested_outputs=requested_outputs,
        result_columns=result_columns,
    )


def _alias_output_names(*, factor_type: str, output_name: str) -> tuple[str, ...]:
    return aliases_for_output_name(factor_type, output_name)
