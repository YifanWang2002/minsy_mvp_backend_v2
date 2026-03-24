"""Regime-focused indicators used by pre-strategy market diagnosis."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from ..base import IndicatorCategory, IndicatorMetadata, IndicatorOutput, IndicatorParam
from ..decorators import indicator


def _rolling_min_periods(window: int) -> int:
    return max(3, int(window) // 3)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0.0, np.nan)


def _true_range(data: pd.DataFrame) -> pd.Series:
    prev_close = data["close"].shift(1)
    tr = pd.concat(
        [
            (data["high"] - data["low"]).abs(),
            (data["high"] - prev_close).abs(),
            (data["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.fillna(0.0)


def _atr(data: pd.DataFrame, period: int) -> pd.Series:
    return _true_range(data).rolling(
        int(period),
        min_periods=_rolling_min_periods(int(period)),
    ).mean()


def _rolling_percentile_rank(series: pd.Series, window: int) -> pd.Series:
    def _rank(values: np.ndarray) -> float:
        if len(values) == 0:
            return float("nan")
        latest = float(values[-1])
        if math.isnan(latest):
            return float("nan")
        valid = values[~np.isnan(values)]
        if len(valid) == 0:
            return float("nan")
        return float((valid <= latest).sum()) / float(len(valid))

    return series.rolling(
        int(window),
        min_periods=_rolling_min_periods(int(window)),
    ).apply(_rank, raw=True)


@indicator(
    metadata=IndicatorMetadata(
        name="efficiency_ratio",
        full_name="Efficiency Ratio",
        category=IndicatorCategory.UTILS,
        description="Net directional move divided by cumulative absolute move",
        params=[IndicatorParam("length", "int", 50, 5, 500, "Lookback window")],
        outputs=[IndicatorOutput("efficiency_ratio", "Efficiency ratio (0-1)")],
        talib_func=None,
        pandas_ta_func=None,
        required_columns=["close"],
    )
)
def _calc_efficiency_ratio(
    data: pd.DataFrame,
    use_talib: bool = True,
    source: str = "close",
    length: int = 50,
    **kwargs: Any,
) -> pd.Series:
    """Kaufman-style efficiency ratio: net move / path length."""
    del use_talib, kwargs
    lookback = int(length)
    src = data[source]
    direction = (src - src.shift(lookback)).abs()
    path = src.diff().abs().rolling(
        lookback,
        min_periods=_rolling_min_periods(lookback),
    ).sum()
    return _safe_divide(direction, path).fillna(0.0).clip(lower=0.0, upper=1.0)


@indicator(
    metadata=IndicatorMetadata(
        name="directional_persistence",
        full_name="Directional Persistence",
        category=IndicatorCategory.UTILS,
        description="Rolling probability that return signs continue",
        params=[IndicatorParam("length", "int", 50, 5, 500, "Lookback window")],
        outputs=[IndicatorOutput("directional_persistence", "Persistence ratio (0-1)")],
        talib_func=None,
        pandas_ta_func=None,
        required_columns=["close"],
    )
)
def _calc_directional_persistence(
    data: pd.DataFrame,
    use_talib: bool = True,
    source: str = "close",
    length: int = 50,
    **kwargs: Any,
) -> pd.Series:
    """Share of bars whose return sign matches the previous bar."""
    del use_talib, kwargs
    lookback = int(length)
    returns = data[source].pct_change()
    sign = np.sign(returns)
    same_sign = sign.eq(sign.shift(1)) & sign.ne(0) & sign.shift(1).ne(0)
    return same_sign.astype(float).rolling(
        lookback,
        min_periods=_rolling_min_periods(lookback),
    ).mean().fillna(0.5).clip(lower=0.0, upper=1.0)


@indicator(
    metadata=IndicatorMetadata(
        name="sign_autocorrelation",
        full_name="Sign Autocorrelation",
        category=IndicatorCategory.UTILS,
        description="Lag-1 autocorrelation of return signs",
        params=[IndicatorParam("length", "int", 50, 5, 500, "Lookback window")],
        outputs=[IndicatorOutput("sign_autocorrelation", "Autocorrelation (-1 to 1)")],
        talib_func=None,
        pandas_ta_func=None,
        required_columns=["close"],
    )
)
def _calc_sign_autocorrelation(
    data: pd.DataFrame,
    use_talib: bool = True,
    source: str = "close",
    length: int = 50,
    **kwargs: Any,
) -> pd.Series:
    """Lag-1 autocorrelation of return signs."""
    del use_talib, kwargs
    lookback = int(length)
    sign = np.sign(data[source].pct_change()).replace(0.0, np.nan)

    def _autocorr(values: np.ndarray) -> float:
        valid = values[~np.isnan(values)]
        if len(valid) < 3:
            return float("nan")
        left = valid[1:]
        right = valid[:-1]
        if np.std(left) == 0.0 or np.std(right) == 0.0:
            return 0.0
        return float(np.corrcoef(left, right)[0, 1])

    return sign.rolling(
        lookback,
        min_periods=_rolling_min_periods(lookback),
    ).apply(_autocorr, raw=True).fillna(0.0).clip(lower=-1.0, upper=1.0)


def _breakout_events(data: pd.DataFrame, length: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    lookback = int(length)
    rolling_min_periods = _rolling_min_periods(lookback)
    prior_high = data["high"].rolling(
        lookback,
        min_periods=rolling_min_periods,
    ).max().shift(1)
    prior_low = data["low"].rolling(
        lookback,
        min_periods=rolling_min_periods,
    ).min().shift(1)
    breakout = (
        (data["close"] > prior_high) | (data["close"] < prior_low)
    ).fillna(False)
    return breakout, prior_high, prior_low


@indicator(
    metadata=IndicatorMetadata(
        name="breakout_frequency",
        full_name="Breakout Frequency",
        category=IndicatorCategory.UTILS,
        description="Rolling share of bars that break prior channel highs/lows",
        params=[
            IndicatorParam("length", "int", 20, 5, 300, "Channel lookback"),
            IndicatorParam("window", "int", 50, 5, 500, "Frequency window"),
        ],
        outputs=[IndicatorOutput("breakout_frequency", "Breakout ratio (0-1)")],
        talib_func=None,
        pandas_ta_func=None,
        required_columns=["high", "low", "close"],
    )
)
def _calc_breakout_frequency(
    data: pd.DataFrame,
    use_talib: bool = True,
    length: int = 20,
    window: int = 50,
    **kwargs: Any,
) -> pd.Series:
    """Rolling frequency of channel breakouts."""
    del use_talib, kwargs
    breakout, _, _ = _breakout_events(data, int(length))
    roll_window = int(window)
    return breakout.astype(float).rolling(
        roll_window,
        min_periods=_rolling_min_periods(roll_window),
    ).mean().fillna(0.0).clip(lower=0.0, upper=1.0)


@indicator(
    metadata=IndicatorMetadata(
        name="false_breakout_frequency",
        full_name="False Breakout Frequency",
        category=IndicatorCategory.UTILS,
        description="Rolling share of breakouts that mean-revert within confirm bars",
        params=[
            IndicatorParam("length", "int", 20, 5, 300, "Channel lookback"),
            IndicatorParam("confirm_bars", "int", 5, 1, 30, "Bars to confirm false break"),
            IndicatorParam("window", "int", 50, 5, 500, "Frequency window"),
        ],
        outputs=[IndicatorOutput("false_breakout_frequency", "False-breakout ratio (0-1)")],
        talib_func=None,
        pandas_ta_func=None,
        required_columns=["high", "low", "close"],
    )
)
def _calc_false_breakout_frequency(
    data: pd.DataFrame,
    use_talib: bool = True,
    length: int = 20,
    confirm_bars: int = 5,
    window: int = 50,
    **kwargs: Any,
) -> pd.Series:
    """Causal-safe rolling fraction of historically failed breakouts.

    At bar ``t`` we only score breakouts that occurred at ``t-confirm_bars`` and
    therefore already have enough observed bars to evaluate mean-reversion.
    """
    del use_talib, kwargs
    breakout, prior_high, prior_low = _breakout_events(data, int(length))
    confirm = int(confirm_bars)
    matured_breakout = pd.Series(0.0, index=data.index, dtype=float)
    matured_false_breakout = pd.Series(0.0, index=data.index, dtype=float)
    close_series = data["close"]

    for current_idx in range(len(data)):
        breakout_idx = current_idx - confirm
        if breakout_idx < 0:
            continue
        if not bool(breakout.iloc[breakout_idx]):
            continue
        matured_breakout.iloc[current_idx] = 1.0
        range_high = float(prior_high.iloc[breakout_idx]) if pd.notna(prior_high.iloc[breakout_idx]) else float("nan")
        range_low = float(prior_low.iloc[breakout_idx]) if pd.notna(prior_low.iloc[breakout_idx]) else float("nan")
        if np.isnan(range_high) or np.isnan(range_low):
            continue
        observation = close_series.iloc[breakout_idx + 1 : current_idx + 1]
        if observation.empty:
            continue
        reverted = bool(((observation <= range_high) & (observation >= range_low)).any())
        if reverted:
            matured_false_breakout.iloc[current_idx] = 1.0

    roll_window = int(window)
    numerator = matured_false_breakout.rolling(
        roll_window,
        min_periods=_rolling_min_periods(roll_window),
    ).sum()
    denominator = matured_breakout.rolling(
        roll_window,
        min_periods=_rolling_min_periods(roll_window),
    ).sum()
    return _safe_divide(numerator, denominator).fillna(0.0).clip(lower=0.0, upper=1.0)


@indicator(
    metadata=IndicatorMetadata(
        name="volatility_regime_ratio",
        full_name="Volatility Regime Ratio",
        category=IndicatorCategory.VOLATILITY,
        description="Ratio of short-horizon to long-horizon realized volatility",
        params=[
            IndicatorParam("short_window", "int", 20, 5, 300, "Short volatility window"),
            IndicatorParam("long_window", "int", 60, 10, 600, "Long volatility window"),
        ],
        outputs=[IndicatorOutput("volatility_regime_ratio", "Short/long volatility ratio")],
        talib_func=None,
        pandas_ta_func=None,
        required_columns=["close"],
    )
)
def _calc_volatility_regime_ratio(
    data: pd.DataFrame,
    use_talib: bool = True,
    source: str = "close",
    short_window: int = 20,
    long_window: int = 60,
    **kwargs: Any,
) -> pd.Series:
    """Short/long realized volatility ratio."""
    del use_talib, kwargs
    short_len = int(short_window)
    long_len = int(long_window)
    returns = data[source].pct_change()
    short_vol = returns.rolling(
        short_len,
        min_periods=_rolling_min_periods(short_len),
    ).std()
    long_vol = returns.rolling(
        long_len,
        min_periods=_rolling_min_periods(long_len),
    ).std()
    return _safe_divide(short_vol, long_vol).fillna(0.0)


@indicator(
    metadata=IndicatorMetadata(
        name="atr_regime_ratio",
        full_name="ATR Regime Ratio",
        category=IndicatorCategory.VOLATILITY,
        description="Ratio of short-term ATR to long-term ATR baseline",
        params=[
            IndicatorParam("atr_period", "int", 14, 2, 200, "ATR base period"),
            IndicatorParam("short_window", "int", 14, 2, 300, "Short ATR smooth window"),
            IndicatorParam("long_window", "int", 50, 5, 600, "Long ATR smooth window"),
        ],
        outputs=[IndicatorOutput("atr_regime_ratio", "Short/long ATR ratio")],
        talib_func=None,
        pandas_ta_func=None,
        required_columns=["high", "low", "close"],
    )
)
def _calc_atr_regime_ratio(
    data: pd.DataFrame,
    use_talib: bool = True,
    atr_period: int = 14,
    short_window: int = 14,
    long_window: int = 50,
    **kwargs: Any,
) -> pd.Series:
    """Short/long ATR moving-average ratio."""
    del use_talib, kwargs
    atr_series = _atr(data, int(atr_period))
    short_len = int(short_window)
    long_len = int(long_window)
    short_atr = atr_series.rolling(
        short_len,
        min_periods=_rolling_min_periods(short_len),
    ).mean()
    long_atr = atr_series.rolling(
        long_len,
        min_periods=_rolling_min_periods(long_len),
    ).mean()
    return _safe_divide(short_atr, long_atr).fillna(0.0)


@indicator(
    metadata=IndicatorMetadata(
        name="squeeze_score",
        full_name="Squeeze Score",
        category=IndicatorCategory.VOLATILITY,
        description="Volatility compression score from BB/KC width percentiles",
        params=[
            IndicatorParam("length", "int", 20, 5, 300, "Band window"),
            IndicatorParam("bb_mult", "float", 2.0, 0.5, 5.0, "Bollinger std multiplier"),
            IndicatorParam("kc_mult", "float", 1.0, 0.5, 5.0, "Keltner ATR multiplier"),
            IndicatorParam("percentile_window", "int", 120, 20, 1000, "Percentile window"),
        ],
        outputs=[IndicatorOutput("squeeze_score", "Squeeze score (0-1)")],
        talib_func=None,
        pandas_ta_func=None,
        required_columns=["high", "low", "close"],
    )
)
def _calc_squeeze_score(
    data: pd.DataFrame,
    use_talib: bool = True,
    length: int = 20,
    bb_mult: float = 2.0,
    kc_mult: float = 1.0,
    percentile_window: int = 120,
    **kwargs: Any,
) -> pd.Series:
    """Compression score from Bollinger and Keltner width percentiles."""
    del use_talib, kwargs
    lookback = int(length)
    min_periods = _rolling_min_periods(lookback)

    mid = data["close"].rolling(lookback, min_periods=min_periods).mean()
    std = data["close"].rolling(lookback, min_periods=min_periods).std()
    upper = mid + (float(bb_mult) * std)
    lower = mid - (float(bb_mult) * std)
    bb_width = _safe_divide(upper - lower, mid)

    atr_series = _atr(data, lookback)
    kc_width = _safe_divide(2.0 * float(kc_mult) * atr_series, mid)

    window = int(percentile_window)
    bb_pct = _rolling_percentile_rank(bb_width, window)
    kc_pct = _rolling_percentile_rank(kc_width, window)

    score = ((1.0 - bb_pct) + (1.0 - kc_pct)) / 2.0
    return score.fillna(0.0).clip(lower=0.0, upper=1.0)


@indicator(
    metadata=IndicatorMetadata(
        name="parkinson_volatility",
        full_name="Parkinson Volatility",
        category=IndicatorCategory.VOLATILITY,
        description="Range-based volatility estimator using high/low prices",
        params=[IndicatorParam("length", "int", 20, 5, 500, "Lookback window")],
        outputs=[IndicatorOutput("parkinson_volatility", "Parkinson volatility estimate")],
        talib_func=None,
        pandas_ta_func=None,
        required_columns=["high", "low"],
    )
)
def _calc_parkinson_volatility(
    data: pd.DataFrame,
    use_talib: bool = True,
    length: int = 20,
    **kwargs: Any,
) -> pd.Series:
    """Range-based Parkinson volatility estimator."""
    del use_talib, kwargs
    lookback = int(length)
    hl_log = np.log((data["high"] / data["low"]).replace(0.0, np.nan))
    variance = (hl_log**2).rolling(
        lookback,
        min_periods=_rolling_min_periods(lookback),
    ).mean() / (4.0 * math.log(2.0))
    return np.sqrt(variance.clip(lower=0.0)).fillna(0.0)


@indicator(
    metadata=IndicatorMetadata(
        name="garman_klass_volatility",
        full_name="Garman-Klass Volatility",
        category=IndicatorCategory.VOLATILITY,
        description="OHLC-based volatility estimator robust to drift",
        params=[IndicatorParam("length", "int", 20, 5, 500, "Lookback window")],
        outputs=[IndicatorOutput("garman_klass_volatility", "Garman-Klass volatility estimate")],
        talib_func=None,
        pandas_ta_func=None,
        required_columns=["open", "high", "low", "close"],
    )
)
def _calc_garman_klass_volatility(
    data: pd.DataFrame,
    use_talib: bool = True,
    length: int = 20,
    **kwargs: Any,
) -> pd.Series:
    """OHLC-based Garman-Klass volatility estimator."""
    del use_talib, kwargs
    lookback = int(length)
    hl_log = np.log((data["high"] / data["low"]).replace(0.0, np.nan))
    co_log = np.log((data["close"] / data["open"]).replace(0.0, np.nan))
    variance = (
        0.5 * (hl_log**2)
        - (2.0 * math.log(2.0) - 1.0) * (co_log**2)
    ).rolling(
        lookback,
        min_periods=_rolling_min_periods(lookback),
    ).mean()
    return np.sqrt(variance.clip(lower=0.0)).fillna(0.0)


@indicator(
    metadata=IndicatorMetadata(
        name="dry_up_reversal_hint",
        full_name="Dry-Up Reversal Hint",
        category=IndicatorCategory.VOLUME,
        description="Rolling probability of reversal after low-volume bars",
        params=[
            IndicatorParam("length", "int", 120, 20, 1000, "Lookback window"),
            IndicatorParam("quantile", "float", 0.2, 0.01, 0.5, "Dry-up volume quantile"),
            IndicatorParam("reversal_bars", "int", 1, 1, 10, "Bars waited before reversal is evaluated"),
        ],
        outputs=[IndicatorOutput("dry_up_reversal_hint", "Dry-up reversal probability (0-1)")],
        talib_func=None,
        pandas_ta_func=None,
        required_columns=["close", "volume"],
    )
)
def _calc_dry_up_reversal_hint(
    data: pd.DataFrame,
    use_talib: bool = True,
    length: int = 120,
    quantile: float = 0.2,
    reversal_bars: int = 1,
    source: str = "close",
    **kwargs: Any,
) -> pd.Series:
    """Causal-safe frequency of dry-up bars followed by reversal after maturation."""
    del use_talib, kwargs
    lookback = int(length)
    q = float(quantile)
    threshold = data["volume"].rolling(
        lookback,
        min_periods=_rolling_min_periods(lookback),
    ).quantile(q)
    dry_up = (data["volume"] <= threshold).fillna(False)

    maturity = int(reversal_bars)
    sign_returns = np.sign(data[source].pct_change().fillna(0.0))
    matured_dry_up = pd.Series(0.0, index=data.index, dtype=float)
    matured_reversal = pd.Series(0.0, index=data.index, dtype=float)

    for current_idx in range(len(data)):
        dry_up_idx = current_idx - maturity
        if dry_up_idx <= 0:
            continue
        if not bool(dry_up.iloc[dry_up_idx]):
            continue
        matured_dry_up.iloc[current_idx] = 1.0
        prev_sign = float(sign_returns.iloc[dry_up_idx - 1])
        if prev_sign == 0.0:
            continue
        path = sign_returns.iloc[dry_up_idx + 1 : current_idx + 1]
        reversed_seen = bool((path == -prev_sign).any())
        if reversed_seen:
            matured_reversal.iloc[current_idx] = 1.0

    numerator = matured_reversal.rolling(
        lookback,
        min_periods=_rolling_min_periods(lookback),
    ).sum()
    denominator = matured_dry_up.rolling(
        lookback,
        min_periods=_rolling_min_periods(lookback),
    ).sum()
    return _safe_divide(numerator, denominator).fillna(0.0).clip(lower=0.0, upper=1.0)
