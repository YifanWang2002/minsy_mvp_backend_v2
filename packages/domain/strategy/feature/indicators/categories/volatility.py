"""Volatility Indicators.

This module contains volatility-based indicators.
Priority: Use TA-Lib if available, fallback to pandas-ta.
"""

from typing import Any, Union

import numpy as np
import pandas as pd

from ..base import IndicatorCategory, IndicatorMetadata, IndicatorOutput, IndicatorParam
from ..registry import IndicatorRegistry

# Check library availability
try:
    import talib
    _TALIB = True
except ImportError:
    _TALIB = False

try:
    import pandas_ta as ta
    _PANDAS_TA = True
except ImportError:
    _PANDAS_TA = False


# =============================================================================
# True Range
# =============================================================================
def _calc_trange(data: pd.DataFrame, use_talib: bool = True, **kwargs) -> pd.Series:
    """Calculate True Range."""
    if use_talib and _TALIB:
        return pd.Series(
            talib.TRANGE(data["high"], data["low"], data["close"]),
            index=data.index, name="TRANGE"
        )
    elif _PANDAS_TA:
        return ta.true_range(data["high"], data["low"], data["close"])
    else:
        high_low = data["high"] - data["low"]
        high_close = abs(data["high"] - data["close"].shift(1))
        low_close = abs(data["low"] - data["close"].shift(1))
        return pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

IndicatorRegistry.register(
    IndicatorMetadata(
        name="trange",
        full_name="True Range",
        category=IndicatorCategory.VOLATILITY,
        description="Maximum of (H-L, |H-Prev Close|, |L-Prev Close|)",
        params=[],
        outputs=[IndicatorOutput("trange", "True Range values")],
        talib_func="TRANGE",
        pandas_ta_func="true_range",
        required_columns=["high", "low", "close"],
    ),
    _calc_trange
)


# =============================================================================
# Average True Range (ATR)
# =============================================================================
def _calc_atr(data: pd.DataFrame, use_talib: bool = True,
              length: int = 14, **kwargs) -> pd.Series:
    """Calculate Average True Range."""
    if use_talib and _TALIB:
        return pd.Series(
            talib.ATR(data["high"], data["low"], data["close"], timeperiod=length),
            index=data.index, name=f"ATR_{length}"
        )
    elif _PANDAS_TA:
        return ta.atr(data["high"], data["low"], data["close"], length=length)
    else:
        high_low = data["high"] - data["low"]
        high_close = abs(data["high"] - data["close"].shift(1))
        low_close = abs(data["low"] - data["close"].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window=length).mean()

IndicatorRegistry.register(
    IndicatorMetadata(
        name="atr",
        full_name="Average True Range",
        category=IndicatorCategory.VOLATILITY,
        description="Smoothed average of True Range",
        params=[
            IndicatorParam("length", "int", 14, 1, 100, "Period length"),
        ],
        outputs=[IndicatorOutput("atr", "ATR values")],
        talib_func="ATR",
        pandas_ta_func="atr",
        required_columns=["high", "low", "close"],
    ),
    _calc_atr
)


# =============================================================================
# Normalized Average True Range (NATR)
# =============================================================================
def _calc_natr(data: pd.DataFrame, use_talib: bool = True,
               length: int = 14, **kwargs) -> pd.Series:
    """Calculate Normalized Average True Range."""
    if use_talib and _TALIB:
        return pd.Series(
            talib.NATR(data["high"], data["low"], data["close"], timeperiod=length),
            index=data.index, name=f"NATR_{length}"
        )
    elif _PANDAS_TA:
        return ta.natr(data["high"], data["low"], data["close"], length=length)
    else:
        atr = _calc_atr(data, use_talib=False, length=length)
        return 100 * atr / data["close"]

IndicatorRegistry.register(
    IndicatorMetadata(
        name="natr",
        full_name="Normalized Average True Range",
        category=IndicatorCategory.VOLATILITY,
        description="ATR expressed as percentage of close price",
        params=[
            IndicatorParam("length", "int", 14, 1, 100, "Period length"),
        ],
        outputs=[IndicatorOutput("natr", "NATR values (%)")],
        talib_func="NATR",
        pandas_ta_func="natr",
        required_columns=["high", "low", "close"],
    ),
    _calc_natr
)


# =============================================================================
# Chandelier Exit - pandas-ta only
# =============================================================================
def _calc_chandelier(data: pd.DataFrame, use_talib: bool = True,
                     length: int = 22, atr_length: int = 14,
                     multiplier: float = 3.0, **kwargs) -> pd.DataFrame:
    """Calculate Chandelier Exit."""
    if _PANDAS_TA:
        return ta.chandelier_exit(
            data["high"], data["low"], data["close"],
            high_length=length, low_length=length,
            atr_length=atr_length, multiplier=multiplier
        )
    else:
        atr = _calc_atr(data, use_talib=False, length=atr_length)
        highest_high = data["high"].rolling(window=length).max()
        lowest_low = data["low"].rolling(window=length).min()
        return pd.DataFrame({
            "CHANDELIERl": highest_high - multiplier * atr,
            "CHANDELIERs": lowest_low + multiplier * atr,
        }, index=data.index)

IndicatorRegistry.register(
    IndicatorMetadata(
        name="chandelier",
        full_name="Chandelier Exit",
        category=IndicatorCategory.VOLATILITY,
        description="ATR-based trailing stop",
        params=[
            IndicatorParam("length", "int", 22, 1, 100, "Lookback period"),
            IndicatorParam("atr_length", "int", 14, 1, 100, "ATR period"),
            IndicatorParam("multiplier", "float", 3.0, 0.5, 10.0, "ATR multiplier"),
        ],
        outputs=[
            IndicatorOutput("CHANDELIERl", "Long exit (sell stop)"),
            IndicatorOutput("CHANDELIERs", "Short exit (buy stop)"),
        ],
        talib_func=None,
        pandas_ta_func="chandelier_exit",
        required_columns=["high", "low", "close"],
    ),
    _calc_chandelier
)


# =============================================================================
# Mass Index - pandas-ta only
# =============================================================================
def _calc_massi(data: pd.DataFrame, use_talib: bool = True,
                fast: int = 9, slow: int = 25, **kwargs) -> pd.Series:
    """Calculate Mass Index."""
    if _PANDAS_TA:
        return ta.massi(data["high"], data["low"], fast=fast, slow=slow)
    else:
        high_low = data["high"] - data["low"]
        ema1 = high_low.ewm(span=fast, adjust=False).mean()
        ema2 = ema1.ewm(span=fast, adjust=False).mean()
        ratio = ema1 / ema2
        return ratio.rolling(window=slow).sum()

IndicatorRegistry.register(
    IndicatorMetadata(
        name="massi",
        full_name="Mass Index",
        category=IndicatorCategory.VOLATILITY,
        description="Identifies reversals based on range expansion",
        params=[
            IndicatorParam("fast", "int", 9, 1, 50, "Fast EMA period"),
            IndicatorParam("slow", "int", 25, 1, 100, "Sum period"),
        ],
        outputs=[IndicatorOutput("massi", "Mass Index values")],
        talib_func=None,
        pandas_ta_func="massi",
        required_columns=["high", "low"],
    ),
    _calc_massi
)


# =============================================================================
# Ulcer Index - pandas-ta only
# =============================================================================
def _calc_ui(data: pd.DataFrame, use_talib: bool = True, source: str = "close",
             length: int = 14, **kwargs) -> pd.Series:
    """Calculate Ulcer Index."""
    src = data[source]
    if _PANDAS_TA:
        return ta.ui(src, length=length)
    else:
        rolling_max = src.rolling(window=length).max()
        pct_drawdown = 100 * (src - rolling_max) / rolling_max
        return np.sqrt((pct_drawdown ** 2).rolling(window=length).mean())

IndicatorRegistry.register(
    IndicatorMetadata(
        name="ui",
        full_name="Ulcer Index",
        category=IndicatorCategory.VOLATILITY,
        description="Measures downside volatility",
        params=[
            IndicatorParam("length", "int", 14, 1, 100, "Period length"),
        ],
        outputs=[IndicatorOutput("ui", "Ulcer Index values")],
        talib_func=None,
        pandas_ta_func="ui",
        required_columns=["close"],
    ),
    _calc_ui
)


# =============================================================================
# ATR Trailing Stop - pandas-ta only
# =============================================================================
def _calc_atrts(data: pd.DataFrame, use_talib: bool = True,
                length: int = 14, multiplier: float = 3.0, **kwargs) -> pd.Series:
    """Calculate ATR Trailing Stop."""
    if _PANDAS_TA:
        return ta.atrts(data["high"], data["low"], data["close"],
                        length=length, k=multiplier)
    else:
        raise NotImplementedError("ATR Trailing Stop requires pandas-ta")

IndicatorRegistry.register(
    IndicatorMetadata(
        name="atrts",
        full_name="ATR Trailing Stop",
        category=IndicatorCategory.VOLATILITY,
        description="ATR-based trailing stop level",
        params=[
            IndicatorParam("length", "int", 14, 1, 100, "ATR period"),
            IndicatorParam("multiplier", "float", 3.0, 0.5, 10.0, "ATR multiplier"),
        ],
        outputs=[IndicatorOutput("atrts", "ATR trailing stop values")],
        talib_func=None,
        pandas_ta_func="atrts",
        required_columns=["high", "low", "close"],
    ),
    _calc_atrts
)


# =============================================================================
# Relative Volatility Index (RVI) - pandas-ta only
# =============================================================================
def _calc_rvi(data: pd.DataFrame, use_talib: bool = True, source: str = "close",
              length: int = 14, **kwargs) -> pd.Series:
    """Calculate Relative Volatility Index."""
    src = data[source]
    if _PANDAS_TA:
        result = ta.rvi(src, length=length)
        if isinstance(result, pd.DataFrame):
            return result.iloc[:, 0]
        return result
    else:
        raise NotImplementedError("RVI requires pandas-ta")

IndicatorRegistry.register(
    IndicatorMetadata(
        name="rvi",
        full_name="Relative Volatility Index",
        category=IndicatorCategory.VOLATILITY,
        description="RSI applied to standard deviation",
        params=[
            IndicatorParam("length", "int", 14, 1, 100, "Period length"),
        ],
        outputs=[IndicatorOutput("rvi", "RVI values (0-100)")],
        talib_func=None,
        pandas_ta_func="rvi",
        required_columns=["close"],
    ),
    _calc_rvi
)


# =============================================================================
# Price Distance - pandas-ta only
# =============================================================================
def _calc_pdist(data: pd.DataFrame, use_talib: bool = True, **kwargs) -> pd.Series:
    """Calculate Price Distance."""
    if _PANDAS_TA:
        return ta.pdist(data["open"], data["high"], data["low"], data["close"])
    else:
        return (data["high"] - data["low"]) / data["close"]

IndicatorRegistry.register(
    IndicatorMetadata(
        name="pdist",
        full_name="Price Distance",
        category=IndicatorCategory.VOLATILITY,
        description="Measures bar range relative to close",
        params=[],
        outputs=[IndicatorOutput("pdist", "Price distance values")],
        talib_func=None,
        pandas_ta_func="pdist",
        required_columns=["open", "high", "low", "close"],
    ),
    _calc_pdist
)


# =============================================================================
# Historical Volatility (Standard Deviation)
# =============================================================================
def _calc_stdev(data: pd.DataFrame, use_talib: bool = True, source: str = "close",
                length: int = 20, **kwargs) -> pd.Series:
    """Calculate Standard Deviation (Historical Volatility)."""
    src = data[source]
    if use_talib and _TALIB:
        return pd.Series(
            talib.STDDEV(src, timeperiod=length),
            index=data.index, name=f"STDEV_{length}"
        )
    elif _PANDAS_TA:
        return ta.stdev(src, length=length)
    else:
        return src.rolling(window=length).std()

IndicatorRegistry.register(
    IndicatorMetadata(
        name="stdev",
        full_name="Standard Deviation",
        category=IndicatorCategory.VOLATILITY,
        description="Historical volatility measure",
        params=[
            IndicatorParam("length", "int", 20, 1, 500, "Period length"),
        ],
        outputs=[IndicatorOutput("stdev", "Standard deviation values")],
        talib_func="STDDEV",
        pandas_ta_func="stdev",
        required_columns=["close"],
    ),
    _calc_stdev
)


# =============================================================================
# Variance
# =============================================================================
def _calc_variance(data: pd.DataFrame, use_talib: bool = True, source: str = "close",
                   length: int = 20, **kwargs) -> pd.Series:
    """Calculate Variance."""
    src = data[source]
    if use_talib and _TALIB:
        return pd.Series(
            talib.VAR(src, timeperiod=length),
            index=data.index, name=f"VAR_{length}"
        )
    elif _PANDAS_TA:
        return ta.variance(src, length=length)
    else:
        return src.rolling(window=length).var()

IndicatorRegistry.register(
    IndicatorMetadata(
        name="variance",
        full_name="Variance",
        category=IndicatorCategory.VOLATILITY,
        description="Squared standard deviation",
        params=[
            IndicatorParam("length", "int", 20, 1, 500, "Period length"),
        ],
        outputs=[IndicatorOutput("variance", "Variance values")],
        talib_func="VAR",
        pandas_ta_func="variance",
        required_columns=["close"],
    ),
    _calc_variance
)
