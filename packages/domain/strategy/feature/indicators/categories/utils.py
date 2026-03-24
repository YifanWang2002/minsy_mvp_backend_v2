"""Utility Indicators (Statistics, Math, Cycle, Performance).

This module contains utility functions for statistical analysis,
mathematical transformations, cycle analysis, and performance metrics.
Priority: Use TA-Lib if available, fallback to pandas-ta.
"""

from typing import Any, Union

import numpy as np
import pandas as pd

from ..base import IndicatorCategory, IndicatorMetadata, IndicatorOutput, IndicatorParam
from ..decorators import indicator

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
# STATISTICS
# =============================================================================

# Linear Regression Slope
@indicator(
    metadata=IndicatorMetadata(
        name="linearreg_slope",
        full_name="Linear Regression Slope",
        category=IndicatorCategory.UTILS,
        description="Slope of linear regression line",
        params=[IndicatorParam("length", "int", 14, 1, 500, "Period length")],
        outputs=[IndicatorOutput("slope", "Regression slope")],
        talib_func="LINEARREG_SLOPE",
        pandas_ta_func="linreg",
        required_columns=["close"],
    )
)
def _calc_linearreg_slope(data: pd.DataFrame, use_talib: bool = True, source: str = "close",
                          length: int = 14, **kwargs) -> pd.Series:
    """Calculate Linear Regression Slope."""
    src = data[source]
    if use_talib and _TALIB:
        return pd.Series(talib.LINEARREG_SLOPE(src, timeperiod=length),
                        index=data.index, name=f"LINREG_SLOPE_{length}")
    elif _PANDAS_TA:
        result = ta.linreg(src, length=length, slope=True)
        if isinstance(result, pd.DataFrame):
            for col in result.columns:
                if "SLOPE" in col.upper():
                    return result[col]
        return result
    else:
        raise NotImplementedError("Linear Regression Slope requires TA-Lib or pandas-ta")


# Linear Regression Angle
@indicator(
    metadata=IndicatorMetadata(
        name="linearreg_angle",
        full_name="Linear Regression Angle",
        category=IndicatorCategory.UTILS,
        description="Angle of linear regression line in degrees",
        params=[IndicatorParam("length", "int", 14, 1, 500, "Period length")],
        outputs=[IndicatorOutput("angle", "Regression angle (degrees)")],
        talib_func="LINEARREG_ANGLE",
        pandas_ta_func="linreg",
        required_columns=["close"],
    )
)
def _calc_linearreg_angle(data: pd.DataFrame, use_talib: bool = True, source: str = "close",
                          length: int = 14, **kwargs) -> pd.Series:
    """Calculate Linear Regression Angle."""
    src = data[source]
    if use_talib and _TALIB:
        return pd.Series(talib.LINEARREG_ANGLE(src, timeperiod=length),
                        index=data.index, name=f"LINREG_ANGLE_{length}")
    elif _PANDAS_TA:
        result = ta.linreg(src, length=length, angle=True)
        if isinstance(result, pd.DataFrame):
            for col in result.columns:
                if "ANGLE" in col.upper():
                    return result[col]
        return result
    else:
        raise NotImplementedError("Linear Regression Angle requires TA-Lib or pandas-ta")


# Linear Regression Intercept
@indicator(
    metadata=IndicatorMetadata(
        name="linearreg_intercept",
        full_name="Linear Regression Intercept",
        category=IndicatorCategory.UTILS,
        description="Intercept of linear regression line",
        params=[IndicatorParam("length", "int", 14, 1, 500, "Period length")],
        outputs=[IndicatorOutput("intercept", "Regression intercept")],
        talib_func="LINEARREG_INTERCEPT",
        pandas_ta_func="linreg",
        required_columns=["close"],
    )
)
def _calc_linearreg_intercept(data: pd.DataFrame, use_talib: bool = True, source: str = "close",
                              length: int = 14, **kwargs) -> pd.Series:
    """Calculate Linear Regression Intercept."""
    src = data[source]
    if use_talib and _TALIB:
        return pd.Series(talib.LINEARREG_INTERCEPT(src, timeperiod=length),
                        index=data.index, name=f"LINREG_INTERCEPT_{length}")
    elif _PANDAS_TA:
        result = ta.linreg(src, length=length, intercept=True)
        if isinstance(result, pd.DataFrame):
            for col in result.columns:
                if "INTERCEPT" in col.upper():
                    return result[col]
        return result
    else:
        raise NotImplementedError("Linear Regression Intercept requires TA-Lib or pandas-ta")


# Time Series Forecast
@indicator(
    metadata=IndicatorMetadata(
        name="tsf",
        full_name="Time Series Forecast",
        category=IndicatorCategory.UTILS,
        description="Linear regression endpoint forecast",
        params=[IndicatorParam("length", "int", 14, 1, 500, "Period length")],
        outputs=[IndicatorOutput("tsf", "Time series forecast")],
        talib_func="TSF",
        pandas_ta_func="linreg",
        required_columns=["close"],
    )
)
def _calc_tsf(data: pd.DataFrame, use_talib: bool = True, source: str = "close",
              length: int = 14, **kwargs) -> pd.Series:
    """Calculate Time Series Forecast."""
    src = data[source]
    if use_talib and _TALIB:
        return pd.Series(talib.TSF(src, timeperiod=length),
                        index=data.index, name=f"TSF_{length}")
    elif _PANDAS_TA:
        result = ta.linreg(src, length=length, tsf=True)
        if isinstance(result, pd.DataFrame):
            for col in result.columns:
                if "TSF" in col.upper():
                    return result[col]
        return result
    else:
        raise NotImplementedError("TSF requires TA-Lib or pandas-ta")


# Z-Score
@indicator(
    metadata=IndicatorMetadata(
        name="zscore",
        full_name="Z-Score",
        category=IndicatorCategory.UTILS,
        description="Number of standard deviations from mean",
        params=[IndicatorParam("length", "int", 30, 2, 500, "Period length")],
        outputs=[IndicatorOutput("zscore", "Z-score values")],
        talib_func=None,
        pandas_ta_func="zscore",
        required_columns=["close"],
    )
)
def _calc_zscore(data: pd.DataFrame, use_talib: bool = True, source: str = "close",
                 length: int = 30, **kwargs) -> pd.Series:
    """Calculate Z-Score."""
    src = data[source]
    if _PANDAS_TA:
        return ta.zscore(src, length=length)
    else:
        mean = src.rolling(window=length).mean()
        std = src.rolling(window=length).std()
        return (src - mean) / std


# Entropy
@indicator(
    metadata=IndicatorMetadata(
        name="entropy",
        full_name="Entropy",
        category=IndicatorCategory.UTILS,
        description="Shannon entropy of price distribution",
        params=[
            IndicatorParam("length", "int", 10, 2, 100, "Period length"),
            IndicatorParam("base", "int", 2, 2, 10, "Logarithm base"),
        ],
        outputs=[IndicatorOutput("entropy", "Entropy values")],
        talib_func=None,
        pandas_ta_func="entropy",
        required_columns=["close"],
    )
)
def _calc_entropy(data: pd.DataFrame, use_talib: bool = True, source: str = "close",
                  length: int = 10, base: int = 2, **kwargs) -> pd.Series:
    """Calculate Entropy."""
    src = data[source]
    if _PANDAS_TA:
        return ta.entropy(src, length=length, base=base)
    else:
        raise NotImplementedError("Entropy requires pandas-ta")


# Kurtosis
@indicator(
    metadata=IndicatorMetadata(
        name="kurtosis",
        full_name="Kurtosis",
        category=IndicatorCategory.UTILS,
        description="Measures tailedness of distribution",
        params=[IndicatorParam("length", "int", 30, 4, 500, "Period length")],
        outputs=[IndicatorOutput("kurtosis", "Kurtosis values")],
        talib_func=None,
        pandas_ta_func="kurtosis",
        required_columns=["close"],
    )
)
def _calc_kurtosis(data: pd.DataFrame, use_talib: bool = True, source: str = "close",
                   length: int = 30, **kwargs) -> pd.Series:
    """Calculate Kurtosis."""
    src = data[source]
    if _PANDAS_TA:
        return ta.kurtosis(src, length=length)
    else:
        return src.rolling(window=length).kurt()


# Skew
@indicator(
    metadata=IndicatorMetadata(
        name="skew",
        full_name="Skewness",
        category=IndicatorCategory.UTILS,
        description="Measures asymmetry of distribution",
        params=[IndicatorParam("length", "int", 30, 3, 500, "Period length")],
        outputs=[IndicatorOutput("skew", "Skewness values")],
        talib_func=None,
        pandas_ta_func="skew",
        required_columns=["close"],
    )
)
def _calc_skew(data: pd.DataFrame, use_talib: bool = True, source: str = "close",
               length: int = 30, **kwargs) -> pd.Series:
    """Calculate Skewness."""
    src = data[source]
    if _PANDAS_TA:
        return ta.skew(src, length=length)
    else:
        return src.rolling(window=length).skew()


# Median
@indicator(
    metadata=IndicatorMetadata(
        name="median",
        full_name="Rolling Median",
        category=IndicatorCategory.UTILS,
        description="Median value over rolling window",
        params=[IndicatorParam("length", "int", 30, 1, 500, "Period length")],
        outputs=[IndicatorOutput("median", "Median values")],
        talib_func=None,
        pandas_ta_func="median",
        required_columns=["close"],
    )
)
def _calc_median(data: pd.DataFrame, use_talib: bool = True, source: str = "close",
                 length: int = 30, **kwargs) -> pd.Series:
    """Calculate Rolling Median."""
    src = data[source]
    if _PANDAS_TA:
        return ta.median(src, length=length)
    else:
        return src.rolling(window=length).median()


# Quantile
@indicator(
    metadata=IndicatorMetadata(
        name="quantile",
        full_name="Rolling Quantile",
        category=IndicatorCategory.UTILS,
        description="Quantile value over rolling window",
        params=[
            IndicatorParam("length", "int", 30, 1, 500, "Period length"),
            IndicatorParam("q", "float", 0.5, 0, 1, "Quantile (0-1)"),
        ],
        outputs=[IndicatorOutput("quantile", "Quantile values")],
        talib_func=None,
        pandas_ta_func="quantile",
        required_columns=["close"],
    )
)
def _calc_quantile(data: pd.DataFrame, use_talib: bool = True, source: str = "close",
                   length: int = 30, q: float = 0.5, **kwargs) -> pd.Series:
    """Calculate Rolling Quantile."""
    src = data[source]
    if _PANDAS_TA:
        return ta.quantile(src, length=length, q=q)
    else:
        return src.rolling(window=length).quantile(q)


# =============================================================================
# MATH OPERATORS
# =============================================================================

# MAX
@indicator(
    metadata=IndicatorMetadata(
        name="max",
        full_name="Highest Value over Period",
        category=IndicatorCategory.UTILS,
        description="Maximum value over rolling window",
        params=[IndicatorParam("length", "int", 30, 1, 500, "Period length")],
        outputs=[IndicatorOutput("max", "Maximum values")],
        talib_func="MAX",
        pandas_ta_func=None,
        required_columns=["close"],
    )
)
def _calc_max(data: pd.DataFrame, use_talib: bool = True, source: str = "close",
              length: int = 30, **kwargs) -> pd.Series:
    """Calculate Rolling Maximum."""
    src = data[source]
    if use_talib and _TALIB:
        return pd.Series(talib.MAX(src, timeperiod=length),
                        index=data.index, name=f"MAX_{length}")
    else:
        return src.rolling(window=length).max()


# MIN
@indicator(
    metadata=IndicatorMetadata(
        name="min",
        full_name="Lowest Value over Period",
        category=IndicatorCategory.UTILS,
        description="Minimum value over rolling window",
        params=[IndicatorParam("length", "int", 30, 1, 500, "Period length")],
        outputs=[IndicatorOutput("min", "Minimum values")],
        talib_func="MIN",
        pandas_ta_func=None,
        required_columns=["close"],
    )
)
def _calc_min(data: pd.DataFrame, use_talib: bool = True, source: str = "close",
              length: int = 30, **kwargs) -> pd.Series:
    """Calculate Rolling Minimum."""
    src = data[source]
    if use_talib and _TALIB:
        return pd.Series(talib.MIN(src, timeperiod=length),
                        index=data.index, name=f"MIN_{length}")
    else:
        return src.rolling(window=length).min()


# SUM
@indicator(
    metadata=IndicatorMetadata(
        name="sum",
        full_name="Summation",
        category=IndicatorCategory.UTILS,
        description="Sum over rolling window",
        params=[IndicatorParam("length", "int", 30, 1, 500, "Period length")],
        outputs=[IndicatorOutput("sum", "Sum values")],
        talib_func="SUM",
        pandas_ta_func=None,
        required_columns=["close"],
    )
)
def _calc_sum(data: pd.DataFrame, use_talib: bool = True, source: str = "close",
              length: int = 30, **kwargs) -> pd.Series:
    """Calculate Rolling Sum."""
    src = data[source]
    if use_talib and _TALIB:
        return pd.Series(talib.SUM(src, timeperiod=length),
                        index=data.index, name=f"SUM_{length}")
    else:
        return src.rolling(window=length).sum()


# =============================================================================
# PRICE TRANSFORM
# =============================================================================

# Average Price
@indicator(
    metadata=IndicatorMetadata(
        name="avgprice",
        full_name="Average Price",
        category=IndicatorCategory.UTILS,
        description="(Open + High + Low + Close) / 4",
        params=[],
        outputs=[IndicatorOutput("avgprice", "Average price")],
        talib_func="AVGPRICE",
        pandas_ta_func=None,
        required_columns=["open", "high", "low", "close"],
    )
)
def _calc_avgprice(data: pd.DataFrame, use_talib: bool = True, **kwargs) -> pd.Series:
    """Calculate Average Price."""
    if use_talib and _TALIB:
        return pd.Series(
            talib.AVGPRICE(data["open"], data["high"], data["low"], data["close"]),
            index=data.index, name="AVGPRICE"
        )
    else:
        return (data["open"] + data["high"] + data["low"] + data["close"]) / 4


# Median Price
@indicator(
    metadata=IndicatorMetadata(
        name="medprice",
        full_name="Median Price",
        category=IndicatorCategory.UTILS,
        description="(High + Low) / 2",
        params=[],
        outputs=[IndicatorOutput("medprice", "Median price")],
        talib_func="MEDPRICE",
        pandas_ta_func=None,
        required_columns=["high", "low"],
    )
)
def _calc_medprice(data: pd.DataFrame, use_talib: bool = True, **kwargs) -> pd.Series:
    """Calculate Median Price."""
    if use_talib and _TALIB:
        return pd.Series(
            talib.MEDPRICE(data["high"], data["low"]),
            index=data.index, name="MEDPRICE"
        )
    else:
        return (data["high"] + data["low"]) / 2


# =============================================================================
# PERFORMANCE
# =============================================================================

# Log Return
@indicator(
    metadata=IndicatorMetadata(
        name="log_return",
        full_name="Log Return",
        category=IndicatorCategory.UTILS,
        description="Logarithmic return",
        params=[
            IndicatorParam("length", "int", 1, 1, 500, "Period length"),
            IndicatorParam("cumulative", "bool", False, description="Cumulative sum"),
        ],
        outputs=[IndicatorOutput("log_return", "Log return values")],
        talib_func=None,
        pandas_ta_func="log_return",
        required_columns=["close"],
    )
)
def _calc_log_return(data: pd.DataFrame, use_talib: bool = True, source: str = "close",
                     length: int = 1, cumulative: bool = False, **kwargs) -> pd.Series:
    """Calculate Log Return."""
    src = data[source]
    if _PANDAS_TA:
        return ta.log_return(src, length=length, cumulative=cumulative)
    else:
        log_ret = np.log(src / src.shift(length))
        if cumulative:
            return log_ret.cumsum()
        return log_ret


# Percent Return
@indicator(
    metadata=IndicatorMetadata(
        name="percent_return",
        full_name="Percent Return",
        category=IndicatorCategory.UTILS,
        description="Percentage return",
        params=[
            IndicatorParam("length", "int", 1, 1, 500, "Period length"),
            IndicatorParam("cumulative", "bool", False, description="Cumulative product"),
        ],
        outputs=[IndicatorOutput("percent_return", "Percent return values")],
        talib_func=None,
        pandas_ta_func="percent_return",
        required_columns=["close"],
    )
)
def _calc_percent_return(data: pd.DataFrame, use_talib: bool = True, source: str = "close",
                         length: int = 1, cumulative: bool = False, **kwargs) -> pd.Series:
    """Calculate Percent Return."""
    src = data[source]
    if _PANDAS_TA:
        return ta.percent_return(src, length=length, cumulative=cumulative)
    else:
        pct_ret = src.pct_change(periods=length) * 100
        if cumulative:
            return ((1 + pct_ret / 100).cumprod() - 1) * 100
        return pct_ret


# Drawdown
@indicator(
    metadata=IndicatorMetadata(
        name="drawdown",
        full_name="Drawdown",
        category=IndicatorCategory.UTILS,
        description="Drawdown from peak",
        params=[],
        outputs=[
            IndicatorOutput("DD", "Drawdown"),
            IndicatorOutput("DDp", "Drawdown %"),
            IndicatorOutput("DDmax", "Maximum drawdown"),
        ],
        talib_func=None,
        pandas_ta_func="drawdown",
        required_columns=["close"],
    )
)
def _calc_drawdown(data: pd.DataFrame, use_talib: bool = True, source: str = "close",
                   **kwargs) -> pd.DataFrame:
    """Calculate Drawdown."""
    src = data[source]
    if _PANDAS_TA:
        return ta.drawdown(src)
    else:
        running_max = src.cummax()
        dd = src / running_max - 1
        return pd.DataFrame({
            "DD": dd,
            "DDp": dd * 100,
            "DDmax": dd.cummin()
        }, index=data.index)

