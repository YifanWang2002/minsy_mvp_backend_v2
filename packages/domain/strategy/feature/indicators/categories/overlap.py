"""Overlap Studies / Moving Averages indicators.

This module contains moving averages and overlay indicators.
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
# Simple Moving Average (SMA)
# =============================================================================
def _calc_sma(data: pd.DataFrame, use_talib: bool = True, source: str = "close",
              length: int = 20, **kwargs) -> pd.Series:
    """Calculate Simple Moving Average."""
    src = data[source]
    if use_talib and _TALIB:
        return pd.Series(talib.SMA(src, timeperiod=length), index=data.index, name=f"SMA_{length}")
    elif _PANDAS_TA:
        return ta.sma(src, length=length)
    else:
        return src.rolling(window=length).mean()

IndicatorRegistry.register(
    IndicatorMetadata(
        name="sma",
        full_name="Simple Moving Average",
        category=IndicatorCategory.OVERLAP,
        description="Average price over a specified period",
        params=[
            IndicatorParam("length", "int", 20, 1, 500, "Period length"),
        ],
        outputs=[IndicatorOutput("sma", "SMA values")],
        talib_func="SMA",
        pandas_ta_func="sma",
        required_columns=["close"],
    ),
    _calc_sma
)


# =============================================================================
# Exponential Moving Average (EMA)
# =============================================================================
def _calc_ema(data: pd.DataFrame, use_talib: bool = True, source: str = "close",
              length: int = 20, **kwargs) -> pd.Series:
    """Calculate Exponential Moving Average."""
    src = data[source]
    if use_talib and _TALIB:
        return pd.Series(talib.EMA(src, timeperiod=length), index=data.index, name=f"EMA_{length}")
    elif _PANDAS_TA:
        return ta.ema(src, length=length)
    else:
        return src.ewm(span=length, adjust=False).mean()

IndicatorRegistry.register(
    IndicatorMetadata(
        name="ema",
        full_name="Exponential Moving Average",
        category=IndicatorCategory.OVERLAP,
        description="Weighted moving average with exponential decay",
        params=[
            IndicatorParam("length", "int", 20, 1, 500, "Period length"),
        ],
        outputs=[IndicatorOutput("ema", "EMA values")],
        talib_func="EMA",
        pandas_ta_func="ema",
        required_columns=["close"],
    ),
    _calc_ema
)


# =============================================================================
# Double Exponential Moving Average (DEMA)
# =============================================================================
def _calc_dema(data: pd.DataFrame, use_talib: bool = True, source: str = "close",
               length: int = 20, **kwargs) -> pd.Series:
    """Calculate Double Exponential Moving Average."""
    src = data[source]
    if use_talib and _TALIB:
        return pd.Series(talib.DEMA(src, timeperiod=length), index=data.index, name=f"DEMA_{length}")
    elif _PANDAS_TA:
        return ta.dema(src, length=length)
    else:
        ema1 = src.ewm(span=length, adjust=False).mean()
        ema2 = ema1.ewm(span=length, adjust=False).mean()
        return 2 * ema1 - ema2

IndicatorRegistry.register(
    IndicatorMetadata(
        name="dema",
        full_name="Double Exponential Moving Average",
        category=IndicatorCategory.OVERLAP,
        description="Reduces lag of standard EMA",
        params=[
            IndicatorParam("length", "int", 20, 1, 500, "Period length"),
        ],
        outputs=[IndicatorOutput("dema", "DEMA values")],
        talib_func="DEMA",
        pandas_ta_func="dema",
        required_columns=["close"],
    ),
    _calc_dema
)


# =============================================================================
# Triple Exponential Moving Average (TEMA)
# =============================================================================
def _calc_tema(data: pd.DataFrame, use_talib: bool = True, source: str = "close",
               length: int = 20, **kwargs) -> pd.Series:
    """Calculate Triple Exponential Moving Average."""
    src = data[source]
    if use_talib and _TALIB:
        return pd.Series(talib.TEMA(src, timeperiod=length), index=data.index, name=f"TEMA_{length}")
    elif _PANDAS_TA:
        return ta.tema(src, length=length)
    else:
        ema1 = src.ewm(span=length, adjust=False).mean()
        ema2 = ema1.ewm(span=length, adjust=False).mean()
        ema3 = ema2.ewm(span=length, adjust=False).mean()
        return 3 * ema1 - 3 * ema2 + ema3

IndicatorRegistry.register(
    IndicatorMetadata(
        name="tema",
        full_name="Triple Exponential Moving Average",
        category=IndicatorCategory.OVERLAP,
        description="Further reduces lag compared to DEMA",
        params=[
            IndicatorParam("length", "int", 20, 1, 500, "Period length"),
        ],
        outputs=[IndicatorOutput("tema", "TEMA values")],
        talib_func="TEMA",
        pandas_ta_func="tema",
        required_columns=["close"],
    ),
    _calc_tema
)


# =============================================================================
# Triangular Moving Average (TRIMA)
# =============================================================================
def _calc_trima(data: pd.DataFrame, use_talib: bool = True, source: str = "close",
                length: int = 20, **kwargs) -> pd.Series:
    """Calculate Triangular Moving Average."""
    src = data[source]
    if use_talib and _TALIB:
        return pd.Series(talib.TRIMA(src, timeperiod=length), index=data.index, name=f"TRIMA_{length}")
    elif _PANDAS_TA:
        return ta.trima(src, length=length)
    else:
        sma1 = src.rolling(window=length).mean()
        return sma1.rolling(window=length).mean()

IndicatorRegistry.register(
    IndicatorMetadata(
        name="trima",
        full_name="Triangular Moving Average",
        category=IndicatorCategory.OVERLAP,
        description="Double-smoothed SMA",
        params=[
            IndicatorParam("length", "int", 20, 1, 500, "Period length"),
        ],
        outputs=[IndicatorOutput("trima", "TRIMA values")],
        talib_func="TRIMA",
        pandas_ta_func="trima",
        required_columns=["close"],
    ),
    _calc_trima
)


# =============================================================================
# Weighted Moving Average (WMA)
# =============================================================================
def _calc_wma(data: pd.DataFrame, use_talib: bool = True, source: str = "close",
              length: int = 20, **kwargs) -> pd.Series:
    """Calculate Weighted Moving Average."""
    src = data[source]
    if use_talib and _TALIB:
        return pd.Series(talib.WMA(src, timeperiod=length), index=data.index, name=f"WMA_{length}")
    elif _PANDAS_TA:
        return ta.wma(src, length=length)
    else:
        weights = np.arange(1, length + 1)
        return src.rolling(window=length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

IndicatorRegistry.register(
    IndicatorMetadata(
        name="wma",
        full_name="Weighted Moving Average",
        category=IndicatorCategory.OVERLAP,
        description="Linearly weighted moving average",
        params=[
            IndicatorParam("length", "int", 20, 1, 500, "Period length"),
        ],
        outputs=[IndicatorOutput("wma", "WMA values")],
        talib_func="WMA",
        pandas_ta_func="wma",
        required_columns=["close"],
    ),
    _calc_wma
)


# =============================================================================
# Kaufman Adaptive Moving Average (KAMA)
# =============================================================================
def _calc_kama(data: pd.DataFrame, use_talib: bool = True, source: str = "close",
               length: int = 10, fast: int = 2, slow: int = 30, **kwargs) -> pd.Series:
    """Calculate Kaufman Adaptive Moving Average."""
    src = data[source]
    if use_talib and _TALIB:
        return pd.Series(talib.KAMA(src, timeperiod=length), index=data.index, name=f"KAMA_{length}")
    elif _PANDAS_TA:
        return ta.kama(src, length=length, fast=fast, slow=slow)
    else:
        # Pure pandas implementation
        change = abs(src - src.shift(length))
        volatility = abs(src.diff()).rolling(window=length).sum()
        er = change / volatility.replace(0, np.nan)
        
        fast_sc = 2 / (fast + 1)
        slow_sc = 2 / (slow + 1)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        
        kama = pd.Series(index=src.index, dtype=float)
        kama.iloc[length - 1] = src.iloc[length - 1]
        
        for i in range(length, len(src)):
            kama.iloc[i] = kama.iloc[i - 1] + sc.iloc[i] * (src.iloc[i] - kama.iloc[i - 1])
        
        return kama

IndicatorRegistry.register(
    IndicatorMetadata(
        name="kama",
        full_name="Kaufman Adaptive Moving Average",
        category=IndicatorCategory.OVERLAP,
        description="Adjusts smoothing based on market efficiency",
        params=[
            IndicatorParam("length", "int", 10, 1, 500, "Efficiency ratio period"),
            IndicatorParam("fast", "int", 2, 1, 50, "Fast smoothing constant"),
            IndicatorParam("slow", "int", 30, 5, 500, "Slow smoothing constant"),
        ],
        outputs=[IndicatorOutput("kama", "KAMA values")],
        talib_func="KAMA",
        pandas_ta_func="kama",
        required_columns=["close"],
    ),
    _calc_kama
)


# =============================================================================
# MESA Adaptive Moving Average (MAMA)
# =============================================================================
def _calc_mama(data: pd.DataFrame, use_talib: bool = True, source: str = "close",
               fastlimit: float = 0.5, slowlimit: float = 0.05, **kwargs) -> pd.DataFrame:
    """Calculate MESA Adaptive Moving Average."""
    src = data[source]
    if use_talib and _TALIB:
        mama, fama = talib.MAMA(src, fastlimit=fastlimit, slowlimit=slowlimit)
        return pd.DataFrame({
            "MAMA": mama,
            "FAMA": fama
        }, index=data.index)
    elif _PANDAS_TA:
        result = ta.mama(src, fastlimit=fastlimit, slowlimit=slowlimit)
        return result
    else:
        raise NotImplementedError("MAMA requires TA-Lib or pandas-ta")

IndicatorRegistry.register(
    IndicatorMetadata(
        name="mama",
        full_name="MESA Adaptive Moving Average",
        category=IndicatorCategory.OVERLAP,
        description="Hilbert Transform-based adaptive MA",
        params=[
            IndicatorParam("fastlimit", "float", 0.5, 0.01, 0.99, "Fast limit"),
            IndicatorParam("slowlimit", "float", 0.05, 0.01, 0.99, "Slow limit"),
        ],
        outputs=[
            IndicatorOutput("MAMA", "MAMA line"),
            IndicatorOutput("FAMA", "Following Adaptive MA"),
        ],
        talib_func="MAMA",
        pandas_ta_func="mama",
        required_columns=["close"],
    ),
    _calc_mama
)


# =============================================================================
# T3 - Triple Exponential Moving Average
# =============================================================================
def _calc_t3(data: pd.DataFrame, use_talib: bool = True, source: str = "close",
             length: int = 5, vfactor: float = 0.7, **kwargs) -> pd.Series:
    """Calculate T3 Moving Average."""
    src = data[source]
    if use_talib and _TALIB:
        return pd.Series(talib.T3(src, timeperiod=length, vfactor=vfactor), index=data.index, name=f"T3_{length}")
    elif _PANDAS_TA:
        return ta.t3(src, length=length, a=vfactor)
    else:
        raise NotImplementedError("T3 requires TA-Lib or pandas-ta")

IndicatorRegistry.register(
    IndicatorMetadata(
        name="t3",
        full_name="T3 - Triple Exponential Moving Average",
        category=IndicatorCategory.OVERLAP,
        description="Tillson T3 moving average",
        params=[
            IndicatorParam("length", "int", 5, 1, 500, "Period length"),
            IndicatorParam("vfactor", "float", 0.7, 0, 1, "Volume factor"),
        ],
        outputs=[IndicatorOutput("t3", "T3 values")],
        talib_func="T3",
        pandas_ta_func="t3",
        required_columns=["close"],
    ),
    _calc_t3
)


# =============================================================================
# Bollinger Bands (BBANDS)
# =============================================================================
def _calc_bbands(data: pd.DataFrame, use_talib: bool = True, source: str = "close",
                 length: int = 20, std: float = 2.0, **kwargs) -> pd.DataFrame:
    """Calculate Bollinger Bands."""
    src = data[source]
    if use_talib and _TALIB:
        upper, middle, lower = talib.BBANDS(src, timeperiod=length, nbdevup=std, nbdevdn=std)
        return pd.DataFrame({
            "BBU": upper,
            "BBM": middle,
            "BBL": lower
        }, index=data.index)
    elif _PANDAS_TA:
        result = ta.bbands(src, length=length, std=std)
        # pandas-ta returns: BBL, BBM, BBU, BBB, BBP
        return result[["BBL_" + str(length) + "_" + str(std), 
                       "BBM_" + str(length) + "_" + str(std),
                       "BBU_" + str(length) + "_" + str(std)]].rename(
            columns=lambda x: x.split("_")[0]
        )
    else:
        middle = src.rolling(window=length).mean()
        std_dev = src.rolling(window=length).std()
        return pd.DataFrame({
            "BBU": middle + std * std_dev,
            "BBM": middle,
            "BBL": middle - std * std_dev
        }, index=data.index)

IndicatorRegistry.register(
    IndicatorMetadata(
        name="bbands",
        full_name="Bollinger Bands",
        category=IndicatorCategory.OVERLAP,
        description="Volatility bands around a moving average",
        params=[
            IndicatorParam("length", "int", 20, 1, 500, "Period length"),
            IndicatorParam("std", "float", 2.0, 0.1, 5.0, "Standard deviation multiplier"),
        ],
        outputs=[
            IndicatorOutput("BBU", "Upper band"),
            IndicatorOutput("BBM", "Middle band"),
            IndicatorOutput("BBL", "Lower band"),
        ],
        talib_func="BBANDS",
        pandas_ta_func="bbands",
        required_columns=["close"],
    ),
    _calc_bbands
)


# =============================================================================
# Parabolic SAR
# =============================================================================
def _calc_sar(data: pd.DataFrame, use_talib: bool = True, 
              af: float = 0.02, max_af: float = 0.2, **kwargs) -> pd.Series:
    """Calculate Parabolic SAR."""
    if use_talib and _TALIB:
        return pd.Series(
            talib.SAR(data["high"], data["low"], acceleration=af, maximum=max_af),
            index=data.index, name="SAR"
        )
    elif _PANDAS_TA:
        result = ta.psar(data["high"], data["low"], data["close"], af0=af, af=af, max_af=max_af)
        # pandas-ta returns multiple columns, get the main SAR value
        if isinstance(result, pd.DataFrame):
            # Get PSARl or PSARs (long/short)
            for col in result.columns:
                if "PSARl" in col or "PSARs" in col:
                    return result[col].combine_first(result.get("PSARs_" + col.split("_")[1], pd.Series()))
        return result
    else:
        raise NotImplementedError("SAR requires TA-Lib or pandas-ta")

IndicatorRegistry.register(
    IndicatorMetadata(
        name="sar",
        full_name="Parabolic SAR",
        category=IndicatorCategory.OVERLAP,
        description="Stop and reverse trailing stop",
        params=[
            IndicatorParam("af", "float", 0.02, 0.001, 0.5, "Acceleration factor"),
            IndicatorParam("max_af", "float", 0.2, 0.01, 1.0, "Maximum acceleration factor"),
        ],
        outputs=[IndicatorOutput("sar", "SAR values")],
        talib_func="SAR",
        pandas_ta_func="psar",
        required_columns=["high", "low"],
    ),
    _calc_sar
)


# =============================================================================
# Midpoint over Period
# =============================================================================
def _calc_midpoint(data: pd.DataFrame, use_talib: bool = True, source: str = "close",
                   length: int = 14, **kwargs) -> pd.Series:
    """Calculate Midpoint over Period."""
    src = data[source]
    if use_talib and _TALIB:
        return pd.Series(talib.MIDPOINT(src, timeperiod=length), index=data.index, name=f"MIDPOINT_{length}")
    elif _PANDAS_TA:
        return ta.midpoint(src, length=length)
    else:
        return (src.rolling(window=length).max() + src.rolling(window=length).min()) / 2

IndicatorRegistry.register(
    IndicatorMetadata(
        name="midpoint",
        full_name="Midpoint over Period",
        category=IndicatorCategory.OVERLAP,
        description="(Highest + Lowest) / 2 over period",
        params=[
            IndicatorParam("length", "int", 14, 1, 500, "Period length"),
        ],
        outputs=[IndicatorOutput("midpoint", "Midpoint values")],
        talib_func="MIDPOINT",
        pandas_ta_func="midpoint",
        required_columns=["close"],
    ),
    _calc_midpoint
)


# =============================================================================
# Midpoint Price over Period
# =============================================================================
def _calc_midprice(data: pd.DataFrame, use_talib: bool = True,
                   length: int = 14, **kwargs) -> pd.Series:
    """Calculate Midpoint Price over Period."""
    if use_talib and _TALIB:
        return pd.Series(
            talib.MIDPRICE(data["high"], data["low"], timeperiod=length),
            index=data.index, name=f"MIDPRICE_{length}"
        )
    elif _PANDAS_TA:
        return ta.midprice(data["high"], data["low"], length=length)
    else:
        return (data["high"].rolling(window=length).max() + data["low"].rolling(window=length).min()) / 2

IndicatorRegistry.register(
    IndicatorMetadata(
        name="midprice",
        full_name="Midpoint Price over Period",
        category=IndicatorCategory.OVERLAP,
        description="(Highest High + Lowest Low) / 2 over period",
        params=[
            IndicatorParam("length", "int", 14, 1, 500, "Period length"),
        ],
        outputs=[IndicatorOutput("midprice", "Midprice values")],
        talib_func="MIDPRICE",
        pandas_ta_func="midprice",
        required_columns=["high", "low"],
    ),
    _calc_midprice
)


# =============================================================================
# Hilbert Transform - Instantaneous Trendline
# =============================================================================
def _calc_ht_trendline(data: pd.DataFrame, use_talib: bool = True, source: str = "close",
                       **kwargs) -> pd.Series:
    """Calculate Hilbert Transform Trendline."""
    src = data[source]
    if use_talib and _TALIB:
        return pd.Series(talib.HT_TRENDLINE(src), index=data.index, name="HT_TRENDLINE")
    elif _PANDAS_TA:
        return ta.ht_trendline(src)
    else:
        raise NotImplementedError("HT_TRENDLINE requires TA-Lib or pandas-ta")

IndicatorRegistry.register(
    IndicatorMetadata(
        name="ht_trendline",
        full_name="Hilbert Transform - Instantaneous Trendline",
        category=IndicatorCategory.OVERLAP,
        description="Dominant cycle trendline",
        params=[],
        outputs=[IndicatorOutput("ht_trendline", "Trendline values")],
        talib_func="HT_TRENDLINE",
        pandas_ta_func="ht_trendline",
        required_columns=["close"],
    ),
    _calc_ht_trendline
)


# =============================================================================
# Linear Regression
# =============================================================================
def _calc_linearreg(data: pd.DataFrame, use_talib: bool = True, source: str = "close",
                    length: int = 14, **kwargs) -> pd.Series:
    """Calculate Linear Regression."""
    src = data[source]
    if use_talib and _TALIB:
        return pd.Series(talib.LINEARREG(src, timeperiod=length), index=data.index, name=f"LINEARREG_{length}")
    elif _PANDAS_TA:
        return ta.linreg(src, length=length)
    else:
        raise NotImplementedError("LINEARREG requires TA-Lib or pandas-ta")

IndicatorRegistry.register(
    IndicatorMetadata(
        name="linearreg",
        full_name="Linear Regression",
        category=IndicatorCategory.OVERLAP,
        description="Linear regression over period",
        params=[
            IndicatorParam("length", "int", 14, 1, 500, "Period length"),
        ],
        outputs=[IndicatorOutput("linearreg", "Linear regression values")],
        talib_func="LINEARREG",
        pandas_ta_func="linreg",
        required_columns=["close"],
    ),
    _calc_linearreg
)


# =============================================================================
# Hull Moving Average (HMA) - pandas-ta only
# =============================================================================
def _calc_hma(data: pd.DataFrame, use_talib: bool = True, source: str = "close",
              length: int = 10, **kwargs) -> pd.Series:
    """Calculate Hull Moving Average."""
    src = data[source]
    if _PANDAS_TA:
        return ta.hma(src, length=length)
    else:
        # Pure pandas implementation
        half_length = int(length / 2)
        sqrt_length = int(np.sqrt(length))
        wma_half = src.rolling(window=half_length).apply(
            lambda x: np.dot(x, np.arange(1, half_length + 1)) / np.arange(1, half_length + 1).sum(), raw=True
        )
        wma_full = src.rolling(window=length).apply(
            lambda x: np.dot(x, np.arange(1, length + 1)) / np.arange(1, length + 1).sum(), raw=True
        )
        raw_hma = 2 * wma_half - wma_full
        return raw_hma.rolling(window=sqrt_length).apply(
            lambda x: np.dot(x, np.arange(1, sqrt_length + 1)) / np.arange(1, sqrt_length + 1).sum(), raw=True
        )

IndicatorRegistry.register(
    IndicatorMetadata(
        name="hma",
        full_name="Hull Moving Average",
        category=IndicatorCategory.OVERLAP,
        description="Responsive moving average with reduced lag",
        params=[
            IndicatorParam("length", "int", 10, 1, 500, "Period length"),
        ],
        outputs=[IndicatorOutput("hma", "HMA values")],
        talib_func=None,
        pandas_ta_func="hma",
        required_columns=["close"],
    ),
    _calc_hma
)


# =============================================================================
# SuperTrend - pandas-ta only
# =============================================================================
def _calc_supertrend(data: pd.DataFrame, use_talib: bool = True,
                     length: int = 7, multiplier: float = 3.0, **kwargs) -> pd.DataFrame:
    """Calculate SuperTrend."""
    if _PANDAS_TA:
        result = ta.supertrend(data["high"], data["low"], data["close"], 
                               length=length, multiplier=multiplier)
        return result
    else:
        raise NotImplementedError("SuperTrend requires pandas-ta")

IndicatorRegistry.register(
    IndicatorMetadata(
        name="supertrend",
        full_name="SuperTrend",
        category=IndicatorCategory.OVERLAP,
        description="ATR-based trend following indicator",
        params=[
            IndicatorParam("length", "int", 7, 1, 100, "ATR period"),
            IndicatorParam("multiplier", "float", 3.0, 0.5, 10.0, "ATR multiplier"),
        ],
        outputs=[
            IndicatorOutput("SUPERT", "SuperTrend value"),
            IndicatorOutput("SUPERTd", "SuperTrend direction (1=up, -1=down)"),
            IndicatorOutput("SUPERTl", "SuperTrend long"),
            IndicatorOutput("SUPERTs", "SuperTrend short"),
        ],
        talib_func=None,
        pandas_ta_func="supertrend",
        required_columns=["high", "low", "close"],
    ),
    _calc_supertrend
)


# =============================================================================
# Ichimoku Kinkō Hyō - pandas-ta only
# =============================================================================
def _calc_ichimoku(data: pd.DataFrame, use_talib: bool = True,
                   tenkan: int = 9, kijun: int = 26, senkou: int = 52, **kwargs) -> pd.DataFrame:
    """Calculate Ichimoku Cloud."""
    if _PANDAS_TA:
        ichimoku, span = ta.ichimoku(data["high"], data["low"], data["close"],
                                      tenkan=tenkan, kijun=kijun, senkou=senkou)
        return ichimoku
    else:
        raise NotImplementedError("Ichimoku requires pandas-ta")

IndicatorRegistry.register(
    IndicatorMetadata(
        name="ichimoku",
        full_name="Ichimoku Kinkō Hyō",
        category=IndicatorCategory.OVERLAP,
        description="Comprehensive trend identification system",
        params=[
            IndicatorParam("tenkan", "int", 9, 1, 100, "Tenkan-sen period"),
            IndicatorParam("kijun", "int", 26, 1, 200, "Kijun-sen period"),
            IndicatorParam("senkou", "int", 52, 1, 500, "Senkou span B period"),
        ],
        outputs=[
            IndicatorOutput("ISA", "Senkou Span A"),
            IndicatorOutput("ISB", "Senkou Span B"),
            IndicatorOutput("ITS", "Tenkan-sen"),
            IndicatorOutput("IKS", "Kijun-sen"),
            IndicatorOutput("ICS", "Chikou Span"),
        ],
        talib_func=None,
        pandas_ta_func="ichimoku",
        required_columns=["high", "low", "close"],
    ),
    _calc_ichimoku
)


# =============================================================================
# VWAP - pandas-ta only
# =============================================================================
def _calc_vwap(data: pd.DataFrame, use_talib: bool = True, **kwargs) -> pd.Series:
    """Calculate Volume Weighted Average Price."""
    if _PANDAS_TA:
        return ta.vwap(data["high"], data["low"], data["close"], data["volume"])
    else:
        typical_price = (data["high"] + data["low"] + data["close"]) / 3
        return (typical_price * data["volume"]).cumsum() / data["volume"].cumsum()

IndicatorRegistry.register(
    IndicatorMetadata(
        name="vwap",
        full_name="Volume Weighted Average Price",
        category=IndicatorCategory.OVERLAP,
        description="Average price weighted by volume",
        params=[],
        outputs=[IndicatorOutput("vwap", "VWAP values")],
        talib_func=None,
        pandas_ta_func="vwap",
        required_columns=["high", "low", "close", "volume"],
    ),
    _calc_vwap
)


# =============================================================================
# Weighted Close Price
# =============================================================================
def _calc_wcp(data: pd.DataFrame, use_talib: bool = True, **kwargs) -> pd.Series:
    """Calculate Weighted Close Price."""
    if use_talib and _TALIB:
        return pd.Series(
            talib.WCLPRICE(data["high"], data["low"], data["close"]),
            index=data.index, name="WCP"
        )
    elif _PANDAS_TA:
        return ta.wcp(data["high"], data["low"], data["close"])
    else:
        return (data["high"] + data["low"] + 2 * data["close"]) / 4

IndicatorRegistry.register(
    IndicatorMetadata(
        name="wcp",
        full_name="Weighted Close Price",
        category=IndicatorCategory.OVERLAP,
        description="(High + Low + 2*Close) / 4",
        params=[],
        outputs=[IndicatorOutput("wcp", "Weighted close values")],
        talib_func="WCLPRICE",
        pandas_ta_func="wcp",
        required_columns=["high", "low", "close"],
    ),
    _calc_wcp
)


# =============================================================================
# HL2 - High-Low Midpoint
# =============================================================================
def _calc_hl2(data: pd.DataFrame, use_talib: bool = True, **kwargs) -> pd.Series:
    """Calculate HL2 (High-Low Midpoint)."""
    if _PANDAS_TA:
        return ta.hl2(data["high"], data["low"])
    else:
        return (data["high"] + data["low"]) / 2

IndicatorRegistry.register(
    IndicatorMetadata(
        name="hl2",
        full_name="HL2 (High-Low Midpoint)",
        category=IndicatorCategory.OVERLAP,
        description="(High + Low) / 2",
        params=[],
        outputs=[IndicatorOutput("hl2", "HL2 values")],
        talib_func=None,
        pandas_ta_func="hl2",
        required_columns=["high", "low"],
    ),
    _calc_hl2
)


# =============================================================================
# HLC3 - Typical Price
# =============================================================================
def _calc_hlc3(data: pd.DataFrame, use_talib: bool = True, **kwargs) -> pd.Series:
    """Calculate HLC3 (Typical Price)."""
    if use_talib and _TALIB:
        return pd.Series(
            talib.TYPPRICE(data["high"], data["low"], data["close"]),
            index=data.index, name="HLC3"
        )
    elif _PANDAS_TA:
        return ta.hlc3(data["high"], data["low"], data["close"])
    else:
        return (data["high"] + data["low"] + data["close"]) / 3

IndicatorRegistry.register(
    IndicatorMetadata(
        name="hlc3",
        full_name="HLC3 (Typical Price)",
        category=IndicatorCategory.OVERLAP,
        description="(High + Low + Close) / 3",
        params=[],
        outputs=[IndicatorOutput("hlc3", "HLC3 values")],
        talib_func="TYPPRICE",
        pandas_ta_func="hlc3",
        required_columns=["high", "low", "close"],
    ),
    _calc_hlc3
)


# =============================================================================
# OHLC4
# =============================================================================
def _calc_ohlc4(data: pd.DataFrame, use_talib: bool = True, **kwargs) -> pd.Series:
    """Calculate OHLC4."""
    if _PANDAS_TA:
        return ta.ohlc4(data["open"], data["high"], data["low"], data["close"])
    else:
        return (data["open"] + data["high"] + data["low"] + data["close"]) / 4

IndicatorRegistry.register(
    IndicatorMetadata(
        name="ohlc4",
        full_name="OHLC4",
        category=IndicatorCategory.OVERLAP,
        description="(Open + High + Low + Close) / 4",
        params=[],
        outputs=[IndicatorOutput("ohlc4", "OHLC4 values")],
        talib_func=None,
        pandas_ta_func="ohlc4",
        required_columns=["open", "high", "low", "close"],
    ),
    _calc_ohlc4
)


# =============================================================================
# Donchian Channels - pandas-ta only
# =============================================================================
def _calc_donchian(data: pd.DataFrame, use_talib: bool = True,
                   lower_length: int = 20, upper_length: int = 20, **kwargs) -> pd.DataFrame:
    """Calculate Donchian Channels."""
    if _PANDAS_TA:
        return ta.donchian(data["high"], data["low"], 
                           lower_length=lower_length, upper_length=upper_length)
    else:
        return pd.DataFrame({
            "DCL": data["low"].rolling(window=lower_length).min(),
            "DCM": (data["high"].rolling(window=upper_length).max() + 
                    data["low"].rolling(window=lower_length).min()) / 2,
            "DCU": data["high"].rolling(window=upper_length).max(),
        }, index=data.index)

IndicatorRegistry.register(
    IndicatorMetadata(
        name="donchian",
        full_name="Donchian Channels",
        category=IndicatorCategory.OVERLAP,
        description="Highest high and lowest low over period",
        params=[
            IndicatorParam("lower_length", "int", 20, 1, 500, "Lower channel period"),
            IndicatorParam("upper_length", "int", 20, 1, 500, "Upper channel period"),
        ],
        outputs=[
            IndicatorOutput("DCL", "Lower channel"),
            IndicatorOutput("DCM", "Middle channel"),
            IndicatorOutput("DCU", "Upper channel"),
        ],
        talib_func=None,
        pandas_ta_func="donchian",
        required_columns=["high", "low"],
    ),
    _calc_donchian
)


# =============================================================================
# Keltner Channels - pandas-ta only
# =============================================================================
def _calc_kc(data: pd.DataFrame, use_talib: bool = True,
             length: int = 20, scalar: float = 2.0, **kwargs) -> pd.DataFrame:
    """Calculate Keltner Channels."""
    if _PANDAS_TA:
        return ta.kc(data["high"], data["low"], data["close"], 
                     length=length, scalar=scalar)
    else:
        raise NotImplementedError("Keltner Channels requires pandas-ta")

IndicatorRegistry.register(
    IndicatorMetadata(
        name="kc",
        full_name="Keltner Channels",
        category=IndicatorCategory.OVERLAP,
        description="EMA-based volatility channels",
        params=[
            IndicatorParam("length", "int", 20, 1, 500, "Period length"),
            IndicatorParam("scalar", "float", 2.0, 0.5, 5.0, "ATR multiplier"),
        ],
        outputs=[
            IndicatorOutput("KCL", "Lower channel"),
            IndicatorOutput("KCB", "Middle band (EMA)"),
            IndicatorOutput("KCU", "Upper channel"),
        ],
        talib_func=None,
        pandas_ta_func="kc",
        required_columns=["high", "low", "close"],
    ),
    _calc_kc
)
