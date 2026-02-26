"""Momentum & Trend Indicators.

This module contains momentum oscillators and trend indicators.
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
# Relative Strength Index (RSI)
# =============================================================================
def _calc_rsi(data: pd.DataFrame, use_talib: bool = True, source: str = "close",
              length: int = 14, **kwargs) -> pd.Series:
    """Calculate Relative Strength Index."""
    src = data[source]
    if use_talib and _TALIB:
        return pd.Series(talib.RSI(src, timeperiod=length), index=data.index, name=f"RSI_{length}")
    elif _PANDAS_TA:
        return ta.rsi(src, length=length)
    else:
        delta = src.diff()
        gain = delta.where(delta > 0, 0).rolling(window=length).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

IndicatorRegistry.register(
    IndicatorMetadata(
        name="rsi",
        full_name="Relative Strength Index",
        category=IndicatorCategory.MOMENTUM,
        description="Momentum oscillator measuring speed and change of price movements",
        params=[
            IndicatorParam("length", "int", 14, 2, 100, "Period length"),
        ],
        outputs=[IndicatorOutput("rsi", "RSI values (0-100)")],
        talib_func="RSI",
        pandas_ta_func="rsi",
        required_columns=["close"],
    ),
    _calc_rsi
)


# =============================================================================
# MACD - Moving Average Convergence Divergence
# =============================================================================
def _calc_macd(data: pd.DataFrame, use_talib: bool = True, source: str = "close",
               fast: int = 12, slow: int = 26, signal: int = 9, **kwargs) -> pd.DataFrame:
    """Calculate MACD."""
    src = data[source]
    if use_talib and _TALIB:
        macd, macd_signal, macd_hist = talib.MACD(src, fastperiod=fast, slowperiod=slow, signalperiod=signal)
        return pd.DataFrame({
            "MACD": macd,
            "MACDs": macd_signal,
            "MACDh": macd_hist
        }, index=data.index)
    elif _PANDAS_TA:
        result = ta.macd(src, fast=fast, slow=slow, signal=signal)
        return result
    else:
        ema_fast = src.ewm(span=fast, adjust=False).mean()
        ema_slow = src.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        macd_hist = macd - macd_signal
        return pd.DataFrame({
            "MACD": macd,
            "MACDs": macd_signal,
            "MACDh": macd_hist
        }, index=data.index)

IndicatorRegistry.register(
    IndicatorMetadata(
        name="macd",
        full_name="Moving Average Convergence Divergence",
        category=IndicatorCategory.MOMENTUM,
        description="Trend-following momentum indicator",
        params=[
            IndicatorParam("fast", "int", 12, 1, 100, "Fast EMA period"),
            IndicatorParam("slow", "int", 26, 1, 200, "Slow EMA period"),
            IndicatorParam("signal", "int", 9, 1, 50, "Signal line period"),
        ],
        outputs=[
            IndicatorOutput("MACD", "MACD line"),
            IndicatorOutput("MACDs", "Signal line"),
            IndicatorOutput("MACDh", "Histogram"),
        ],
        talib_func="MACD",
        pandas_ta_func="macd",
        required_columns=["close"],
    ),
    _calc_macd
)


# =============================================================================
# Average Directional Index (ADX)
# =============================================================================
def _calc_adx(data: pd.DataFrame, use_talib: bool = True,
              length: int = 14, **kwargs) -> Union[pd.Series, pd.DataFrame]:
    """Calculate Average Directional Index."""
    if use_talib and _TALIB:
        adx = talib.ADX(data["high"], data["low"], data["close"], timeperiod=length)
        plus_di = talib.PLUS_DI(data["high"], data["low"], data["close"], timeperiod=length)
        minus_di = talib.MINUS_DI(data["high"], data["low"], data["close"], timeperiod=length)
        return pd.DataFrame({
            "ADX": adx,
            "DMP": plus_di,
            "DMN": minus_di
        }, index=data.index)
    elif _PANDAS_TA:
        return ta.adx(data["high"], data["low"], data["close"], length=length)
    else:
        raise NotImplementedError("ADX requires TA-Lib or pandas-ta")

IndicatorRegistry.register(
    IndicatorMetadata(
        name="adx",
        full_name="Average Directional Index",
        category=IndicatorCategory.MOMENTUM,
        description="Measures trend strength regardless of direction",
        params=[
            IndicatorParam("length", "int", 14, 1, 100, "Period length"),
        ],
        outputs=[
            IndicatorOutput("ADX", "ADX value (trend strength)"),
            IndicatorOutput("DMP", "Plus Directional Indicator (+DI)"),
            IndicatorOutput("DMN", "Minus Directional Indicator (-DI)"),
        ],
        talib_func="ADX",
        pandas_ta_func="adx",
        required_columns=["high", "low", "close"],
    ),
    _calc_adx
)


# =============================================================================
# Stochastic Oscillator
# =============================================================================
def _calc_stoch(data: pd.DataFrame, use_talib: bool = True,
                fastk_period: int = 14, slowk_period: int = 3, slowd_period: int = 3,
                **kwargs) -> pd.DataFrame:
    """Calculate Stochastic Oscillator."""
    if use_talib and _TALIB:
        slowk, slowd = talib.STOCH(
            data["high"], data["low"], data["close"],
            fastk_period=fastk_period, slowk_period=slowk_period, slowd_period=slowd_period
        )
        return pd.DataFrame({
            "STOCHk": slowk,
            "STOCHd": slowd
        }, index=data.index)
    elif _PANDAS_TA:
        return ta.stoch(data["high"], data["low"], data["close"],
                        k=fastk_period, d=slowd_period, smooth_k=slowk_period)
    else:
        lowest_low = data["low"].rolling(window=fastk_period).min()
        highest_high = data["high"].rolling(window=fastk_period).max()
        fastk = 100 * (data["close"] - lowest_low) / (highest_high - lowest_low)
        slowk = fastk.rolling(window=slowk_period).mean()
        slowd = slowk.rolling(window=slowd_period).mean()
        return pd.DataFrame({
            "STOCHk": slowk,
            "STOCHd": slowd
        }, index=data.index)

IndicatorRegistry.register(
    IndicatorMetadata(
        name="stoch",
        full_name="Stochastic Oscillator",
        category=IndicatorCategory.MOMENTUM,
        description="Momentum indicator comparing closing price to price range",
        params=[
            IndicatorParam("fastk_period", "int", 14, 1, 100, "Fast %K period"),
            IndicatorParam("slowk_period", "int", 3, 1, 50, "Slow %K period"),
            IndicatorParam("slowd_period", "int", 3, 1, 50, "Slow %D period"),
        ],
        outputs=[
            IndicatorOutput("STOCHk", "Slow %K"),
            IndicatorOutput("STOCHd", "Slow %D"),
        ],
        talib_func="STOCH",
        pandas_ta_func="stoch",
        required_columns=["high", "low", "close"],
    ),
    _calc_stoch
)


# =============================================================================
# Stochastic Fast
# =============================================================================
def _calc_stochf(data: pd.DataFrame, use_talib: bool = True,
                 fastk_period: int = 5, fastd_period: int = 3, **kwargs) -> pd.DataFrame:
    """Calculate Fast Stochastic."""
    if use_talib and _TALIB:
        fastk, fastd = talib.STOCHF(
            data["high"], data["low"], data["close"],
            fastk_period=fastk_period, fastd_period=fastd_period
        )
        return pd.DataFrame({
            "STOCHFk": fastk,
            "STOCHFd": fastd
        }, index=data.index)
    elif _PANDAS_TA:
        result = ta.stoch(data["high"], data["low"], data["close"],
                          k=fastk_period, d=fastd_period, smooth_k=1)
        return result
    else:
        lowest_low = data["low"].rolling(window=fastk_period).min()
        highest_high = data["high"].rolling(window=fastk_period).max()
        fastk = 100 * (data["close"] - lowest_low) / (highest_high - lowest_low)
        fastd = fastk.rolling(window=fastd_period).mean()
        return pd.DataFrame({
            "STOCHFk": fastk,
            "STOCHFd": fastd
        }, index=data.index)

IndicatorRegistry.register(
    IndicatorMetadata(
        name="stochf",
        full_name="Stochastic Fast",
        category=IndicatorCategory.MOMENTUM,
        description="Fast version of stochastic oscillator",
        params=[
            IndicatorParam("fastk_period", "int", 5, 1, 100, "Fast %K period"),
            IndicatorParam("fastd_period", "int", 3, 1, 50, "Fast %D period"),
        ],
        outputs=[
            IndicatorOutput("STOCHFk", "Fast %K"),
            IndicatorOutput("STOCHFd", "Fast %D"),
        ],
        talib_func="STOCHF",
        pandas_ta_func="stoch",
        required_columns=["high", "low", "close"],
    ),
    _calc_stochf
)


# =============================================================================
# Stochastic RSI
# =============================================================================
def _calc_stochrsi(data: pd.DataFrame, use_talib: bool = True, source: str = "close",
                   length: int = 14, rsi_length: int = 14, k: int = 3, d: int = 3,
                   **kwargs) -> pd.DataFrame:
    """Calculate Stochastic RSI."""
    if use_talib and _TALIB:
        fastk, fastd = talib.STOCHRSI(data[source], timeperiod=length,
                                       fastk_period=k, fastd_period=d)
        return pd.DataFrame({
            "STOCHRSIk": fastk,
            "STOCHRSId": fastd
        }, index=data.index)
    elif _PANDAS_TA:
        return ta.stochrsi(data[source], length=length, rsi_length=rsi_length, k=k, d=d)
    else:
        raise NotImplementedError("StochRSI requires TA-Lib or pandas-ta")

IndicatorRegistry.register(
    IndicatorMetadata(
        name="stochrsi",
        full_name="Stochastic RSI",
        category=IndicatorCategory.MOMENTUM,
        description="RSI processed through stochastic formula",
        params=[
            IndicatorParam("length", "int", 14, 2, 100, "Stochastic period"),
            IndicatorParam("rsi_length", "int", 14, 2, 100, "RSI period"),
            IndicatorParam("k", "int", 3, 1, 50, "%K smoothing"),
            IndicatorParam("d", "int", 3, 1, 50, "%D smoothing"),
        ],
        outputs=[
            IndicatorOutput("STOCHRSIk", "Stochastic RSI %K"),
            IndicatorOutput("STOCHRSId", "Stochastic RSI %D"),
        ],
        talib_func="STOCHRSI",
        pandas_ta_func="stochrsi",
        required_columns=["close"],
    ),
    _calc_stochrsi
)


# =============================================================================
# Commodity Channel Index (CCI)
# =============================================================================
def _calc_cci(data: pd.DataFrame, use_talib: bool = True,
              length: int = 20, **kwargs) -> pd.Series:
    """Calculate Commodity Channel Index."""
    if use_talib and _TALIB:
        return pd.Series(
            talib.CCI(data["high"], data["low"], data["close"], timeperiod=length),
            index=data.index, name=f"CCI_{length}"
        )
    elif _PANDAS_TA:
        return ta.cci(data["high"], data["low"], data["close"], length=length)
    else:
        tp = (data["high"] + data["low"] + data["close"]) / 3
        sma = tp.rolling(window=length).mean()
        mad = tp.rolling(window=length).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        return (tp - sma) / (0.015 * mad)

IndicatorRegistry.register(
    IndicatorMetadata(
        name="cci",
        full_name="Commodity Channel Index",
        category=IndicatorCategory.MOMENTUM,
        description="Identifies cyclical trends",
        params=[
            IndicatorParam("length", "int", 20, 1, 100, "Period length"),
        ],
        outputs=[IndicatorOutput("cci", "CCI values")],
        talib_func="CCI",
        pandas_ta_func="cci",
        required_columns=["high", "low", "close"],
    ),
    _calc_cci
)


# =============================================================================
# Williams %R
# =============================================================================
def _calc_willr(data: pd.DataFrame, use_talib: bool = True,
                length: int = 14, **kwargs) -> pd.Series:
    """Calculate Williams %R."""
    if use_talib and _TALIB:
        return pd.Series(
            talib.WILLR(data["high"], data["low"], data["close"], timeperiod=length),
            index=data.index, name=f"WILLR_{length}"
        )
    elif _PANDAS_TA:
        return ta.willr(data["high"], data["low"], data["close"], length=length)
    else:
        highest_high = data["high"].rolling(window=length).max()
        lowest_low = data["low"].rolling(window=length).min()
        return -100 * (highest_high - data["close"]) / (highest_high - lowest_low)

IndicatorRegistry.register(
    IndicatorMetadata(
        name="willr",
        full_name="Williams %R",
        category=IndicatorCategory.MOMENTUM,
        description="Momentum indicator similar to stochastic",
        params=[
            IndicatorParam("length", "int", 14, 1, 100, "Period length"),
        ],
        outputs=[IndicatorOutput("willr", "Williams %R values (-100 to 0)")],
        talib_func="WILLR",
        pandas_ta_func="willr",
        required_columns=["high", "low", "close"],
    ),
    _calc_willr
)


# =============================================================================
# Momentum (MOM)
# =============================================================================
def _calc_mom(data: pd.DataFrame, use_talib: bool = True, source: str = "close",
              length: int = 10, **kwargs) -> pd.Series:
    """Calculate Momentum."""
    src = data[source]
    if use_talib and _TALIB:
        return pd.Series(talib.MOM(src, timeperiod=length), index=data.index, name=f"MOM_{length}")
    elif _PANDAS_TA:
        return ta.mom(src, length=length)
    else:
        return src - src.shift(length)

IndicatorRegistry.register(
    IndicatorMetadata(
        name="mom",
        full_name="Momentum",
        category=IndicatorCategory.MOMENTUM,
        description="Price change over period",
        params=[
            IndicatorParam("length", "int", 10, 1, 100, "Period length"),
        ],
        outputs=[IndicatorOutput("mom", "Momentum values")],
        talib_func="MOM",
        pandas_ta_func="mom",
        required_columns=["close"],
    ),
    _calc_mom
)


# =============================================================================
# Rate of Change (ROC)
# =============================================================================
def _calc_roc(data: pd.DataFrame, use_talib: bool = True, source: str = "close",
              length: int = 10, **kwargs) -> pd.Series:
    """Calculate Rate of Change."""
    src = data[source]
    if use_talib and _TALIB:
        return pd.Series(talib.ROC(src, timeperiod=length), index=data.index, name=f"ROC_{length}")
    elif _PANDAS_TA:
        return ta.roc(src, length=length)
    else:
        return 100 * (src - src.shift(length)) / src.shift(length)

IndicatorRegistry.register(
    IndicatorMetadata(
        name="roc",
        full_name="Rate of Change",
        category=IndicatorCategory.MOMENTUM,
        description="Percentage price change over period",
        params=[
            IndicatorParam("length", "int", 10, 1, 100, "Period length"),
        ],
        outputs=[IndicatorOutput("roc", "ROC values (%)")],
        talib_func="ROC",
        pandas_ta_func="roc",
        required_columns=["close"],
    ),
    _calc_roc
)


# =============================================================================
# Aroon
# =============================================================================
def _calc_aroon(data: pd.DataFrame, use_talib: bool = True,
                length: int = 14, **kwargs) -> pd.DataFrame:
    """Calculate Aroon Indicator."""
    if use_talib and _TALIB:
        aroondown, aroonup = talib.AROON(data["high"], data["low"], timeperiod=length)
        return pd.DataFrame({
            "AROONU": aroonup,
            "AROOND": aroondown,
            "AROONOSC": aroonup - aroondown
        }, index=data.index)
    elif _PANDAS_TA:
        return ta.aroon(data["high"], data["low"], length=length)
    else:
        aroonup = 100 * data["high"].rolling(window=length + 1).apply(
            lambda x: length - x.argmax(), raw=True
        ) / length
        aroondown = 100 * data["low"].rolling(window=length + 1).apply(
            lambda x: length - x.argmin(), raw=True
        ) / length
        return pd.DataFrame({
            "AROONU": aroonup,
            "AROOND": aroondown,
            "AROONOSC": aroonup - aroondown
        }, index=data.index)

IndicatorRegistry.register(
    IndicatorMetadata(
        name="aroon",
        full_name="Aroon",
        category=IndicatorCategory.MOMENTUM,
        description="Identifies trend changes and strength",
        params=[
            IndicatorParam("length", "int", 14, 1, 100, "Period length"),
        ],
        outputs=[
            IndicatorOutput("AROONU", "Aroon Up"),
            IndicatorOutput("AROOND", "Aroon Down"),
            IndicatorOutput("AROONOSC", "Aroon Oscillator"),
        ],
        talib_func="AROON",
        pandas_ta_func="aroon",
        required_columns=["high", "low"],
    ),
    _calc_aroon
)


# =============================================================================
# Balance of Power (BOP)
# =============================================================================
def _calc_bop(data: pd.DataFrame, use_talib: bool = True, **kwargs) -> pd.Series:
    """Calculate Balance of Power."""
    if use_talib and _TALIB:
        return pd.Series(
            talib.BOP(data["open"], data["high"], data["low"], data["close"]),
            index=data.index, name="BOP"
        )
    elif _PANDAS_TA:
        return ta.bop(data["open"], data["high"], data["low"], data["close"])
    else:
        return (data["close"] - data["open"]) / (data["high"] - data["low"])

IndicatorRegistry.register(
    IndicatorMetadata(
        name="bop",
        full_name="Balance of Power",
        category=IndicatorCategory.MOMENTUM,
        description="Measures buying/selling pressure",
        params=[],
        outputs=[IndicatorOutput("bop", "BOP values (-1 to 1)")],
        talib_func="BOP",
        pandas_ta_func="bop",
        required_columns=["open", "high", "low", "close"],
    ),
    _calc_bop
)


# =============================================================================
# Percentage Price Oscillator (PPO)
# =============================================================================
def _calc_ppo(data: pd.DataFrame, use_talib: bool = True, source: str = "close",
              fast: int = 12, slow: int = 26, **kwargs) -> pd.Series:
    """Calculate Percentage Price Oscillator."""
    src = data[source]
    if use_talib and _TALIB:
        return pd.Series(
            talib.PPO(src, fastperiod=fast, slowperiod=slow),
            index=data.index, name=f"PPO_{fast}_{slow}"
        )
    elif _PANDAS_TA:
        result = ta.ppo(src, fast=fast, slow=slow)
        if isinstance(result, pd.DataFrame):
            return result.iloc[:, 0]
        return result
    else:
        ema_fast = src.ewm(span=fast, adjust=False).mean()
        ema_slow = src.ewm(span=slow, adjust=False).mean()
        return 100 * (ema_fast - ema_slow) / ema_slow

IndicatorRegistry.register(
    IndicatorMetadata(
        name="ppo",
        full_name="Percentage Price Oscillator",
        category=IndicatorCategory.MOMENTUM,
        description="MACD expressed as percentage",
        params=[
            IndicatorParam("fast", "int", 12, 1, 100, "Fast EMA period"),
            IndicatorParam("slow", "int", 26, 1, 200, "Slow EMA period"),
        ],
        outputs=[IndicatorOutput("ppo", "PPO values (%)")],
        talib_func="PPO",
        pandas_ta_func="ppo",
        required_columns=["close"],
    ),
    _calc_ppo
)


# =============================================================================
# Ultimate Oscillator
# =============================================================================
def _calc_ultosc(data: pd.DataFrame, use_talib: bool = True,
                 period1: int = 7, period2: int = 14, period3: int = 28,
                 **kwargs) -> pd.Series:
    """Calculate Ultimate Oscillator."""
    if use_talib and _TALIB:
        return pd.Series(
            talib.ULTOSC(data["high"], data["low"], data["close"],
                         timeperiod1=period1, timeperiod2=period2, timeperiod3=period3),
            index=data.index, name="ULTOSC"
        )
    elif _PANDAS_TA:
        return ta.uo(data["high"], data["low"], data["close"],
                     fast=period1, medium=period2, slow=period3)
    else:
        raise NotImplementedError("Ultimate Oscillator requires TA-Lib or pandas-ta")

IndicatorRegistry.register(
    IndicatorMetadata(
        name="ultosc",
        full_name="Ultimate Oscillator",
        category=IndicatorCategory.MOMENTUM,
        description="Multi-timeframe momentum oscillator",
        params=[
            IndicatorParam("period1", "int", 7, 1, 50, "Short period"),
            IndicatorParam("period2", "int", 14, 1, 100, "Medium period"),
            IndicatorParam("period3", "int", 28, 1, 200, "Long period"),
        ],
        outputs=[IndicatorOutput("ultosc", "Ultimate Oscillator values (0-100)")],
        talib_func="ULTOSC",
        pandas_ta_func="uo",
        required_columns=["high", "low", "close"],
    ),
    _calc_ultosc
)


# =============================================================================
# TRIX
# =============================================================================
def _calc_trix(data: pd.DataFrame, use_talib: bool = True, source: str = "close",
               length: int = 30, **kwargs) -> pd.Series:
    """Calculate TRIX."""
    src = data[source]
    if use_talib and _TALIB:
        return pd.Series(talib.TRIX(src, timeperiod=length), index=data.index, name=f"TRIX_{length}")
    elif _PANDAS_TA:
        result = ta.trix(src, length=length)
        if isinstance(result, pd.DataFrame):
            return result.iloc[:, 0]
        return result
    else:
        ema1 = src.ewm(span=length, adjust=False).mean()
        ema2 = ema1.ewm(span=length, adjust=False).mean()
        ema3 = ema2.ewm(span=length, adjust=False).mean()
        return 100 * (ema3 - ema3.shift(1)) / ema3.shift(1)

IndicatorRegistry.register(
    IndicatorMetadata(
        name="trix",
        full_name="TRIX",
        category=IndicatorCategory.MOMENTUM,
        description="Triple smoothed EMA rate of change",
        params=[
            IndicatorParam("length", "int", 30, 1, 100, "Period length"),
        ],
        outputs=[IndicatorOutput("trix", "TRIX values")],
        talib_func="TRIX",
        pandas_ta_func="trix",
        required_columns=["close"],
    ),
    _calc_trix
)


# =============================================================================
# Chande Momentum Oscillator (CMO)
# =============================================================================
def _calc_cmo(data: pd.DataFrame, use_talib: bool = True, source: str = "close",
              length: int = 14, **kwargs) -> pd.Series:
    """Calculate Chande Momentum Oscillator."""
    src = data[source]
    if use_talib and _TALIB:
        return pd.Series(talib.CMO(src, timeperiod=length), index=data.index, name=f"CMO_{length}")
    elif _PANDAS_TA:
        return ta.cmo(src, length=length)
    else:
        delta = src.diff()
        gain = delta.where(delta > 0, 0).rolling(window=length).sum()
        loss = (-delta.where(delta < 0, 0)).rolling(window=length).sum()
        return 100 * (gain - loss) / (gain + loss)

IndicatorRegistry.register(
    IndicatorMetadata(
        name="cmo",
        full_name="Chande Momentum Oscillator",
        category=IndicatorCategory.MOMENTUM,
        description="Momentum oscillator using gains and losses",
        params=[
            IndicatorParam("length", "int", 14, 1, 100, "Period length"),
        ],
        outputs=[IndicatorOutput("cmo", "CMO values (-100 to 100)")],
        talib_func="CMO",
        pandas_ta_func="cmo",
        required_columns=["close"],
    ),
    _calc_cmo
)


# =============================================================================
# Absolute Price Oscillator (APO)
# =============================================================================
def _calc_apo(data: pd.DataFrame, use_talib: bool = True, source: str = "close",
              fast: int = 12, slow: int = 26, **kwargs) -> pd.Series:
    """Calculate Absolute Price Oscillator."""
    src = data[source]
    if use_talib and _TALIB:
        return pd.Series(
            talib.APO(src, fastperiod=fast, slowperiod=slow),
            index=data.index, name=f"APO_{fast}_{slow}"
        )
    elif _PANDAS_TA:
        return ta.apo(src, fast=fast, slow=slow)
    else:
        ema_fast = src.ewm(span=fast, adjust=False).mean()
        ema_slow = src.ewm(span=slow, adjust=False).mean()
        return ema_fast - ema_slow

IndicatorRegistry.register(
    IndicatorMetadata(
        name="apo",
        full_name="Absolute Price Oscillator",
        category=IndicatorCategory.MOMENTUM,
        description="Difference between fast and slow EMAs",
        params=[
            IndicatorParam("fast", "int", 12, 1, 100, "Fast EMA period"),
            IndicatorParam("slow", "int", 26, 1, 200, "Slow EMA period"),
        ],
        outputs=[IndicatorOutput("apo", "APO values")],
        talib_func="APO",
        pandas_ta_func="apo",
        required_columns=["close"],
    ),
    _calc_apo
)


# =============================================================================
# Plus/Minus Directional Movement
# =============================================================================
def _calc_plus_dm(data: pd.DataFrame, use_talib: bool = True,
                  length: int = 14, **kwargs) -> pd.Series:
    """Calculate Plus Directional Movement."""
    if use_talib and _TALIB:
        return pd.Series(
            talib.PLUS_DM(data["high"], data["low"], timeperiod=length),
            index=data.index, name=f"PLUS_DM_{length}"
        )
    else:
        raise NotImplementedError("+DM requires TA-Lib")

IndicatorRegistry.register(
    IndicatorMetadata(
        name="plus_dm",
        full_name="Plus Directional Movement",
        category=IndicatorCategory.MOMENTUM,
        description="Upward price movement",
        params=[
            IndicatorParam("length", "int", 14, 1, 100, "Period length"),
        ],
        outputs=[IndicatorOutput("plus_dm", "+DM values")],
        talib_func="PLUS_DM",
        pandas_ta_func=None,
        required_columns=["high", "low"],
    ),
    _calc_plus_dm
)


def _calc_minus_dm(data: pd.DataFrame, use_talib: bool = True,
                   length: int = 14, **kwargs) -> pd.Series:
    """Calculate Minus Directional Movement."""
    if use_talib and _TALIB:
        return pd.Series(
            talib.MINUS_DM(data["high"], data["low"], timeperiod=length),
            index=data.index, name=f"MINUS_DM_{length}"
        )
    else:
        raise NotImplementedError("-DM requires TA-Lib")

IndicatorRegistry.register(
    IndicatorMetadata(
        name="minus_dm",
        full_name="Minus Directional Movement",
        category=IndicatorCategory.MOMENTUM,
        description="Downward price movement",
        params=[
            IndicatorParam("length", "int", 14, 1, 100, "Period length"),
        ],
        outputs=[IndicatorOutput("minus_dm", "-DM values")],
        talib_func="MINUS_DM",
        pandas_ta_func=None,
        required_columns=["high", "low"],
    ),
    _calc_minus_dm
)


# =============================================================================
# Directional Movement Index (DX)
# =============================================================================
def _calc_dx(data: pd.DataFrame, use_talib: bool = True,
             length: int = 14, **kwargs) -> pd.Series:
    """Calculate Directional Movement Index."""
    if use_talib and _TALIB:
        return pd.Series(
            talib.DX(data["high"], data["low"], data["close"], timeperiod=length),
            index=data.index, name=f"DX_{length}"
        )
    else:
        raise NotImplementedError("DX requires TA-Lib")

IndicatorRegistry.register(
    IndicatorMetadata(
        name="dx",
        full_name="Directional Movement Index",
        category=IndicatorCategory.MOMENTUM,
        description="Measures trend strength",
        params=[
            IndicatorParam("length", "int", 14, 1, 100, "Period length"),
        ],
        outputs=[IndicatorOutput("dx", "DX values")],
        talib_func="DX",
        pandas_ta_func=None,
        required_columns=["high", "low", "close"],
    ),
    _calc_dx
)


# =============================================================================
# Average Directional Movement Index Rating (ADXR)
# =============================================================================
def _calc_adxr(data: pd.DataFrame, use_talib: bool = True,
               length: int = 14, **kwargs) -> pd.Series:
    """Calculate ADXR."""
    if use_talib and _TALIB:
        return pd.Series(
            talib.ADXR(data["high"], data["low"], data["close"], timeperiod=length),
            index=data.index, name=f"ADXR_{length}"
        )
    else:
        raise NotImplementedError("ADXR requires TA-Lib")

IndicatorRegistry.register(
    IndicatorMetadata(
        name="adxr",
        full_name="Average Directional Movement Index Rating",
        category=IndicatorCategory.MOMENTUM,
        description="Smoothed ADX",
        params=[
            IndicatorParam("length", "int", 14, 1, 100, "Period length"),
        ],
        outputs=[IndicatorOutput("adxr", "ADXR values")],
        talib_func="ADXR",
        pandas_ta_func=None,
        required_columns=["high", "low", "close"],
    ),
    _calc_adxr
)


# =============================================================================
# Choppiness Index - pandas-ta only
# =============================================================================
def _calc_chop(data: pd.DataFrame, use_talib: bool = True,
               length: int = 14, **kwargs) -> pd.Series:
    """Calculate Choppiness Index."""
    if _PANDAS_TA:
        return ta.chop(data["high"], data["low"], data["close"], length=length)
    else:
        raise NotImplementedError("Choppiness Index requires pandas-ta")

IndicatorRegistry.register(
    IndicatorMetadata(
        name="chop",
        full_name="Choppiness Index",
        category=IndicatorCategory.MOMENTUM,
        description="Measures market trendiness vs choppiness",
        params=[
            IndicatorParam("length", "int", 14, 1, 100, "Period length"),
        ],
        outputs=[IndicatorOutput("chop", "Chop values (0-100)")],
        talib_func=None,
        pandas_ta_func="chop",
        required_columns=["high", "low", "close"],
    ),
    _calc_chop
)
