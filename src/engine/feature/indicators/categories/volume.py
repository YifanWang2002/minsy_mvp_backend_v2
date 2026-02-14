"""Volume Indicators.

This module contains volume-based indicators.
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
# On Balance Volume (OBV)
# =============================================================================
def _calc_obv(data: pd.DataFrame, use_talib: bool = True, **kwargs) -> pd.Series:
    """Calculate On Balance Volume."""
    if use_talib and _TALIB:
        return pd.Series(
            talib.OBV(data["close"], data["volume"]),
            index=data.index, name="OBV"
        )
    elif _PANDAS_TA:
        return ta.obv(data["close"], data["volume"])
    else:
        direction = np.sign(data["close"].diff())
        return (direction * data["volume"]).cumsum()

IndicatorRegistry.register(
    IndicatorMetadata(
        name="obv",
        full_name="On Balance Volume",
        category=IndicatorCategory.VOLUME,
        description="Cumulative volume based on price direction",
        params=[],
        outputs=[IndicatorOutput("obv", "OBV values")],
        talib_func="OBV",
        pandas_ta_func="obv",
        required_columns=["close", "volume"],
    ),
    _calc_obv
)


# =============================================================================
# Accumulation/Distribution Line (AD)
# =============================================================================
def _calc_ad(data: pd.DataFrame, use_talib: bool = True, **kwargs) -> pd.Series:
    """Calculate Accumulation/Distribution Line."""
    if use_talib and _TALIB:
        return pd.Series(
            talib.AD(data["high"], data["low"], data["close"], data["volume"]),
            index=data.index, name="AD"
        )
    elif _PANDAS_TA:
        return ta.ad(data["high"], data["low"], data["close"], data["volume"])
    else:
        clv = ((data["close"] - data["low"]) - (data["high"] - data["close"])) / \
              (data["high"] - data["low"]).replace(0, np.nan)
        return (clv * data["volume"]).cumsum()

IndicatorRegistry.register(
    IndicatorMetadata(
        name="ad",
        full_name="Accumulation/Distribution Line",
        category=IndicatorCategory.VOLUME,
        description="Measures cumulative money flow",
        params=[],
        outputs=[IndicatorOutput("ad", "A/D Line values")],
        talib_func="AD",
        pandas_ta_func="ad",
        required_columns=["high", "low", "close", "volume"],
    ),
    _calc_ad
)


# =============================================================================
# Accumulation/Distribution Oscillator (ADOSC)
# =============================================================================
def _calc_adosc(data: pd.DataFrame, use_talib: bool = True,
                fast: int = 3, slow: int = 10, **kwargs) -> pd.Series:
    """Calculate A/D Oscillator (Chaikin Oscillator)."""
    if use_talib and _TALIB:
        return pd.Series(
            talib.ADOSC(data["high"], data["low"], data["close"], data["volume"],
                        fastperiod=fast, slowperiod=slow),
            index=data.index, name="ADOSC"
        )
    elif _PANDAS_TA:
        return ta.adosc(data["high"], data["low"], data["close"], data["volume"],
                        fast=fast, slow=slow)
    else:
        ad = _calc_ad(data, use_talib=False)
        return ad.ewm(span=fast, adjust=False).mean() - ad.ewm(span=slow, adjust=False).mean()

IndicatorRegistry.register(
    IndicatorMetadata(
        name="adosc",
        full_name="Chaikin A/D Oscillator",
        category=IndicatorCategory.VOLUME,
        description="Momentum of A/D line",
        params=[
            IndicatorParam("fast", "int", 3, 1, 50, "Fast EMA period"),
            IndicatorParam("slow", "int", 10, 1, 100, "Slow EMA period"),
        ],
        outputs=[IndicatorOutput("adosc", "A/D Oscillator values")],
        talib_func="ADOSC",
        pandas_ta_func="adosc",
        required_columns=["high", "low", "close", "volume"],
    ),
    _calc_adosc
)


# =============================================================================
# Money Flow Index (MFI)
# =============================================================================
def _calc_mfi(data: pd.DataFrame, use_talib: bool = True,
              length: int = 14, **kwargs) -> pd.Series:
    """Calculate Money Flow Index."""
    if use_talib and _TALIB:
        return pd.Series(
            talib.MFI(data["high"], data["low"], data["close"], data["volume"],
                      timeperiod=length),
            index=data.index, name=f"MFI_{length}"
        )
    elif _PANDAS_TA:
        return ta.mfi(data["high"], data["low"], data["close"], data["volume"],
                      length=length)
    else:
        typical_price = (data["high"] + data["low"] + data["close"]) / 3
        raw_money_flow = typical_price * data["volume"]
        
        positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0)
        
        positive_mf = positive_flow.rolling(window=length).sum()
        negative_mf = negative_flow.rolling(window=length).sum()
        
        money_ratio = positive_mf / negative_mf.replace(0, np.nan)
        return 100 - (100 / (1 + money_ratio))

IndicatorRegistry.register(
    IndicatorMetadata(
        name="mfi",
        full_name="Money Flow Index",
        category=IndicatorCategory.VOLUME,
        description="Volume-weighted RSI",
        params=[
            IndicatorParam("length", "int", 14, 1, 100, "Period length"),
        ],
        outputs=[IndicatorOutput("mfi", "MFI values (0-100)")],
        talib_func="MFI",
        pandas_ta_func="mfi",
        required_columns=["high", "low", "close", "volume"],
    ),
    _calc_mfi
)


# =============================================================================
# Chaikin Money Flow (CMF) - pandas-ta only
# =============================================================================
def _calc_cmf(data: pd.DataFrame, use_talib: bool = True,
              length: int = 20, **kwargs) -> pd.Series:
    """Calculate Chaikin Money Flow."""
    if _PANDAS_TA:
        return ta.cmf(data["high"], data["low"], data["close"], data["volume"],
                      length=length)
    else:
        clv = ((data["close"] - data["low"]) - (data["high"] - data["close"])) / \
              (data["high"] - data["low"]).replace(0, np.nan)
        money_flow_volume = clv * data["volume"]
        return money_flow_volume.rolling(window=length).sum() / \
               data["volume"].rolling(window=length).sum()

IndicatorRegistry.register(
    IndicatorMetadata(
        name="cmf",
        full_name="Chaikin Money Flow",
        category=IndicatorCategory.VOLUME,
        description="Measures money flow over a period",
        params=[
            IndicatorParam("length", "int", 20, 1, 100, "Period length"),
        ],
        outputs=[IndicatorOutput("cmf", "CMF values (-1 to 1)")],
        talib_func=None,
        pandas_ta_func="cmf",
        required_columns=["high", "low", "close", "volume"],
    ),
    _calc_cmf
)


# =============================================================================
# Elder's Force Index (EFI) - pandas-ta only
# =============================================================================
def _calc_efi(data: pd.DataFrame, use_talib: bool = True,
              length: int = 13, **kwargs) -> pd.Series:
    """Calculate Elder's Force Index."""
    if _PANDAS_TA:
        return ta.efi(data["close"], data["volume"], length=length)
    else:
        force = data["close"].diff() * data["volume"]
        return force.ewm(span=length, adjust=False).mean()

IndicatorRegistry.register(
    IndicatorMetadata(
        name="efi",
        full_name="Elder's Force Index",
        category=IndicatorCategory.VOLUME,
        description="Combines price change and volume",
        params=[
            IndicatorParam("length", "int", 13, 1, 100, "EMA period"),
        ],
        outputs=[IndicatorOutput("efi", "EFI values")],
        talib_func=None,
        pandas_ta_func="efi",
        required_columns=["close", "volume"],
    ),
    _calc_efi
)


# =============================================================================
# Ease of Movement (EOM) - pandas-ta only
# =============================================================================
def _calc_eom(data: pd.DataFrame, use_talib: bool = True,
              length: int = 14, divisor: int = 100000000, **kwargs) -> pd.Series:
    """Calculate Ease of Movement."""
    if _PANDAS_TA:
        return ta.eom(data["high"], data["low"], data["close"], data["volume"],
                      length=length, divisor=divisor)
    else:
        dm = ((data["high"] + data["low"]) / 2).diff()
        br = (data["volume"] / divisor) / (data["high"] - data["low"]).replace(0, np.nan)
        eom = dm / br
        return eom.rolling(window=length).mean()

IndicatorRegistry.register(
    IndicatorMetadata(
        name="eom",
        full_name="Ease of Movement",
        category=IndicatorCategory.VOLUME,
        description="Price movement per unit volume",
        params=[
            IndicatorParam("length", "int", 14, 1, 100, "Smoothing period"),
            IndicatorParam("divisor", "int", 100000000, 1, 1000000000, "Volume divisor"),
        ],
        outputs=[IndicatorOutput("eom", "EOM values")],
        talib_func=None,
        pandas_ta_func="eom",
        required_columns=["high", "low", "close", "volume"],
    ),
    _calc_eom
)


# =============================================================================
# Klinger Volume Oscillator (KVO) - pandas-ta only
# =============================================================================
def _calc_kvo(data: pd.DataFrame, use_talib: bool = True,
              fast: int = 34, slow: int = 55, signal: int = 13, **kwargs) -> pd.DataFrame:
    """Calculate Klinger Volume Oscillator."""
    if _PANDAS_TA:
        return ta.kvo(data["high"], data["low"], data["close"], data["volume"],
                      fast=fast, slow=slow, signal=signal)
    else:
        raise NotImplementedError("KVO requires pandas-ta")

IndicatorRegistry.register(
    IndicatorMetadata(
        name="kvo",
        full_name="Klinger Volume Oscillator",
        category=IndicatorCategory.VOLUME,
        description="Volume-based trend indicator",
        params=[
            IndicatorParam("fast", "int", 34, 1, 100, "Fast EMA period"),
            IndicatorParam("slow", "int", 55, 1, 200, "Slow EMA period"),
            IndicatorParam("signal", "int", 13, 1, 50, "Signal line period"),
        ],
        outputs=[
            IndicatorOutput("KVO", "Klinger Volume Oscillator"),
            IndicatorOutput("KVOs", "Signal line"),
        ],
        talib_func=None,
        pandas_ta_func="kvo",
        required_columns=["high", "low", "close", "volume"],
    ),
    _calc_kvo
)


# =============================================================================
# Negative Volume Index (NVI) - pandas-ta only
# =============================================================================
def _calc_nvi(data: pd.DataFrame, use_talib: bool = True,
              length: int = 255, initial: int = 1000, **kwargs) -> pd.Series:
    """Calculate Negative Volume Index."""
    if _PANDAS_TA:
        return ta.nvi(data["close"], data["volume"], length=length, initial=initial)
    else:
        nvi = pd.Series(index=data.index, dtype=float)
        nvi.iloc[0] = initial
        
        for i in range(1, len(data)):
            if data["volume"].iloc[i] < data["volume"].iloc[i - 1]:
                pct_change = (data["close"].iloc[i] - data["close"].iloc[i - 1]) / \
                            data["close"].iloc[i - 1]
                nvi.iloc[i] = nvi.iloc[i - 1] * (1 + pct_change)
            else:
                nvi.iloc[i] = nvi.iloc[i - 1]
        
        return nvi

IndicatorRegistry.register(
    IndicatorMetadata(
        name="nvi",
        full_name="Negative Volume Index",
        category=IndicatorCategory.VOLUME,
        description="Tracks price on declining volume days",
        params=[
            IndicatorParam("length", "int", 255, 1, 500, "EMA period"),
            IndicatorParam("initial", "int", 1000, 1, 10000, "Initial value"),
        ],
        outputs=[IndicatorOutput("nvi", "NVI values")],
        talib_func=None,
        pandas_ta_func="nvi",
        required_columns=["close", "volume"],
    ),
    _calc_nvi
)


# =============================================================================
# Positive Volume Index (PVI) - pandas-ta only
# =============================================================================
def _calc_pvi(data: pd.DataFrame, use_talib: bool = True,
              length: int = 255, initial: int = 1000, **kwargs) -> pd.Series:
    """Calculate Positive Volume Index."""
    if _PANDAS_TA:
        result = ta.pvi(data["close"], data["volume"], length=length, initial=initial)
        if isinstance(result, pd.DataFrame):
            return result.iloc[:, 0]
        return result
    else:
        pvi = pd.Series(index=data.index, dtype=float)
        pvi.iloc[0] = initial
        
        for i in range(1, len(data)):
            if data["volume"].iloc[i] > data["volume"].iloc[i - 1]:
                pct_change = (data["close"].iloc[i] - data["close"].iloc[i - 1]) / \
                            data["close"].iloc[i - 1]
                pvi.iloc[i] = pvi.iloc[i - 1] * (1 + pct_change)
            else:
                pvi.iloc[i] = pvi.iloc[i - 1]
        
        return pvi

IndicatorRegistry.register(
    IndicatorMetadata(
        name="pvi",
        full_name="Positive Volume Index",
        category=IndicatorCategory.VOLUME,
        description="Tracks price on rising volume days",
        params=[
            IndicatorParam("length", "int", 255, 1, 500, "EMA period"),
            IndicatorParam("initial", "int", 1000, 1, 10000, "Initial value"),
        ],
        outputs=[IndicatorOutput("pvi", "PVI values")],
        talib_func=None,
        pandas_ta_func="pvi",
        required_columns=["close", "volume"],
    ),
    _calc_pvi
)


# =============================================================================
# Percentage Volume Oscillator (PVO) - pandas-ta only
# =============================================================================
def _calc_pvo(data: pd.DataFrame, use_talib: bool = True,
              fast: int = 12, slow: int = 26, signal: int = 9, **kwargs) -> pd.DataFrame:
    """Calculate Percentage Volume Oscillator."""
    if _PANDAS_TA:
        return ta.pvo(data["volume"], fast=fast, slow=slow, signal=signal)
    else:
        fast_ema = data["volume"].ewm(span=fast, adjust=False).mean()
        slow_ema = data["volume"].ewm(span=slow, adjust=False).mean()
        pvo = 100 * (fast_ema - slow_ema) / slow_ema
        pvo_signal = pvo.ewm(span=signal, adjust=False).mean()
        return pd.DataFrame({
            "PVO": pvo,
            "PVOs": pvo_signal,
            "PVOh": pvo - pvo_signal
        }, index=data.index)

IndicatorRegistry.register(
    IndicatorMetadata(
        name="pvo",
        full_name="Percentage Volume Oscillator",
        category=IndicatorCategory.VOLUME,
        description="MACD for volume",
        params=[
            IndicatorParam("fast", "int", 12, 1, 100, "Fast EMA period"),
            IndicatorParam("slow", "int", 26, 1, 200, "Slow EMA period"),
            IndicatorParam("signal", "int", 9, 1, 50, "Signal line period"),
        ],
        outputs=[
            IndicatorOutput("PVO", "PVO line"),
            IndicatorOutput("PVOs", "Signal line"),
            IndicatorOutput("PVOh", "Histogram"),
        ],
        talib_func=None,
        pandas_ta_func="pvo",
        required_columns=["volume"],
    ),
    _calc_pvo
)


# =============================================================================
# Price-Volume Trend (PVT) - pandas-ta only
# =============================================================================
def _calc_pvt(data: pd.DataFrame, use_talib: bool = True, **kwargs) -> pd.Series:
    """Calculate Price-Volume Trend."""
    if _PANDAS_TA:
        return ta.pvt(data["close"], data["volume"])
    else:
        pct_change = data["close"].pct_change()
        return (pct_change * data["volume"]).cumsum()

IndicatorRegistry.register(
    IndicatorMetadata(
        name="pvt",
        full_name="Price-Volume Trend",
        category=IndicatorCategory.VOLUME,
        description="Cumulative volume adjusted by price change %",
        params=[],
        outputs=[IndicatorOutput("pvt", "PVT values")],
        talib_func=None,
        pandas_ta_func="pvt",
        required_columns=["close", "volume"],
    ),
    _calc_pvt
)


# =============================================================================
# Volume Weighted Moving Average (VWMA) - pandas-ta only
# =============================================================================
def _calc_vwma(data: pd.DataFrame, use_talib: bool = True, source: str = "close",
               length: int = 20, **kwargs) -> pd.Series:
    """Calculate Volume Weighted Moving Average."""
    src = data[source]
    if _PANDAS_TA:
        return ta.vwma(src, data["volume"], length=length)
    else:
        return (src * data["volume"]).rolling(window=length).sum() / \
               data["volume"].rolling(window=length).sum()

IndicatorRegistry.register(
    IndicatorMetadata(
        name="vwma",
        full_name="Volume Weighted Moving Average",
        category=IndicatorCategory.VOLUME,
        description="MA weighted by volume",
        params=[
            IndicatorParam("length", "int", 20, 1, 500, "Period length"),
        ],
        outputs=[IndicatorOutput("vwma", "VWMA values")],
        talib_func=None,
        pandas_ta_func="vwma",
        required_columns=["close", "volume"],
    ),
    _calc_vwma
)
