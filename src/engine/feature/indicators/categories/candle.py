"""Candlestick Pattern Recognition.

This module contains candlestick pattern recognition indicators.
Priority: Use TA-Lib if available, fallback to pandas-ta.
"""

from typing import Any

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
# Candlestick Pattern Metadata
# =============================================================================
# All candlestick patterns from TA-Lib
CDL_PATTERNS = {
    "cdl_2crows": ("CDL2CROWS", "Two Crows"),
    "cdl_3blackcrows": ("CDL3BLACKCROWS", "Three Black Crows"),
    "cdl_3inside": ("CDL3INSIDE", "Three Inside Up/Down"),
    "cdl_3linestrike": ("CDL3LINESTRIKE", "Three-Line Strike"),
    "cdl_3outside": ("CDL3OUTSIDE", "Three Outside Up/Down"),
    "cdl_3starsinsouth": ("CDL3STARSINSOUTH", "Three Stars In The South"),
    "cdl_3whitesoldiers": ("CDL3WHITESOLDIERS", "Three Advancing White Soldiers"),
    "cdl_abandonedbaby": ("CDLABANDONEDBABY", "Abandoned Baby"),
    "cdl_advanceblock": ("CDLADVANCEBLOCK", "Advance Block"),
    "cdl_belthold": ("CDLBELTHOLD", "Belt-hold"),
    "cdl_breakaway": ("CDLBREAKAWAY", "Breakaway"),
    "cdl_closingmarubozu": ("CDLCLOSINGMARUBOZU", "Closing Marubozu"),
    "cdl_concealbabyswall": ("CDLCONCEALBABYSWALL", "Concealing Baby Swallow"),
    "cdl_counterattack": ("CDLCOUNTERATTACK", "Counterattack"),
    "cdl_darkcloudcover": ("CDLDARKCLOUDCOVER", "Dark Cloud Cover"),
    "cdl_doji": ("CDLDOJI", "Doji"),
    "cdl_dojistar": ("CDLDOJISTAR", "Doji Star"),
    "cdl_dragonflydoji": ("CDLDRAGONFLYDOJI", "Dragonfly Doji"),
    "cdl_engulfing": ("CDLENGULFING", "Engulfing Pattern"),
    "cdl_eveningdojistar": ("CDLEVENINGDOJISTAR", "Evening Doji Star"),
    "cdl_eveningstar": ("CDLEVENINGSTAR", "Evening Star"),
    "cdl_gapsidesidewhite": ("CDLGAPSIDESIDEWHITE", "Up/Down-gap side-by-side white lines"),
    "cdl_gravestonedoji": ("CDLGRAVESTONEDOJI", "Gravestone Doji"),
    "cdl_hammer": ("CDLHAMMER", "Hammer"),
    "cdl_hangingman": ("CDLHANGINGMAN", "Hanging Man"),
    "cdl_harami": ("CDLHARAMI", "Harami Pattern"),
    "cdl_haramicross": ("CDLHARAMICROSS", "Harami Cross Pattern"),
    "cdl_highwave": ("CDLHIGHWAVE", "High-Wave Candle"),
    "cdl_hikkake": ("CDLHIKKAKE", "Hikkake Pattern"),
    "cdl_hikkakemod": ("CDLHIKKAKEMOD", "Modified Hikkake Pattern"),
    "cdl_homingpigeon": ("CDLHOMINGPIGEON", "Homing Pigeon"),
    "cdl_identical3crows": ("CDLIDENTICAL3CROWS", "Identical Three Crows"),
    "cdl_inneck": ("CDLINNECK", "In-Neck Pattern"),
    "cdl_invertedhammer": ("CDLINVERTEDHAMMER", "Inverted Hammer"),
    "cdl_kicking": ("CDLKICKING", "Kicking"),
    "cdl_kickingbylength": ("CDLKICKINGBYLENGTH", "Kicking - bull/bear determined by the longer marubozu"),
    "cdl_ladderbottom": ("CDLLADDERBOTTOM", "Ladder Bottom"),
    "cdl_longleggeddoji": ("CDLLONGLEGGEDDOJI", "Long Legged Doji"),
    "cdl_longline": ("CDLLONGLINE", "Long Line Candle"),
    "cdl_marubozu": ("CDLMARUBOZU", "Marubozu"),
    "cdl_matchinglow": ("CDLMATCHINGLOW", "Matching Low"),
    "cdl_mathold": ("CDLMATHOLD", "Mat Hold"),
    "cdl_morningdojistar": ("CDLMORNINGDOJISTAR", "Morning Doji Star"),
    "cdl_morningstar": ("CDLMORNINGSTAR", "Morning Star"),
    "cdl_onneck": ("CDLONNECK", "On-Neck Pattern"),
    "cdl_piercing": ("CDLPIERCING", "Piercing Pattern"),
    "cdl_rickshawman": ("CDLRICKSHAWMAN", "Rickshaw Man"),
    "cdl_risefall3methods": ("CDLRISEFALL3METHODS", "Rising/Falling Three Methods"),
    "cdl_separatinglines": ("CDLSEPARATINGLINES", "Separating Lines"),
    "cdl_shootingstar": ("CDLSHOOTINGSTAR", "Shooting Star"),
    "cdl_shortline": ("CDLSHORTLINE", "Short Line Candle"),
    "cdl_spinningtop": ("CDLSPINNINGTOP", "Spinning Top"),
    "cdl_stalledpattern": ("CDLSTALLEDPATTERN", "Stalled Pattern"),
    "cdl_sticksandwich": ("CDLSTICKSANDWICH", "Stick Sandwich"),
    "cdl_takuri": ("CDLTAKURI", "Takuri (Dragonfly Doji with very long lower shadow)"),
    "cdl_tasukigap": ("CDLTASUKIGAP", "Tasuki Gap"),
    "cdl_thrusting": ("CDLTHRUSTING", "Thrusting Pattern"),
    "cdl_tristar": ("CDLTRISTAR", "Tristar Pattern"),
    "cdl_unique3river": ("CDLUNIQUE3RIVER", "Unique 3 River"),
    "cdl_upsidegap2crows": ("CDLUPSIDEGAP2CROWS", "Upside Gap Two Crows"),
    "cdl_xsidegap3methods": ("CDLXSIDEGAP3METHODS", "Upside/Downside Gap Three Methods"),
}


def _create_cdl_calculator(talib_func_name: str):
    """Create a calculator function for a candlestick pattern."""
    def calculator(data: pd.DataFrame, use_talib: bool = True,
                   penetration: float = 0.0, **kwargs) -> pd.Series:
        if use_talib and _TALIB:
            func = getattr(talib, talib_func_name)
            # Some patterns have penetration parameter
            if talib_func_name in ["CDLABANDONEDBABY", "CDLDARKCLOUDCOVER", 
                                   "CDLEVENINGDOJISTAR", "CDLEVENINGSTAR",
                                   "CDLMATHOLD", "CDLMORNINGDOJISTAR", 
                                   "CDLMORNINGSTAR"]:
                result = func(data["open"], data["high"], data["low"], data["close"],
                             penetration=penetration)
            else:
                result = func(data["open"], data["high"], data["low"], data["close"])
            return pd.Series(result, index=data.index, name=talib_func_name)
        elif _PANDAS_TA:
            # pandas-ta uses cdl_pattern with name parameter
            pattern_name = talib_func_name.replace("CDL", "").lower()
            return ta.cdl_pattern(data["open"], data["high"], data["low"], data["close"],
                                  name=pattern_name)
        else:
            raise NotImplementedError(f"{talib_func_name} requires TA-Lib or pandas-ta")
    return calculator


# Register all candlestick patterns
for indicator_name, (talib_func, full_name) in CDL_PATTERNS.items():
    # Determine if pattern has penetration parameter
    has_penetration = talib_func in ["CDLABANDONEDBABY", "CDLDARKCLOUDCOVER", 
                                     "CDLEVENINGDOJISTAR", "CDLEVENINGSTAR",
                                     "CDLMATHOLD", "CDLMORNINGDOJISTAR", 
                                     "CDLMORNINGSTAR"]
    
    params = []
    if has_penetration:
        params.append(IndicatorParam("penetration", "float", 0.0, 0.0, 1.0, "Penetration threshold"))
    
    IndicatorRegistry.register(
        IndicatorMetadata(
            name=indicator_name,
            full_name=full_name,
            category=IndicatorCategory.CANDLE,
            description=f"Candlestick pattern: {full_name}",
            params=params,
            outputs=[IndicatorOutput(indicator_name, "Pattern signal: +100 bullish, -100 bearish, 0 no pattern")],
            talib_func=talib_func,
            pandas_ta_func="cdl_pattern",
            required_columns=["open", "high", "low", "close"],
        ),
        _create_cdl_calculator(talib_func)
    )


# =============================================================================
# All Patterns (combined)
# =============================================================================
def _calc_cdl_all(data: pd.DataFrame, use_talib: bool = True, **kwargs) -> pd.DataFrame:
    """Calculate all candlestick patterns."""
    results = {}
    
    if use_talib and _TALIB:
        for indicator_name, (talib_func, _) in CDL_PATTERNS.items():
            try:
                func = getattr(talib, talib_func)
                results[indicator_name] = func(data["open"], data["high"], 
                                               data["low"], data["close"])
            except Exception:
                pass
    elif _PANDAS_TA:
        result = ta.cdl_pattern(data["open"], data["high"], data["low"], data["close"],
                                name="all")
        return result
    
    return pd.DataFrame(results, index=data.index)

IndicatorRegistry.register(
    IndicatorMetadata(
        name="cdl_all",
        full_name="All Candlestick Patterns",
        category=IndicatorCategory.CANDLE,
        description="Calculate all candlestick patterns at once",
        params=[],
        outputs=[IndicatorOutput("cdl_all", "DataFrame with all patterns")],
        talib_func=None,
        pandas_ta_func="cdl_pattern",
        required_columns=["open", "high", "low", "close"],
    ),
    _calc_cdl_all
)


# =============================================================================
# Heikin Ashi - pandas-ta only
# =============================================================================
def _calc_ha(data: pd.DataFrame, use_talib: bool = True, **kwargs) -> pd.DataFrame:
    """Calculate Heikin Ashi candles."""
    if _PANDAS_TA:
        return ta.ha(data["open"], data["high"], data["low"], data["close"])
    else:
        ha_close = (data["open"] + data["high"] + data["low"] + data["close"]) / 4
        
        ha_open = pd.Series(index=data.index, dtype=float)
        ha_open.iloc[0] = (data["open"].iloc[0] + data["close"].iloc[0]) / 2
        
        for i in range(1, len(data)):
            ha_open.iloc[i] = (ha_open.iloc[i - 1] + ha_close.iloc[i - 1]) / 2
        
        ha_high = pd.concat([data["high"], ha_open, ha_close], axis=1).max(axis=1)
        ha_low = pd.concat([data["low"], ha_open, ha_close], axis=1).min(axis=1)
        
        return pd.DataFrame({
            "HA_open": ha_open,
            "HA_high": ha_high,
            "HA_low": ha_low,
            "HA_close": ha_close
        }, index=data.index)

IndicatorRegistry.register(
    IndicatorMetadata(
        name="ha",
        full_name="Heikin Ashi",
        category=IndicatorCategory.CANDLE,
        description="Smoothed candlesticks for trend identification",
        params=[],
        outputs=[
            IndicatorOutput("HA_open", "Heikin Ashi Open"),
            IndicatorOutput("HA_high", "Heikin Ashi High"),
            IndicatorOutput("HA_low", "Heikin Ashi Low"),
            IndicatorOutput("HA_close", "Heikin Ashi Close"),
        ],
        talib_func=None,
        pandas_ta_func="ha",
        required_columns=["open", "high", "low", "close"],
    ),
    _calc_ha
)


# =============================================================================
# Inside Bar - pandas-ta only
# =============================================================================
def _calc_cdl_inside(data: pd.DataFrame, use_talib: bool = True, **kwargs) -> pd.Series:
    """Detect Inside Bar pattern."""
    if _PANDAS_TA:
        return ta.cdl_inside(data["open"], data["high"], data["low"], data["close"])
    else:
        inside = (data["high"] < data["high"].shift(1)) & (data["low"] > data["low"].shift(1))
        return inside.astype(int) * 100

IndicatorRegistry.register(
    IndicatorMetadata(
        name="cdl_inside",
        full_name="Inside Bar",
        category=IndicatorCategory.CANDLE,
        description="Detects inside bar pattern (narrowing range)",
        params=[],
        outputs=[IndicatorOutput("cdl_inside", "Inside bar signal")],
        talib_func=None,
        pandas_ta_func="cdl_inside",
        required_columns=["open", "high", "low", "close"],
    ),
    _calc_cdl_inside
)


# =============================================================================
# Z Candles - pandas-ta only
# =============================================================================
def _calc_cdl_z(data: pd.DataFrame, use_talib: bool = True,
                length: int = 10, **kwargs) -> pd.DataFrame:
    """Calculate Z-score normalized candles."""
    if _PANDAS_TA:
        return ta.cdl_z(data["open"], data["high"], data["low"], data["close"],
                        length=length)
    else:
        raise NotImplementedError("cdl_z requires pandas-ta")

IndicatorRegistry.register(
    IndicatorMetadata(
        name="cdl_z",
        full_name="Z Candles",
        category=IndicatorCategory.CANDLE,
        description="Z-score normalized OHLC values",
        params=[
            IndicatorParam("length", "int", 10, 1, 100, "Z-score lookback period"),
        ],
        outputs=[
            IndicatorOutput("Z_open", "Z-score of Open"),
            IndicatorOutput("Z_high", "Z-score of High"),
            IndicatorOutput("Z_low", "Z-score of Low"),
            IndicatorOutput("Z_close", "Z-score of Close"),
        ],
        talib_func=None,
        pandas_ta_func="cdl_z",
        required_columns=["open", "high", "low", "close"],
    ),
    _calc_cdl_z
)
