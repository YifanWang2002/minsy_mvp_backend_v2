# 因子系统 README（自动生成）

> 本文件由 `scripts/generate_indicator_docs.py` 自动生成，请勿手改。

- 指标总数：`100`

## 分类统计

| category | count |
|---|---:|
| overlap | 25 |
| momentum | 22 |
| volatility | 16 |
| volume | 14 |
| utils | 23 |

## overlap (25)

| indicator | full_name | outputs | version | status | required_columns |
|---|---|---|---|---|---|
| `bbands` | Bollinger Bands | `BBU, BBM, BBL` | `1.0.0` | `active` | `close` |
| `dema` | Double Exponential Moving Average | `dema` | `1.0.0` | `active` | `close` |
| `donchian` | Donchian Channels | `DCL, DCM, DCU` | `1.0.0` | `active` | `high,low` |
| `ema` | Exponential Moving Average | `ema` | `1.0.0` | `active` | `close` |
| `hl2` | HL2 (High-Low Midpoint) | `hl2` | `1.0.0` | `active` | `high,low` |
| `hlc3` | HLC3 (Typical Price) | `hlc3` | `1.0.0` | `active` | `high,low,close` |
| `hma` | Hull Moving Average | `hma` | `1.0.0` | `active` | `close` |
| `ht_trendline` | Hilbert Transform - Instantaneous Trendline | `ht_trendline` | `1.0.0` | `active` | `close` |
| `ichimoku` | Ichimoku Kinkō Hyō | `ISA, ISB, ITS, IKS, ICS` | `1.0.0` | `active` | `high,low,close` |
| `kama` | Kaufman Adaptive Moving Average | `kama` | `1.0.0` | `active` | `close` |
| `kc` | Keltner Channels | `KCL, KCB, KCU` | `1.0.0` | `active` | `high,low,close` |
| `linearreg` | Linear Regression | `linearreg` | `1.0.0` | `active` | `close` |
| `mama` | MESA Adaptive Moving Average | `MAMA, FAMA` | `1.0.0` | `active` | `close` |
| `midpoint` | Midpoint over Period | `midpoint` | `1.0.0` | `active` | `close` |
| `midprice` | Midpoint Price over Period | `midprice` | `1.0.0` | `active` | `high,low` |
| `ohlc4` | OHLC4 | `ohlc4` | `1.0.0` | `active` | `open,high,low,close` |
| `sar` | Parabolic SAR | `sar` | `1.0.0` | `active` | `high,low` |
| `sma` | Simple Moving Average | `sma` | `1.0.0` | `active` | `close` |
| `supertrend` | SuperTrend | `SUPERT, SUPERTd, SUPERTl, SUPERTs` | `1.0.0` | `active` | `high,low,close` |
| `t3` | T3 - Triple Exponential Moving Average | `t3` | `1.0.0` | `active` | `close` |
| `tema` | Triple Exponential Moving Average | `tema` | `1.0.0` | `active` | `close` |
| `trima` | Triangular Moving Average | `trima` | `1.0.0` | `active` | `close` |
| `vwap` | Volume Weighted Average Price | `vwap` | `1.0.0` | `active` | `high,low,close,volume` |
| `wcp` | Weighted Close Price | `wcp` | `1.0.0` | `active` | `high,low,close` |
| `wma` | Weighted Moving Average | `wma` | `1.0.0` | `active` | `close` |

## momentum (22)

| indicator | full_name | outputs | version | status | required_columns |
|---|---|---|---|---|---|
| `adx` | Average Directional Index | `ADX, DMP, DMN` | `1.0.0` | `active` | `high,low,close` |
| `adxr` | Average Directional Movement Index Rating | `adxr` | `1.0.0` | `active` | `high,low,close` |
| `apo` | Absolute Price Oscillator | `apo` | `1.0.0` | `active` | `close` |
| `aroon` | Aroon | `AROONU, AROOND, AROONOSC` | `1.0.0` | `active` | `high,low` |
| `bop` | Balance of Power | `bop` | `1.0.0` | `active` | `open,high,low,close` |
| `cci` | Commodity Channel Index | `cci` | `1.0.0` | `active` | `high,low,close` |
| `chop` | Choppiness Index | `chop` | `1.0.0` | `active` | `high,low,close` |
| `cmo` | Chande Momentum Oscillator | `cmo` | `1.0.0` | `active` | `close` |
| `dx` | Directional Movement Index | `dx` | `1.0.0` | `active` | `high,low,close` |
| `macd` | Moving Average Convergence Divergence | `MACD, MACDs, MACDh` | `1.0.0` | `active` | `close` |
| `minus_dm` | Minus Directional Movement | `minus_dm` | `1.0.0` | `active` | `high,low` |
| `mom` | Momentum | `mom` | `1.0.0` | `active` | `close` |
| `plus_dm` | Plus Directional Movement | `plus_dm` | `1.0.0` | `active` | `high,low` |
| `ppo` | Percentage Price Oscillator | `ppo` | `1.0.0` | `active` | `close` |
| `roc` | Rate of Change | `roc` | `1.0.0` | `active` | `close` |
| `rsi` | Relative Strength Index | `rsi` | `1.0.0` | `active` | `close` |
| `stoch` | Stochastic Oscillator | `STOCHk, STOCHd` | `1.0.0` | `active` | `high,low,close` |
| `stochf` | Stochastic Fast | `STOCHFk, STOCHFd` | `1.0.0` | `active` | `high,low,close` |
| `stochrsi` | Stochastic RSI | `STOCHRSIk, STOCHRSId` | `1.0.0` | `active` | `close` |
| `trix` | TRIX | `trix` | `1.0.0` | `active` | `close` |
| `ultosc` | Ultimate Oscillator | `ultosc` | `1.0.0` | `active` | `high,low,close` |
| `willr` | Williams %R | `willr` | `1.0.0` | `active` | `high,low,close` |

## volatility (16)

| indicator | full_name | outputs | version | status | required_columns |
|---|---|---|---|---|---|
| `atr` | Average True Range | `atr` | `1.0.0` | `active` | `high,low,close` |
| `atr_regime_ratio` | ATR Regime Ratio | `atr_regime_ratio` | `1.0.0` | `active` | `high,low,close` |
| `atrts` | ATR Trailing Stop | `atrts` | `1.0.0` | `active` | `high,low,close` |
| `chandelier` | Chandelier Exit | `CHANDELIERl, CHANDELIERs` | `1.0.0` | `active` | `high,low,close` |
| `garman_klass_volatility` | Garman-Klass Volatility | `garman_klass_volatility` | `1.0.0` | `active` | `open,high,low,close` |
| `massi` | Mass Index | `massi` | `1.0.0` | `active` | `high,low` |
| `natr` | Normalized Average True Range | `natr` | `1.0.0` | `active` | `high,low,close` |
| `parkinson_volatility` | Parkinson Volatility | `parkinson_volatility` | `1.0.0` | `active` | `high,low` |
| `pdist` | Price Distance | `pdist` | `1.0.0` | `active` | `open,high,low,close` |
| `rvi` | Relative Volatility Index | `rvi` | `1.0.0` | `active` | `close` |
| `squeeze_score` | Squeeze Score | `squeeze_score` | `1.0.0` | `active` | `high,low,close` |
| `stdev` | Standard Deviation | `stdev` | `1.0.0` | `active` | `close` |
| `trange` | True Range | `trange` | `1.0.0` | `active` | `high,low,close` |
| `ui` | Ulcer Index | `ui` | `1.0.0` | `active` | `close` |
| `variance` | Variance | `variance` | `1.0.0` | `active` | `close` |
| `volatility_regime_ratio` | Volatility Regime Ratio | `volatility_regime_ratio` | `1.0.0` | `active` | `close` |

## volume (14)

| indicator | full_name | outputs | version | status | required_columns |
|---|---|---|---|---|---|
| `ad` | Accumulation/Distribution Line | `ad` | `1.0.0` | `active` | `high,low,close,volume` |
| `adosc` | Chaikin A/D Oscillator | `adosc` | `1.0.0` | `active` | `high,low,close,volume` |
| `cmf` | Chaikin Money Flow | `cmf` | `1.0.0` | `active` | `high,low,close,volume` |
| `dry_up_reversal_hint` | Dry-Up Reversal Hint | `dry_up_reversal_hint` | `1.0.0` | `active` | `close,volume` |
| `efi` | Elder's Force Index | `efi` | `1.0.0` | `active` | `close,volume` |
| `eom` | Ease of Movement | `eom` | `1.0.0` | `active` | `high,low,close,volume` |
| `kvo` | Klinger Volume Oscillator | `KVO, KVOs` | `1.0.0` | `active` | `high,low,close,volume` |
| `mfi` | Money Flow Index | `mfi` | `1.0.0` | `active` | `high,low,close,volume` |
| `nvi` | Negative Volume Index | `nvi` | `1.0.0` | `active` | `close,volume` |
| `obv` | On Balance Volume | `obv` | `1.0.0` | `active` | `close,volume` |
| `pvi` | Positive Volume Index | `pvi` | `1.0.0` | `active` | `close,volume` |
| `pvo` | Percentage Volume Oscillator | `PVO, PVOs, PVOh` | `1.0.0` | `active` | `volume` |
| `pvt` | Price-Volume Trend | `pvt` | `1.0.0` | `active` | `close,volume` |
| `vwma` | Volume Weighted Moving Average | `vwma` | `1.0.0` | `active` | `close,volume` |

## utils (23)

| indicator | full_name | outputs | version | status | required_columns |
|---|---|---|---|---|---|
| `avgprice` | Average Price | `avgprice` | `1.0.0` | `active` | `open,high,low,close` |
| `breakout_frequency` | Breakout Frequency | `breakout_frequency` | `1.0.0` | `active` | `high,low,close` |
| `directional_persistence` | Directional Persistence | `directional_persistence` | `1.0.0` | `active` | `close` |
| `drawdown` | Drawdown | `DD, DDp, DDmax` | `1.0.0` | `active` | `close` |
| `efficiency_ratio` | Efficiency Ratio | `efficiency_ratio` | `1.0.0` | `active` | `close` |
| `entropy` | Entropy | `entropy` | `1.0.0` | `active` | `close` |
| `false_breakout_frequency` | False Breakout Frequency | `false_breakout_frequency` | `1.0.0` | `active` | `high,low,close` |
| `kurtosis` | Kurtosis | `kurtosis` | `1.0.0` | `active` | `close` |
| `linearreg_angle` | Linear Regression Angle | `angle` | `1.0.0` | `active` | `close` |
| `linearreg_intercept` | Linear Regression Intercept | `intercept` | `1.0.0` | `active` | `close` |
| `linearreg_slope` | Linear Regression Slope | `slope` | `1.0.0` | `active` | `close` |
| `log_return` | Log Return | `log_return` | `1.0.0` | `active` | `close` |
| `max` | Highest Value over Period | `max` | `1.0.0` | `active` | `close` |
| `median` | Rolling Median | `median` | `1.0.0` | `active` | `close` |
| `medprice` | Median Price | `medprice` | `1.0.0` | `active` | `high,low` |
| `min` | Lowest Value over Period | `min` | `1.0.0` | `active` | `close` |
| `percent_return` | Percent Return | `percent_return` | `1.0.0` | `active` | `close` |
| `quantile` | Rolling Quantile | `quantile` | `1.0.0` | `active` | `close` |
| `sign_autocorrelation` | Sign Autocorrelation | `sign_autocorrelation` | `1.0.0` | `active` | `close` |
| `skew` | Skewness | `skew` | `1.0.0` | `active` | `close` |
| `sum` | Summation | `sum` | `1.0.0` | `active` | `close` |
| `tsf` | Time Series Forecast | `tsf` | `1.0.0` | `active` | `close` |
| `zscore` | Z-Score | `zscore` | `1.0.0` | `active` | `close` |

## 生成命令

```bash
PYTHONPATH=. uv run python scripts/generate_indicator_docs.py
PYTHONPATH=. uv run python scripts/generate_indicator_docs.py --check
```
