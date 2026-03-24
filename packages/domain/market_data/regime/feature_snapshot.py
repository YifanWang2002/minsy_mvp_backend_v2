"""Feature snapshot extraction for pre-strategy regime diagnosis."""

from __future__ import annotations

import math
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from packages.domain.strategy.feature.contracts import resolve_multi_output_assignments
from packages.domain.strategy.feature.indicators import IndicatorRegistry, IndicatorWrapper

_EPS = 1e-12
_FEATURE_ENGINE = IndicatorWrapper()
_FEATURE_ENGINE_CACHE_MAX_ENTRIES = max(
    32,
    int(os.getenv("PRE_STRATEGY_FEATURE_CACHE_MAX_ENTRIES", "512").strip() or 512),
)
_FEATURE_ENGINE_CACHE_TTL_SECONDS = max(
    1.0,
    float(os.getenv("PRE_STRATEGY_FEATURE_CACHE_TTL_SECONDS", "300").strip() or 300.0),
)


@dataclass(slots=True)
class _FeatureEngineCacheEntry:
    created_at_epoch: float
    series: pd.Series


_FEATURE_ENGINE_CACHE: OrderedDict[str, _FeatureEngineCacheEntry] = OrderedDict()
_FEATURE_ENGINE_CACHE_HITS = 0
_FEATURE_ENGINE_CACHE_MISSES = 0


def _fingerprint_frame_for_cache(data: pd.DataFrame) -> tuple[Any, ...]:
    if data.empty:
        return (0, "", "", 0.0, 0.0, 0.0, 0.0)
    tail = data.tail(min(5, len(data)))
    return (
        int(len(data)),
        str(data.index[0]),
        str(data.index[-1]),
        _safe_float(data["close"].iloc[0]),
        _safe_float(data["close"].iloc[-1]),
        _safe_float(tail["high"].sum()),
        _safe_float(tail["low"].sum()),
        _safe_float(tail["volume"].sum()),
    )


def _feature_engine_cache_key(
    *,
    data: pd.DataFrame,
    indicator: str,
    source: str,
    params: dict[str, Any],
    requested_output: str | None,
) -> str:
    items = tuple(sorted((str(key), repr(value)) for key, value in params.items()))
    frame_fingerprint = _fingerprint_frame_for_cache(data)
    return repr(
        (
            indicator,
            source,
            requested_output or "",
            items,
            frame_fingerprint,
        )
    )


def _feature_engine_cache_get(key: str) -> pd.Series | None:
    global _FEATURE_ENGINE_CACHE_HITS, _FEATURE_ENGINE_CACHE_MISSES
    entry = _FEATURE_ENGINE_CACHE.get(key)
    now = time.time()
    if entry is None:
        _FEATURE_ENGINE_CACHE_MISSES += 1
        return None
    if (now - float(entry.created_at_epoch)) > _FEATURE_ENGINE_CACHE_TTL_SECONDS:
        _FEATURE_ENGINE_CACHE.pop(key, None)
        _FEATURE_ENGINE_CACHE_MISSES += 1
        return None
    _FEATURE_ENGINE_CACHE.move_to_end(key)
    _FEATURE_ENGINE_CACHE_HITS += 1
    return entry.series.copy()


def _feature_engine_cache_set(key: str, series: pd.Series) -> None:
    _FEATURE_ENGINE_CACHE[key] = _FeatureEngineCacheEntry(
        created_at_epoch=time.time(),
        series=series.copy(),
    )
    _FEATURE_ENGINE_CACHE.move_to_end(key)
    while len(_FEATURE_ENGINE_CACHE) > _FEATURE_ENGINE_CACHE_MAX_ENTRIES:
        _FEATURE_ENGINE_CACHE.popitem(last=False)


def reset_feature_engine_cache() -> None:
    _FEATURE_ENGINE_CACHE.clear()
    global _FEATURE_ENGINE_CACHE_HITS, _FEATURE_ENGINE_CACHE_MISSES
    _FEATURE_ENGINE_CACHE_HITS = 0
    _FEATURE_ENGINE_CACHE_MISSES = 0


def get_feature_engine_cache_stats() -> dict[str, Any]:
    return {
        "entries": int(len(_FEATURE_ENGINE_CACHE)),
        "hits": int(_FEATURE_ENGINE_CACHE_HITS),
        "misses": int(_FEATURE_ENGINE_CACHE_MISSES),
        "ttl_seconds": float(_FEATURE_ENGINE_CACHE_TTL_SECONDS),
        "max_entries": int(_FEATURE_ENGINE_CACHE_MAX_ENTRIES),
    }


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(numeric) or math.isinf(numeric):
        return default
    return numeric


def _safe_bool(value: Any) -> bool:
    return bool(value)


def _series_tail(series: pd.Series, default: float = 0.0) -> float:
    if series.empty:
        return default
    return _safe_float(series.iloc[-1], default=default)


def _rolling_percentile(series: pd.Series, window: int) -> float:
    if len(series) < 5:
        return 0.5
    bounded = series.dropna().tail(max(window, 10))
    if bounded.empty:
        return 0.5
    latest = bounded.iloc[-1]
    rank = float((bounded <= latest).sum())
    return rank / float(len(bounded))


def _compute_true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.fillna(0.0)


def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = _compute_true_range(df)
    return tr.rolling(period, min_periods=max(2, period // 3)).mean()


def _compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = _compute_true_range(df)
    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean().replace(0.0, np.nan)
    plus_di = 100.0 * pd.Series(plus_dm, index=df.index).ewm(
        alpha=1.0 / period,
        adjust=False,
    ).mean() / atr
    minus_di = 100.0 * pd.Series(minus_dm, index=df.index).ewm(
        alpha=1.0 / period,
        adjust=False,
    ).mean() / atr
    dx = (100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)).fillna(0.0)
    adx = dx.ewm(alpha=1.0 / period, adjust=False).mean()
    return adx.fillna(0.0)


def _compute_chop(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = _compute_true_range(df)
    tr_sum = tr.rolling(period, min_periods=max(2, period // 3)).sum()
    high_n = df["high"].rolling(period, min_periods=max(2, period // 3)).max()
    low_n = df["low"].rolling(period, min_periods=max(2, period // 3)).min()
    denom = (high_n - low_n).replace(0.0, np.nan)
    chop = 100.0 * np.log10((tr_sum / denom).replace(0.0, np.nan)) / np.log10(float(period))
    return chop.fillna(50.0).clip(lower=0.0, upper=100.0)


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0).clip(lower=0.0, upper=100.0)


def _compute_cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    ma = tp.rolling(period, min_periods=max(2, period // 3)).mean()
    mad = tp.rolling(period, min_periods=max(2, period // 3)).apply(
        lambda values: float(np.mean(np.abs(values - np.mean(values)))),
        raw=True,
    )
    denom = (0.015 * mad).replace(0.0, np.nan)
    cci = (tp - ma) / denom
    return cci.fillna(0.0)


def _compute_stoch(df: pd.DataFrame, period: int = 14) -> tuple[pd.Series, pd.Series]:
    low_n = df["low"].rolling(period, min_periods=max(2, period // 3)).min()
    high_n = df["high"].rolling(period, min_periods=max(2, period // 3)).max()
    k = 100.0 * (df["close"] - low_n) / (high_n - low_n).replace(0.0, np.nan)
    d = k.rolling(3, min_periods=1).mean()
    return k.fillna(50.0), d.fillna(50.0)


def _compute_aroon(df: pd.DataFrame, period: int = 14) -> tuple[pd.Series, pd.Series]:
    def _idx_max(values: np.ndarray) -> float:
        return float(np.argmax(values))

    def _idx_min(values: np.ndarray) -> float:
        return float(np.argmin(values))

    high_pos = df["high"].rolling(period + 1, min_periods=max(2, period // 3)).apply(
        _idx_max,
        raw=True,
    )
    low_pos = df["low"].rolling(period + 1, min_periods=max(2, period // 3)).apply(
        _idx_min,
        raw=True,
    )
    aroon_up = 100.0 * high_pos / float(period)
    aroon_down = 100.0 * low_pos / float(period)
    return aroon_up.fillna(50.0), aroon_down.fillna(50.0)


def _compute_macd_hist(close: pd.Series) -> pd.Series:
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=9, adjust=False).mean()
    return (macd - signal).fillna(0.0)


def _compute_supertrend_state(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.Series:
    atr = _compute_atr(df, period=period).bfill().fillna(0.0)
    hl2 = (df["high"] + df["low"]) / 2.0
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr

    trend = pd.Series(index=df.index, dtype=float)
    trend.iloc[0] = 1.0
    final_upper = upper.copy()
    final_lower = lower.copy()

    for i in range(1, len(df)):
        prev_i = i - 1
        if upper.iloc[i] < final_upper.iloc[prev_i] or df["close"].iloc[prev_i] > final_upper.iloc[prev_i]:
            final_upper.iloc[i] = upper.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[prev_i]

        if lower.iloc[i] > final_lower.iloc[prev_i] or df["close"].iloc[prev_i] < final_lower.iloc[prev_i]:
            final_lower.iloc[i] = lower.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[prev_i]

        if trend.iloc[prev_i] < 0 and df["close"].iloc[i] > final_upper.iloc[prev_i]:
            trend.iloc[i] = 1.0
        elif trend.iloc[prev_i] > 0 and df["close"].iloc[i] < final_lower.iloc[prev_i]:
            trend.iloc[i] = -1.0
        else:
            trend.iloc[i] = trend.iloc[prev_i]

    return trend.fillna(1.0)


def _compute_pivots(
    df: pd.DataFrame,
    *,
    pivot_window: int,
) -> tuple[pd.Series, pd.Series]:
    highs = df["high"]
    lows = df["low"]
    max_n = highs.rolling((2 * pivot_window) + 1, center=True).max()
    min_n = lows.rolling((2 * pivot_window) + 1, center=True).min()
    pivot_high = highs.where((highs == max_n) & max_n.notna())
    pivot_low = lows.where((lows == min_n) & min_n.notna())
    return pivot_high.dropna(), pivot_low.dropna()


def _run_ratio(sequence: pd.Series, *, comparator: str) -> float:
    if len(sequence) < 2:
        return 0.0
    deltas = sequence.diff().dropna()
    if deltas.empty:
        return 0.0
    if comparator == "gt":
        return float((deltas > 0).mean())
    return float((deltas < 0).mean())


def _max_drawdown(close: pd.Series) -> float:
    if close.empty:
        return 0.0
    running_max = close.cummax().replace(0.0, np.nan)
    drawdown = (close / running_max) - 1.0
    return _safe_float(drawdown.min(), default=0.0)


def _estimate_hurst_proxy(close: pd.Series) -> float:
    values = close.dropna().values
    if len(values) < 64:
        return 0.5
    lags = np.array([2, 4, 8, 16], dtype=float)
    tau: list[float] = []
    for lag in lags:
        diff = np.subtract(values[int(lag):], values[:-int(lag)])
        if len(diff) == 0:
            continue
        tau.append(float(np.std(diff)))
    if len(tau) < 2:
        return 0.5
    slope, _ = np.polyfit(np.log(lags[: len(tau)]), np.log(np.array(tau) + _EPS), 1)
    return _clip(float(slope), 0.0, 1.0)


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _serialize_float_map(raw: dict[str, Any]) -> dict[str, float]:
    return {key: _safe_float(value) for key, value in raw.items()}


def _serialize_bool_map(raw: dict[str, Any]) -> dict[str, bool]:
    return {key: _safe_bool(value) for key, value in raw.items()}


_ATOMIC_REGIME_FACTOR_SPECS: dict[str, dict[str, Any]] = {
    "efficiency_ratio": {
        "indicator": "efficiency_ratio",
        "params": {"length": 50},
        "source": "close",
        "group": "efficiency_noise",
    },
    "directional_persistence": {
        "indicator": "directional_persistence",
        "params": {"length": 50},
        "source": "close",
        "group": "efficiency_noise",
    },
    "sign_autocorrelation": {
        "indicator": "sign_autocorrelation",
        "params": {"length": 50},
        "source": "close",
        "group": "efficiency_noise",
    },
    "breakout_frequency": {
        "indicator": "breakout_frequency",
        "params": {"length": 20},
        "group": "swing_structure",
    },
    "false_breakout_frequency": {
        "indicator": "false_breakout_frequency",
        "params": {"length": 20, "confirm_bars": 5},
        "group": "swing_structure",
    },
    "volatility_regime_ratio": {
        "indicator": "volatility_regime_ratio",
        "params": {"short_window": 20, "long_window": 60},
        "source": "close",
        "group": "volatility_state",
    },
    "atr_regime_ratio": {
        "indicator": "atr_regime_ratio",
        "params": {"atr_period": 14, "short_window": 14, "long_window": 50},
        "group": "volatility_state",
    },
    "squeeze_score": {
        "indicator": "squeeze_score",
        "params": {"length": 20, "bb_mult": 2.0, "kc_mult": 1.0, "percentile_window": 120},
        "group": "volatility_state",
    },
    "parkinson_volatility": {
        "indicator": "parkinson_volatility",
        "params": {"length": 120},
        "group": "volatility_level",
    },
    "garman_klass_volatility": {
        "indicator": "garman_klass_volatility",
        "params": {"length": 120},
        "group": "volatility_level",
    },
    "dry_up_reversal_hint": {
        "indicator": "dry_up_reversal_hint",
        "params": {"length": 120, "quantile": 0.2, "reversal_bars": 1},
        "source": "close",
        "group": "volume_participation",
    },
}


def _atomic_factor_policy_payload() -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for key, spec in _ATOMIC_REGIME_FACTOR_SPECS.items():
        indicator = str(spec.get("indicator", key)).strip().lower()
        metadata = IndicatorRegistry.get(indicator)
        payload.append(
            {
                "snapshot_field": key,
                "group": str(spec.get("group", "")).strip(),
                "indicator": indicator,
                "params": dict(spec.get("params", {})),
                "source": str(spec.get("source", "close")).strip().lower() or "close",
                "version": str(getattr(metadata, "version", "1.0.0")) if metadata else "1.0.0",
                "status": str(getattr(metadata, "status", "active")) if metadata else "active",
                "deprecated_since": getattr(metadata, "deprecated_since", None) if metadata else None,
                "replacement": getattr(metadata, "replacement", None) if metadata else None,
                "remove_after": getattr(metadata, "remove_after", None) if metadata else None,
            }
        )
    payload.sort(key=lambda item: str(item.get("snapshot_field", "")))
    return payload


def _compute_with_feature_engine(
    data: pd.DataFrame,
    *,
    factors: dict[str, dict[str, Any]],
    use_cache: bool = True,
) -> dict[str, pd.Series]:
    """Compute atomic regime factors through the shared indicator engine."""
    computed: dict[str, pd.Series] = {}
    for factor_name, spec in factors.items():
        indicator = str(spec.get("indicator", factor_name)).strip().lower()
        metadata = IndicatorRegistry.get(indicator)
        if metadata is None:
            raise ValueError(f"Indicator not registered for atomic factor '{factor_name}': {indicator}")
        params = spec.get("params", {})
        source = str(spec.get("source", "close")).strip().lower() or "close"
        requested_output = spec.get("output")
        cache_key = _feature_engine_cache_key(
            data=data,
            indicator=indicator,
            source=source,
            params=dict(params if isinstance(params, dict) else {}),
            requested_output=str(requested_output).strip() if isinstance(requested_output, str) else None,
        )
        if use_cache:
            cached = _feature_engine_cache_get(cache_key)
            if cached is not None:
                computed[factor_name] = cached
                continue
        raw = _FEATURE_ENGINE.calculate(
            indicator,
            data,
            source=source,
            **(params if isinstance(params, dict) else {}),
        )

        if isinstance(raw, pd.Series):
            numeric = pd.to_numeric(raw, errors="coerce")
            if numeric.empty:
                raise ValueError(f"Atomic factor '{factor_name}' returned empty series.")
            if use_cache:
                _feature_engine_cache_set(cache_key, numeric)
            computed[factor_name] = numeric
            continue

        if raw.empty:
            raise ValueError(f"Atomic factor '{factor_name}' returned empty dataframe.")

        if isinstance(requested_output, str) and requested_output.strip():
            assignments = resolve_multi_output_assignments(
                indicator=indicator,
                requested_outputs=(requested_output.strip(),),
                result_columns=list(raw.columns),
            )
            if assignments:
                numeric = pd.to_numeric(raw[assignments[0][0]], errors="coerce")
                if use_cache:
                    _feature_engine_cache_set(cache_key, numeric)
                computed[factor_name] = numeric
                continue

        numeric = pd.to_numeric(raw.iloc[:, 0], errors="coerce")
        if use_cache:
            _feature_engine_cache_set(cache_key, numeric)
        computed[factor_name] = numeric
    return computed


def build_regime_feature_snapshot(
    data: pd.DataFrame,
    *,
    timeframe: str,
    lookback_bars: int,
    pivot_window: int = 5,
) -> dict[str, Any]:
    """Compute grouped regime features for one OHLCV window."""

    if data.empty:
        raise ValueError("Cannot build regime snapshot from empty data.")

    required_cols = {"open", "high", "low", "close", "volume"}
    missing = required_cols.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = data.copy()
    df = df.sort_index()
    if len(df) > lookback_bars:
        df = df.tail(lookback_bars)
    returns = df["close"].pct_change().fillna(0.0)
    log_returns = np.log(df["close"].replace(0.0, np.nan)).diff().fillna(0.0)
    atr = _compute_atr(df, period=14).fillna(0.0)
    realized_vol = returns.rolling(20, min_periods=5).std().fillna(0.0)
    short_vol = returns.rolling(20, min_periods=5).std().fillna(0.0)
    long_vol = returns.rolling(60, min_periods=10).std().bfill().fillna(0.0)
    short_atr = atr.rolling(14, min_periods=5).mean().fillna(0.0)
    long_atr = atr.rolling(50, min_periods=10).mean().bfill().fillna(0.0)

    up_mask = returns > 0
    down_mask = returns < 0
    bar_range = (df["high"] - df["low"]).replace(0.0, np.nan)
    body = (df["close"] - df["open"]).abs()
    wick = (df["high"] - df["low"] - body).clip(lower=0.0)
    doji_mask = (body / (bar_range + _EPS)) <= 0.1

    consecutive_up = (
        returns.gt(0).astype(int).groupby((returns.le(0)).cumsum()).cumsum().max()
    )
    consecutive_down = (
        returns.lt(0).astype(int).groupby((returns.ge(0)).cumsum()).cumsum().max()
    )

    pivot_high, pivot_low = _compute_pivots(df, pivot_window=max(2, int(pivot_window)))
    hh_ratio = _run_ratio(pivot_high, comparator="gt")
    lh_ratio = _run_ratio(pivot_high, comparator="lt")
    hl_ratio = _run_ratio(pivot_low, comparator="gt")
    ll_ratio = _run_ratio(pivot_low, comparator="lt")

    pivots = pd.concat([pivot_high, pivot_low]).sort_index()
    if len(pivots) >= 2:
        x = np.arange(len(pivots))
        pivot_slope = _safe_float(np.polyfit(x, pivots.values, 1)[0], default=0.0)
    else:
        pivot_slope = 0.0

    breakout_high = df["close"] > df["high"].rolling(20, min_periods=5).max().shift(1)
    breakout_low = df["close"] < df["low"].rolling(20, min_periods=5).min().shift(1)
    breakout_mask = (breakout_high | breakout_low).fillna(False)

    recent_high = _safe_float(df["high"].rolling(50, min_periods=5).max().iloc[-1], default=df["high"].iloc[-1])
    recent_low = _safe_float(df["low"].rolling(50, min_periods=5).min().iloc[-1], default=df["low"].iloc[-1])
    recent_mid = (recent_high + recent_low) / 2.0
    close_latest = _safe_float(df["close"].iloc[-1], default=0.0)
    range_span = max(recent_high - recent_low, _EPS)
    distance_to_high = (recent_high - close_latest) / range_span
    distance_to_low = (close_latest - recent_low) / range_span
    distance_to_mid = (close_latest - recent_mid) / range_span

    choppiness = _series_tail(_compute_chop(df, period=14), default=50.0)
    zigzag_turn_density = _safe_float(len(pivots) / max(len(df), 1), default=0.0)
    hurst_proxy = _estimate_hurst_proxy(df["close"])

    bb_mid = df["close"].rolling(20, min_periods=5).mean()
    bb_std = df["close"].rolling(20, min_periods=5).std().fillna(0.0)
    bb_upper = bb_mid + (2.0 * bb_std)
    bb_lower = bb_mid - (2.0 * bb_std)
    bb_width = (bb_upper - bb_lower) / bb_mid.replace(0.0, np.nan)
    bb_position = (df["close"] - bb_lower) / (bb_upper - bb_lower).replace(0.0, np.nan)

    keltner_mid = df["close"].rolling(20, min_periods=5).mean()
    keltner_width = (2.0 * atr) / keltner_mid.replace(0.0, np.nan)
    bb_width_percentile = _rolling_percentile(bb_width, window=120)
    kc_width_percentile = _rolling_percentile(keltner_width, window=120)

    short_vol_latest = _series_tail(short_vol, default=0.0)
    long_vol_latest = max(_series_tail(long_vol, default=0.0), _EPS)

    adx = _series_tail(_compute_adx(df, period=14), default=20.0)
    aroon_up, aroon_down = _compute_aroon(df, period=14)
    sma20 = df["close"].rolling(20, min_periods=5).mean()
    ma_slope = _safe_float((sma20.iloc[-1] - sma20.iloc[max(0, len(sma20) - 6)]) / max(abs(close_latest), _EPS), default=0.0)

    lr_window = min(50, len(df))
    if lr_window >= 2:
        x = np.arange(lr_window)
        lr_slope = _safe_float(np.polyfit(x, df["close"].tail(lr_window).values, 1)[0], default=0.0)
    else:
        lr_slope = 0.0
    lr_angle = math.degrees(math.atan(lr_slope))

    cumulative_volume = df["volume"].cumsum().replace(0.0, np.nan)
    vwap = (((df["high"] + df["low"] + df["close"]) / 3.0) * df["volume"]).cumsum() / cumulative_volume
    vwap_latest = _series_tail(vwap, default=close_latest)
    dist_ma = (close_latest - _series_tail(sma20, default=close_latest)) / max(abs(_series_tail(sma20, default=close_latest)), _EPS)
    dist_vwap = (close_latest - vwap_latest) / max(abs(vwap_latest), _EPS)

    donchian_high = _safe_float(df["high"].rolling(20, min_periods=5).max().iloc[-1], default=recent_high)
    donchian_low = _safe_float(df["low"].rolling(20, min_periods=5).min().iloc[-1], default=recent_low)
    donchian_position = (close_latest - donchian_low) / max(donchian_high - donchian_low, _EPS)

    macd_hist = _compute_macd_hist(df["close"])
    macd_hist_latest = _series_tail(macd_hist, default=0.0)
    macd_slope = _safe_float(macd_hist.diff().tail(5).mean(), default=0.0)
    supertrend_state = _series_tail(_compute_supertrend_state(df, period=10, multiplier=3.0), default=1.0)

    rsi = _series_tail(_compute_rsi(df["close"], period=14), default=50.0)
    cci = _series_tail(_compute_cci(df, period=20), default=0.0)
    stoch_k, stoch_d = _compute_stoch(df, period=14)
    zscore_window = min(50, len(df))
    zscore_mean = _safe_float(df["close"].tail(zscore_window).mean(), default=close_latest)
    zscore_std = max(_safe_float(df["close"].tail(zscore_window).std(), default=0.0), _EPS)
    price_zscore = (close_latest - zscore_mean) / zscore_std
    range_midpoint = (recent_high + recent_low) / 2.0
    percentile_rank = _rolling_percentile(df["close"], window=120)

    volume_window = df["volume"].tail(min(len(df), 120))
    volume_reliable = _safe_bool(
        len(volume_window) >= 20 and volume_window.notna().all() and volume_window.sum() > 0
    )
    relative_volume = _safe_float(
        df["volume"].iloc[-1]
        / max(
            _safe_float(
                df["volume"].rolling(20, min_periods=5).mean().iloc[-1],
                default=0.0,
            ),
            _EPS,
        ),
        default=1.0,
    )
    obv = (np.sign(returns).fillna(0.0) * df["volume"]).cumsum()
    obv_slope = _safe_float(obv.diff().tail(10).mean(), default=0.0)

    clv = (
        ((df["close"] - df["low"]) - (df["high"] - df["close"]))
        / (df["high"] - df["low"]).replace(0.0, np.nan)
    ).fillna(0.0)
    cmf = (
        (clv * df["volume"]).rolling(20, min_periods=5).sum()
        / df["volume"].rolling(20, min_periods=5).sum().replace(0.0, np.nan)
    ).fillna(0.0)

    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    raw_money_flow = typical * df["volume"]
    positive_mf = raw_money_flow.where(typical > typical.shift(1), 0.0)
    negative_mf = raw_money_flow.where(typical < typical.shift(1), 0.0).abs()
    mfi = 100.0 - (
        100.0
        / (
            1.0
            + (
                positive_mf.rolling(14, min_periods=5).sum()
                / negative_mf.rolling(14, min_periods=5).sum().replace(0.0, np.nan)
            )
        )
    )
    pvt = (returns.fillna(0.0) * df["volume"]).cumsum()
    efi = ((df["close"].diff().fillna(0.0)) * df["volume"]).ewm(span=13, adjust=False).mean()

    breakout_vol = _safe_float(df.loc[breakout_mask, "volume"].mean(), default=0.0)
    normal_vol = _safe_float(df["volume"].rolling(20, min_periods=5).mean().iloc[-1], default=0.0)
    breakout_volume_expansion = breakout_vol / max(normal_vol, _EPS)

    engine_window = max(20, len(df))
    atomic_specs: dict[str, dict[str, Any]] = {}
    for key, spec in _ATOMIC_REGIME_FACTOR_SPECS.items():
        copied = dict(spec)
        copied["params"] = dict(spec.get("params", {}))
        atomic_specs[key] = copied
    atomic_specs["breakout_frequency"]["params"]["window"] = engine_window
    atomic_specs["false_breakout_frequency"]["params"]["window"] = engine_window
    atomic_specs["parkinson_volatility"]["params"]["length"] = engine_window
    atomic_specs["garman_klass_volatility"]["params"]["length"] = engine_window
    atomic_specs["dry_up_reversal_hint"]["params"]["length"] = engine_window

    engine_factor_series = _compute_with_feature_engine(
        df,
        factors=atomic_specs,
    )
    efficiency_ratio = _series_tail(
        engine_factor_series["efficiency_ratio"],
        default=0.0,
    )
    directional_persistence = _series_tail(
        engine_factor_series["directional_persistence"],
        default=0.0,
    )
    sign_autocorr = _series_tail(
        engine_factor_series["sign_autocorrelation"],
        default=0.0,
    )
    breakout_frequency = _series_tail(
        engine_factor_series["breakout_frequency"],
        default=0.0,
    )
    false_breakout_frequency = _series_tail(
        engine_factor_series["false_breakout_frequency"],
        default=0.0,
    )
    vol_short_long_ratio = _series_tail(
        engine_factor_series["volatility_regime_ratio"],
        default=0.0,
    )
    atr_short_long_ratio = _series_tail(
        engine_factor_series["atr_regime_ratio"],
        default=0.0,
    )
    squeeze_score = _series_tail(
        engine_factor_series["squeeze_score"],
        default=0.0,
    )
    parkinson_volatility = _series_tail(
        engine_factor_series["parkinson_volatility"],
        default=0.0,
    )
    garman_klass_volatility = _series_tail(
        engine_factor_series["garman_klass_volatility"],
        default=0.0,
    )
    dry_up_reversal_hint = _series_tail(
        engine_factor_series["dry_up_reversal_hint"],
        default=0.0,
    )
    vol_change_rate = vol_short_long_ratio - 1.0

    atr_percentile = _rolling_percentile(atr, 120)
    vol_percentile = _rolling_percentile(short_vol, 120)
    volatility_coupling = _serialize_bool_map(
        {
            "trend_with_low_vol": adx >= 25.0 and vol_short_long_ratio <= 1.0,
            "trend_with_expanding_vol": adx >= 25.0 and vol_short_long_ratio > 1.1,
            "range_with_low_vol": adx < 20.0 and vol_short_long_ratio <= 1.0,
            "panic_reversal_with_high_vol": vol_percentile >= 0.85 and abs(price_zscore) >= 1.5,
        }
    )

    extreme_q = returns.quantile([0.01, 0.05, 0.95, 0.99]).to_dict()

    snapshot: dict[str, Any] = {
        "window_stats": _serialize_float_map(
            {
                "cumulative_return": (close_latest / max(_safe_float(df["close"].iloc[0], default=close_latest), _EPS)) - 1.0,
                "recent_return": _safe_float(df["close"].pct_change(20).iloc[-1], default=0.0),
                "rolling_return_mean": _safe_float(returns.rolling(20, min_periods=5).mean().iloc[-1], default=0.0),
                "rolling_return_std": _safe_float(returns.rolling(20, min_periods=5).std().iloc[-1], default=0.0),
                "return_skew": _safe_float(returns.tail(120).skew(), default=0.0),
                "return_kurtosis": _safe_float(returns.tail(120).kurt(), default=0.0),
                "volume_percentile": _rolling_percentile(df["volume"], 120),
            }
        ),
        "price_path_summary": _serialize_float_map(
            {
                "max_drawdown": _max_drawdown(df["close"]),
                "up_bar_ratio": _safe_float(up_mask.mean(), default=0.0),
                "down_bar_ratio": _safe_float(down_mask.mean(), default=0.0),
                "doji_ratio": _safe_float(doji_mask.mean(), default=0.0),
                "avg_body_size": _safe_float(body.mean(), default=0.0),
                "avg_wick_size": _safe_float(wick.mean(), default=0.0),
                "gap_frequency": _safe_float((df["open"] - df["close"].shift(1)).abs().div(df["close"].shift(1).abs() + _EPS).gt(0.002).mean(), default=0.0),
                "close_location_value": _safe_float(clv.tail(20).mean(), default=0.0),
                "max_consecutive_up": _safe_float(consecutive_up, default=0.0),
                "max_consecutive_down": _safe_float(consecutive_down, default=0.0),
            }
        ),
        "swing_structure": _serialize_float_map(
            {
                "higher_high_ratio": hh_ratio,
                "higher_low_ratio": hl_ratio,
                "lower_high_ratio": lh_ratio,
                "lower_low_ratio": ll_ratio,
                "pivot_slope": pivot_slope,
                "breakout_frequency": breakout_frequency,
                "false_breakout_frequency": false_breakout_frequency,
                "distance_to_recent_high": distance_to_high,
                "distance_to_recent_low": distance_to_low,
                "distance_to_recent_midrange": distance_to_mid,
            }
        ),
        "efficiency_noise": _serialize_float_map(
            {
                "efficiency_ratio": efficiency_ratio,
                "hurst_proxy": hurst_proxy,
                "directional_persistence": directional_persistence,
                "sign_autocorrelation": sign_autocorr,
                "choppiness_index": choppiness,
                "zigzag_turn_density": zigzag_turn_density,
            }
        ),
        "volatility_level": _serialize_float_map(
            {
                "rolling_std_returns": short_vol_latest,
                "atr": _series_tail(atr, default=0.0),
                "natr": _series_tail(atr / df["close"].replace(0.0, np.nan), default=0.0),
                "realized_volatility": _series_tail(realized_vol, default=0.0),
                "parkinson_volatility": parkinson_volatility,
                "garman_klass_volatility": garman_klass_volatility,
                "upside_volatility": _safe_float(returns.where(up_mask).std(), default=0.0),
                "downside_volatility": _safe_float(returns.where(down_mask).std(), default=0.0),
                "atr_percentile": atr_percentile,
                "extreme_return_q01": _safe_float(extreme_q.get(0.01), default=0.0),
                "extreme_return_q05": _safe_float(extreme_q.get(0.05), default=0.0),
                "extreme_return_q95": _safe_float(extreme_q.get(0.95), default=0.0),
                "extreme_return_q99": _safe_float(extreme_q.get(0.99), default=0.0),
            }
        ),
        "volatility_state": _serialize_float_map(
            {
                "vol_percentile": vol_percentile,
                "short_long_vol_ratio": vol_short_long_ratio,
                "atr_short_long_ratio": atr_short_long_ratio,
                "squeeze_score": squeeze_score,
                "bb_width_percentile": bb_width_percentile,
                "kc_width_percentile": kc_width_percentile,
                "volatility_change_rate": vol_change_rate,
                "volatility_expansion_flag": 1.0 if vol_short_long_ratio > 1.1 else 0.0,
                "volatility_contraction_flag": 1.0 if vol_short_long_ratio < 0.9 else 0.0,
            }
        ),
        "volatility_direction_coupling": volatility_coupling,
        "trend_reversion": _serialize_float_map(
            {
                "adx": adx,
                "aroon_up": _series_tail(aroon_up, default=50.0),
                "aroon_down": _series_tail(aroon_down, default=50.0),
                "moving_average_slope": ma_slope,
                "linear_regression_slope": lr_slope,
                "linear_regression_angle": lr_angle,
                "distance_from_ma": dist_ma,
                "distance_from_vwap": dist_vwap,
                "donchian_breakout_position": donchian_position,
                "macd_histogram": macd_hist_latest,
                "macd_slope": macd_slope,
                "supertrend_state": supertrend_state,
                "price_zscore": price_zscore,
                "bollinger_position": _series_tail(bb_position, default=0.5),
                "cci": cci,
                "rsi": rsi,
                "stoch_k": _series_tail(stoch_k, default=50.0),
                "stoch_d": _series_tail(stoch_d, default=50.0),
                "deviation_from_vwap": dist_vwap,
                "distance_to_range_midpoint": (close_latest - range_midpoint) / max(range_span, _EPS),
                "percentile_rank_within_range": percentile_rank,
                "return_autocorrelation": _safe_float(returns.autocorr(lag=1), default=0.0),
                "chop": choppiness,
            }
        ),
        "volume_participation": _serialize_float_map(
            {
                "volume_reliable_flag": 1.0 if volume_reliable else 0.0,
                "relative_volume": relative_volume,
                "obv_slope": obv_slope,
                "cmf": _series_tail(cmf, default=0.0),
                "mfi": _series_tail(mfi, default=50.0),
                "pvt": _series_tail(pvt, default=0.0),
                "efi": _series_tail(efi, default=0.0),
                "breakout_volume_expansion": breakout_volume_expansion,
                "dry_up_reversal_hint": dry_up_reversal_hint,
            }
        ),
        "meta": {
            "timeframe": timeframe,
            "bars": int(len(df)),
            "pivot_window": int(max(2, int(pivot_window))),
            "volume_reliable": volume_reliable,
            "window_start_utc": df.index[0].isoformat() if hasattr(df.index[0], "isoformat") else str(df.index[0]),
            "window_end_utc": df.index[-1].isoformat() if hasattr(df.index[-1], "isoformat") else str(df.index[-1]),
            "latest_close": close_latest,
            "latest_log_return": _safe_float(log_returns.iloc[-1], default=0.0),
            "atomic_factor_policies": _atomic_factor_policy_payload(),
            "feature_engine_cache": get_feature_engine_cache_stats(),
        },
    }
    field_usage_layer: dict[str, str] = {}
    for group_name, payload in snapshot.items():
        if not isinstance(payload, dict) or group_name == "meta":
            continue
        for field_name in payload:
            field_usage_layer[f"{group_name}.{field_name}"] = "realtime_feature"
    snapshot["meta"]["field_usage_layer"] = field_usage_layer
    return snapshot
