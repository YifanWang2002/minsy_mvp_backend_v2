"""Programmatic scoring for strategy-family recommendation."""

from __future__ import annotations

import math
from typing import Any

from packages.domain.market_data.regime.types import FamilyScores, StrategyFamilyId

_FAMILY_IDS: tuple[StrategyFamilyId, ...] = (
    "trend_continuation",
    "mean_reversion",
    "volatility_regime",
)


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(numeric) or math.isinf(numeric):
        return default
    return numeric


def _softmax(raw: dict[str, float], temperature: float = 0.7) -> dict[str, float]:
    safe_temperature = max(temperature, 1e-6)
    exp_values = {
        key: math.exp(value / safe_temperature)
        for key, value in raw.items()
    }
    total = sum(exp_values.values())
    if total <= 0:
        uniform = 1.0 / float(len(raw))
        return {key: uniform for key in raw}
    return {key: exp_values[key] / total for key in raw}


def _bucket_closeness(center: float, value: float, width: float = 0.5) -> float:
    distance = abs(value - center)
    return _clip01(1.0 - (distance / max(width, 1e-6)))


def score_strategy_families(snapshot: dict[str, Any]) -> FamilyScores:
    """Compute trend/reversion/volatility family probabilities."""

    trend_block = snapshot.get("trend_reversion", {})
    noise_block = snapshot.get("efficiency_noise", {})
    swing_block = snapshot.get("swing_structure", {})
    vol_state = snapshot.get("volatility_state", {})
    vol_level = snapshot.get("volatility_level", {})

    adx = _safe_float(trend_block.get("adx"), default=20.0)
    chop = _safe_float(trend_block.get("chop"), default=50.0)
    er = _clip01(_safe_float(noise_block.get("efficiency_ratio"), default=0.4))
    sign_autocorr = _safe_float(noise_block.get("sign_autocorrelation"), default=0.0)
    breakout_frequency = _clip01(_safe_float(swing_block.get("breakout_frequency"), default=0.05))

    zscore_abs = abs(_safe_float(trend_block.get("price_zscore"), default=0.0))
    bb_position = _clip01(_safe_float(trend_block.get("bollinger_position"), default=0.5))

    vol_percentile = _clip01(_safe_float(vol_state.get("vol_percentile"), default=0.5))
    vol_change_rate = _safe_float(vol_state.get("volatility_change_rate"), default=0.0)
    atr_short_long = _safe_float(vol_state.get("atr_short_long_ratio"), default=1.0)
    squeeze_score = _clip01(_safe_float(vol_state.get("squeeze_score"), default=0.5))
    realized_vol = _clip01(_safe_float(vol_level.get("realized_volatility"), default=0.5))

    adx_strength = _clip01((adx - 15.0) / 25.0)
    chop_strength = _clip01(chop / 100.0)
    anti_chop_strength = 1.0 - chop_strength
    positive_autocorr = _clip01((sign_autocorr + 1.0) / 2.0)
    low_autocorr = 1.0 - positive_autocorr
    zscore_extreme = _clip01(zscore_abs / 2.5)
    bb_mid_closeness = _bucket_closeness(0.5, bb_position, width=0.5)
    atr_regime = _clip01((atr_short_long - 1.0) / 0.8)
    vol_change_norm = _clip01(abs(vol_change_rate) / 0.8)

    trend_raw = (
        0.30 * adx_strength
        + 0.22 * er
        + 0.18 * anti_chop_strength
        + 0.15 * positive_autocorr
        + 0.15 * breakout_frequency
    )
    mean_reversion_raw = (
        0.30 * chop_strength
        + 0.22 * bb_mid_closeness
        + 0.18 * low_autocorr
        + 0.15 * (1.0 - er)
        + 0.15 * zscore_extreme
    )
    volatility_raw = (
        0.28 * vol_percentile
        + 0.22 * atr_regime
        + 0.20 * vol_change_norm
        + 0.16 * squeeze_score
        + 0.14 * realized_vol
    )

    raw_scores = {
        "trend_continuation": _clip01(trend_raw),
        "mean_reversion": _clip01(mean_reversion_raw),
        "volatility_regime": _clip01(volatility_raw),
    }
    probs = _softmax(raw_scores, temperature=0.7)

    ordered = sorted(probs.items(), key=lambda item: item[1], reverse=True)
    top_family = ordered[0][0]
    top_value = ordered[0][1]
    second_value = ordered[1][1] if len(ordered) > 1 else 0.0
    confidence = max(0.0, top_value - second_value)

    evidence_for = {
        "trend_continuation": [],
        "mean_reversion": [],
        "volatility_regime": [],
    }
    evidence_against = {
        "trend_continuation": [],
        "mean_reversion": [],
        "volatility_regime": [],
    }

    if adx_strength >= 0.55:
        evidence_for["trend_continuation"].append("ADX is elevated, trend strength is present.")
    else:
        evidence_against["trend_continuation"].append("ADX is muted, directional trend may be weak.")
    if er >= 0.55:
        evidence_for["trend_continuation"].append("Path efficiency is high, directional moves are cleaner.")
    else:
        evidence_against["trend_continuation"].append("Path efficiency is low, trend continuation is noisier.")

    if chop_strength >= 0.58:
        evidence_for["mean_reversion"].append("Choppiness is high, range-style behavior is active.")
    else:
        evidence_against["mean_reversion"].append("Choppiness is not dominant, pure range logic is less favored.")
    if bb_mid_closeness >= 0.55:
        evidence_for["mean_reversion"].append("Price spends time near Bollinger mid-zone.")
    else:
        evidence_against["mean_reversion"].append("Price is not centered, midpoint reversion is weaker.")

    if vol_percentile >= 0.58:
        evidence_for["volatility_regime"].append("Volatility percentile is elevated.")
    else:
        evidence_against["volatility_regime"].append("Volatility percentile is not elevated.")
    if atr_regime >= 0.55:
        evidence_for["volatility_regime"].append("ATR short/long ratio signals expansion.")
    else:
        evidence_against["volatility_regime"].append("ATR regime is stable, volatility-switch edge is weaker.")

    return FamilyScores(
        trend_continuation=probs["trend_continuation"],
        mean_reversion=probs["mean_reversion"],
        volatility_regime=probs["volatility_regime"],
        recommended_family=top_family,  # type: ignore[arg-type]
        confidence=confidence,
        evidence_for=evidence_for,
        evidence_against=evidence_against,
    )


def build_family_option_subtitles(snapshot: dict[str, Any], scores: FamilyScores) -> dict[str, str]:
    """Build concise option subtitles for AgentUI choice prompt."""

    trend_block = snapshot.get("trend_reversion", {})
    vol_state = snapshot.get("volatility_state", {})
    chop = _safe_float(trend_block.get("chop"), default=50.0)
    adx = _safe_float(trend_block.get("adx"), default=20.0)
    vol_pct = _safe_float(vol_state.get("vol_percentile"), default=0.5) * 100.0

    subtitles: dict[str, str] = {
        "trend_continuation": f"ADX {adx:.1f}, CHOP {chop:.1f}; use trend-following entries.",
        "mean_reversion": f"CHOP {chop:.1f}; prefer fade setups near range edges.",
        "volatility_regime": f"Vol percentile {vol_pct:.0f}%; favor expansion/contraction logic.",
    }
    recommended = scores.recommended_family
    for family in _FAMILY_IDS:
        prefix = "Recommended: " if family == recommended else "Less preferred: "
        subtitles[family] = prefix + subtitles[family]
    return subtitles
