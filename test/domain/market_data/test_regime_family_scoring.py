from __future__ import annotations

from packages.domain.market_data.regime.family_scoring import (
    build_family_option_subtitles,
    score_strategy_families,
)


def test_family_scoring_prefers_trend_when_trend_signals_dominate() -> None:
    snapshot = {
        "trend_reversion": {
            "adx": 36.0,
            "chop": 24.0,
            "price_zscore": 0.4,
            "bollinger_position": 0.82,
        },
        "efficiency_noise": {
            "efficiency_ratio": 0.82,
            "sign_autocorrelation": 0.55,
            "directional_persistence": 0.76,
        },
        "swing_structure": {
            "breakout_frequency": 0.38,
        },
        "volatility_state": {
            "vol_percentile": 0.42,
            "volatility_change_rate": 0.08,
            "atr_short_long_ratio": 1.03,
            "squeeze_score": 0.35,
        },
        "volatility_level": {
            "realized_volatility": 0.31,
        },
    }

    scores = score_strategy_families(snapshot)

    assert scores.recommended_family == "trend_continuation"
    total = (
        scores.trend_continuation
        + scores.mean_reversion
        + scores.volatility_regime
    )
    assert abs(total - 1.0) < 1e-6
    assert 0.0 <= scores.confidence <= 1.0


def test_family_option_subtitles_returns_all_three_families() -> None:
    snapshot = {
        "trend_reversion": {"adx": 24.0, "chop": 52.0},
        "volatility_state": {"vol_percentile": 0.63},
    }
    scores = score_strategy_families(
        {
            **snapshot,
            "efficiency_noise": {},
            "swing_structure": {},
            "volatility_level": {},
        }
    )

    subtitles = build_family_option_subtitles(snapshot, scores)

    assert set(subtitles.keys()) == {
        "trend_continuation",
        "mean_reversion",
        "volatility_regime",
    }
    assert all(isinstance(value, str) and value for value in subtitles.values())
