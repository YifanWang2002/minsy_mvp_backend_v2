from __future__ import annotations

from src.engine.feature import FactorKind, FeatureRegistry
from src.engine.feature.indicators import IndicatorRegistry, list_indicators


def test_indicator_registry_is_backed_by_feature_registry() -> None:
    names = list_indicators()
    assert len(names) >= 80

    for expected in ("ema", "rsi", "atr", "macd", "cdl_engulfing", "zscore"):
        assert expected in names
        assert IndicatorRegistry.get(expected) is not None
        assert FeatureRegistry.has(kind=FactorKind.INDICATOR, name=expected)


def test_feature_registry_supports_non_indicator_factor_kinds() -> None:
    factor_name = "ml_alpha_signal_v1"
    metadata = {"name": factor_name, "description": "dummy ml factor"}

    FeatureRegistry.register(
        kind=FactorKind.ML_SIGNAL,
        name=factor_name,
        metadata=metadata,
    )

    assert FeatureRegistry.has(kind=FactorKind.ML_SIGNAL, name=factor_name)
    assert FeatureRegistry.get_metadata(kind=FactorKind.ML_SIGNAL, name=factor_name) == metadata

    FeatureRegistry.clear(kind=FactorKind.ML_SIGNAL)
    assert not FeatureRegistry.has(kind=FactorKind.ML_SIGNAL, name=factor_name)
