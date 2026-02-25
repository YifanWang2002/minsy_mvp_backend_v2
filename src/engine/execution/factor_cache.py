"""Execution-layer re-export of shared factor cache."""

from src.engine.market_data.factor_cache import (
    FactorCache,
    FactorCacheStats,
    factor_signature,
)

__all__ = ["FactorCache", "FactorCacheStats", "factor_signature"]
