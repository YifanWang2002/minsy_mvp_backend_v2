from __future__ import annotations

from src.engine.market_data.factor_cache import FactorCache, factor_signature


def test_factor_cache_hits_for_same_signature_and_bar() -> None:
    cache = FactorCache(max_entries=100)
    signature = factor_signature(
        factor_type="ema",
        params={"length": 20},
        source="close",
    )
    calls = {"count": 0}

    def _compute() -> float:
        calls["count"] += 1
        return 123.45

    first = cache.get_or_compute(
        market="crypto",
        symbol="BTCUSDT",
        timeframe="1m",
        signature=signature,
        bar_ts=1_700_000_000_000,
        compute=_compute,
    )
    second = cache.get_or_compute(
        market="crypto",
        symbol="BTCUSDT",
        timeframe="1m",
        signature=signature,
        bar_ts=1_700_000_000_000,
        compute=_compute,
    )

    assert first == second == 123.45
    assert calls["count"] == 1
    stats = cache.stats()
    assert stats.misses == 1
    assert stats.hits == 1
    assert stats.hit_rate == 0.5


def test_factor_cache_recomputes_for_new_bar_timestamp() -> None:
    cache = FactorCache(max_entries=100)
    signature = factor_signature(
        factor_type="rsi",
        params={"length": 14},
        source="close",
    )
    calls = {"count": 0}

    def _compute() -> int:
        calls["count"] += 1
        return calls["count"]

    first = cache.get_or_compute(
        market="stocks",
        symbol="AAPL",
        timeframe="5m",
        signature=signature,
        bar_ts=1000,
        compute=_compute,
    )
    second = cache.get_or_compute(
        market="stocks",
        symbol="AAPL",
        timeframe="5m",
        signature=signature,
        bar_ts=2000,
        compute=_compute,
    )

    assert first == 1
    assert second == 2
    assert calls["count"] == 2
