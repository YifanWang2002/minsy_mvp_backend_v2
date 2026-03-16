from __future__ import annotations

from packages.domain.market_data.regime.timeframe_mapper import (
    SUPPORTED_REGIME_TIMEFRAMES,
    map_pre_strategy_timeframes,
)


def test_timeframe_mapper_returns_supported_primary_and_secondary() -> None:
    opportunities = (
        "few_per_month",
        "few_per_week",
        "daily",
        "multiple_per_day",
    )
    holdings = (
        "intraday_scalp",
        "intraday",
        "swing_days",
        "position_weeks_plus",
    )

    supported = set(SUPPORTED_REGIME_TIMEFRAMES)
    for frequency in opportunities:
        for holding in holdings:
            plan = map_pre_strategy_timeframes(
                opportunity_frequency_bucket=frequency,
                holding_period_bucket=holding,
            )
            assert plan.primary in supported
            assert plan.secondary in supported
            assert plan.candidates == (plan.primary, plan.secondary)
            assert isinstance(plan.mapping_reason, str)
            assert plan.mapping_reason.strip()


def test_timeframe_mapper_known_high_frequency_scalp_mapping() -> None:
    plan = map_pre_strategy_timeframes(
        opportunity_frequency_bucket="multiple_per_day",
        holding_period_bucket="intraday_scalp",
    )

    assert plan.primary == "1m"
    assert plan.secondary == "5m"
