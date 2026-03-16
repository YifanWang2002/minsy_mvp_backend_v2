"""Map pre-strategy preference buckets to concrete analysis timeframes."""

from __future__ import annotations

from packages.domain.market_data.regime.types import TimeframePlan

SUPPORTED_REGIME_TIMEFRAMES: tuple[str, ...] = (
    "1m",
    "5m",
    "15m",
    "1h",
    "4h",
    "1d",
)

_OPPORTUNITY_BUCKETS: frozenset[str] = frozenset(
    {"few_per_month", "few_per_week", "daily", "multiple_per_day"}
)
_HOLDING_BUCKETS: frozenset[str] = frozenset(
    {"intraday_scalp", "intraday", "swing_days", "position_weeks_plus"}
)

_MAPPING_TABLE: dict[tuple[str, str], tuple[str, str, str]] = {
    ("multiple_per_day", "intraday_scalp"): (
        "1m",
        "5m",
        "High setup cadence with scalp holding horizon.",
    ),
    ("daily", "intraday_scalp"): (
        "1m",
        "5m",
        "Scalp holding still requires fast execution context.",
    ),
    ("few_per_week", "intraday_scalp"): (
        "5m",
        "15m",
        "Lower signal cadence allows slightly slower intraday charting.",
    ),
    ("few_per_month", "intraday_scalp"): (
        "15m",
        "1h",
        "Sparse opportunities favor slower scalp-like tactical entries.",
    ),
    ("multiple_per_day", "intraday"): (
        "5m",
        "15m",
        "Intraday with frequent setups emphasizes tactical execution windows.",
    ),
    ("daily", "intraday"): (
        "15m",
        "1h",
        "Daily setup cadence for intraday entries favors mid-speed context.",
    ),
    ("few_per_week", "intraday"): (
        "1h",
        "4h",
        "Fewer intraday opportunities require broader context alignment.",
    ),
    ("few_per_month", "intraday"): (
        "4h",
        "1d",
        "Sparse intraday opportunities should anchor on higher-level structure.",
    ),
    ("multiple_per_day", "swing_days"): (
        "15m",
        "1h",
        "Frequent setups with multi-day holding need fast trigger + broad context.",
    ),
    ("daily", "swing_days"): (
        "1h",
        "4h",
        "Balanced swing cadence maps to medium-term trend structure.",
    ),
    ("few_per_week", "swing_days"): (
        "4h",
        "1d",
        "Lower swing cadence favors slower regime filters.",
    ),
    ("few_per_month", "swing_days"): (
        "1d",
        "4h",
        "Sparse swing opportunities should prioritize daily-level structure.",
    ),
    ("multiple_per_day", "position_weeks_plus"): (
        "1h",
        "4h",
        "Long holding with frequent entries uses tactical 1h timing.",
    ),
    ("daily", "position_weeks_plus"): (
        "4h",
        "1d",
        "Position style with daily opportunities needs higher timeframe anchor.",
    ),
    ("few_per_week", "position_weeks_plus"): (
        "1d",
        "4h",
        "Position strategy with weekly setups is primarily daily-driven.",
    ),
    ("few_per_month", "position_weeks_plus"): (
        "1d",
        "4h",
        "Monthly position setups should remain daily-regime first.",
    ),
}


def map_pre_strategy_timeframes(
    *,
    opportunity_frequency_bucket: str,
    holding_period_bucket: str,
) -> TimeframePlan:
    """Return one primary + one secondary timeframe for regime analysis."""

    frequency_key = str(opportunity_frequency_bucket).strip().lower()
    holding_key = str(holding_period_bucket).strip().lower()
    if frequency_key not in _OPPORTUNITY_BUCKETS:
        raise ValueError(f"Unsupported opportunity_frequency_bucket: {opportunity_frequency_bucket}")
    if holding_key not in _HOLDING_BUCKETS:
        raise ValueError(f"Unsupported holding_period_bucket: {holding_period_bucket}")

    primary, secondary, reason = _MAPPING_TABLE[(frequency_key, holding_key)]
    return TimeframePlan(
        primary=primary,
        secondary=secondary,
        candidates=(primary, secondary),
        mapping_reason=reason,
    )

