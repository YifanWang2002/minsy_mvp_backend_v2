from __future__ import annotations

from apps.api.agents.skills.kyc_skills import build_kyc_dynamic_state
from apps.api.agents.skills.pre_strategy_skills import build_pre_strategy_dynamic_state
from apps.api.agents.skills.strategy_skills import build_strategy_dynamic_state


def test_000_kyc_state_is_compact_kv_block() -> None:
    block = build_kyc_dynamic_state(
        missing_fields=["risk_tolerance"],
        collected_fields={"trading_years_bucket": "years_3_5"},
    )
    assert block.startswith("[SESSION STATE]\n")
    assert "phase=kyc" in block
    assert "collected=trading_years_bucket=years_3_5" in block
    assert "has_missing=true" in block
    assert "- already_collected:" not in block


def test_010_pre_strategy_state_keeps_key_fields_in_compact_format() -> None:
    block = build_pre_strategy_dynamic_state(
        missing_fields=["holding_period_bucket"],
        collected_fields={
            "target_market": "us_stocks",
            "target_instrument": "SPY",
            "opportunity_frequency_bucket": "daily",
        },
        kyc_profile={
            "trading_years_bucket": "years_5_plus",
            "risk_tolerance": "moderate",
        },
        symbol_newly_provided_this_turn_hint=True,
        inferred_instrument_from_user_message="SPY",
    )
    assert "phase=pre_strategy" in block
    assert "target_market=us_stocks" in block
    assert "target_instrument=SPY" in block
    assert "mapped_market_data_symbol=SPY" in block
    assert "download_requires_user_confirmation=true" in block
    assert "- target_market:" not in block


def test_020_strategy_state_keeps_key_fields_in_compact_format() -> None:
    block = build_strategy_dynamic_state(
        missing_fields=[],
        collected_fields={
            "strategy_id": "00000000-0000-4000-8000-000000000001",
            "strategy_market": "us_stocks",
            "strategy_primary_symbol": "SPY",
            "strategy_tickers_csv": "SPY,QQQ",
            "strategy_timeframe": "1d",
        },
        pre_strategy_fields={"target_market": "us_stocks"},
        session_id="00000000-0000-4000-8000-000000000002",
    )
    assert "phase=strategy" in block
    assert "strategy_id=00000000-0000-4000-8000-000000000001" in block
    assert "tool_compat_session_id=00000000-0000-4000-8000-000000000002" in block
    assert "missing=none - all collected" in block
    assert "- confirmed_strategy_id:" not in block
