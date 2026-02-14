from __future__ import annotations

from src.agents.handlers import pre_strategy_handler as handler_mod
from src.agents.handlers.pre_strategy_handler import PreStrategyHandler
from src.agents.skills import pre_strategy_skills as skills_mod


def _sample_market_catalog() -> dict[str, tuple[str, ...]]:
    return {
        "us_stocks": ("SPY", "QQQ", "AAPL"),
        "crypto": ("BTCUSD", "ETHUSD"),
        "forex": ("EURUSD", "USDJPY"),
        "futures": ("ES", "NQ"),
    }


def _sample_valid_values() -> dict[str, set[str]]:
    return {
        "target_market": {"us_stocks", "crypto", "forex", "futures"},
        "target_instrument": {"SPY", "QQQ", "AAPL", "BTCUSD", "ETHUSD", "EURUSD", "USDJPY", "ES", "NQ"},
        "opportunity_frequency_bucket": {
            "few_per_month",
            "few_per_week",
            "daily",
            "multiple_per_day",
        },
        "holding_period_bucket": {
            "intraday_scalp",
            "intraday",
            "swing_days",
            "position_weeks_plus",
        },
    }


def test_symbol_conversion_rules_across_markets() -> None:
    assert skills_mod.get_market_data_symbol_for_market_instrument(
        market="us_stocks",
        instrument="brk.b",
    ) == "BRK-B"
    assert skills_mod.get_market_data_symbol_for_market_instrument(
        market="crypto",
        instrument="BTCUSD",
    ) == "BTC-USD"
    assert skills_mod.get_market_data_symbol_for_market_instrument(
        market="forex",
        instrument="EURUSD",
    ) == "EURUSD=X"
    assert skills_mod.get_market_data_symbol_for_market_instrument(
        market="futures",
        instrument="ES",
    ) == "ES=F"

    assert skills_mod.get_tradingview_symbol_for_market_instrument(
        market="us_stocks",
        instrument="SPY",
    ) == "SPY"
    assert skills_mod.get_tradingview_symbol_for_market_instrument(
        market="crypto",
        instrument="BTCUSD",
    ) == "BINANCE:BTCUSDT"
    assert skills_mod.get_tradingview_symbol_for_market_instrument(
        market="forex",
        instrument="EURUSD",
    ) == "FX:EURUSD"
    assert skills_mod.get_tradingview_symbol_for_market_instrument(
        market="futures",
        instrument="ES",
    ) == "ES1!"


def test_dynamic_state_uses_catalog_and_normalized_values(monkeypatch) -> None:
    monkeypatch.setattr(
        skills_mod,
        "get_pre_strategy_market_instrument_map",
        _sample_market_catalog,
    )

    state = skills_mod.build_pre_strategy_dynamic_state(
        missing_fields=["target_instrument", "opportunity_frequency_bucket"],
        collected_fields={"target_market": "stock", "target_instrument": "spy"},
        kyc_profile={"risk_tolerance": "aggressive"},
        symbol_newly_provided_this_turn_hint=True,
        inferred_instrument_from_user_message="spy",
    )

    assert "- selected_target_market: us_stocks" in state
    assert "- selected_target_instrument: SPY" in state
    assert "- allowed_instruments_for_selected_market: SPY, QQQ, AAPL" in state
    assert "- mapped_market_data_symbol_for_selected_instrument: SPY" in state
    assert "- mapped_tradingview_symbol_for_selected_instrument: SPY" in state
    assert "- symbol_newly_provided_this_turn_hint: true" in state
    assert "market_symbol_catalog:" not in state
    assert "instrument_symbol_map (instrument:market_data|tradingview):" not in state


def test_sanitize_profile_infers_market_and_drops_cross_market_instrument(monkeypatch) -> None:
    monkeypatch.setattr(handler_mod, "get_pre_strategy_valid_values", _sample_valid_values)
    monkeypatch.setattr(handler_mod, "get_instruments_for_market", lambda market: _sample_market_catalog().get(market, ()))
    monkeypatch.setattr(
        handler_mod,
        "get_market_for_instrument",
        lambda instrument: next(
            (
                market
                for market, symbols in _sample_market_catalog().items()
                if instrument in symbols
            ),
            None,
        ),
    )

    handler = PreStrategyHandler()

    inferred = handler._sanitize_profile(
        {
            "target_instrument": "BTCUSD",
            "opportunity_frequency_bucket": "daily",
            "holding_period_bucket": "intraday",
        }
    )
    assert inferred["target_market"] == "crypto"
    assert inferred["target_instrument"] == "BTCUSD"

    mismatch = handler._sanitize_profile(
        {
            "target_market": "us_stocks",
            "target_instrument": "BTCUSD",
            "opportunity_frequency_bucket": "daily",
            "holding_period_bucket": "intraday",
        }
    )
    assert mismatch["target_market"] == "us_stocks"
    assert "target_instrument" not in mismatch


def test_validate_patch_normalizes_market_alias_and_symbol_case(monkeypatch) -> None:
    monkeypatch.setattr(handler_mod, "get_pre_strategy_valid_values", _sample_valid_values)

    handler = PreStrategyHandler()
    validated = handler._validate_patch(
        {
            "target_market": "stock",
            "target_instrument": "spy",
            "opportunity_frequency_bucket": "daily",
            "holding_period_bucket": "swing_days",
        }
    )

    assert validated == {
        "target_market": "us_stocks",
        "target_instrument": "SPY",
        "opportunity_frequency_bucket": "daily",
        "holding_period_bucket": "swing_days",
    }


def test_infer_instrument_supports_market_data_style_aliases(monkeypatch) -> None:
    monkeypatch.setattr(
        handler_mod,
        "get_pre_strategy_market_instrument_map",
        _sample_market_catalog,
    )

    assert (
        handler_mod._infer_instrument_from_message(
            user_message="请看 btc-usd 的行情",
            selected_market="crypto",
        )
        == "BTCUSD"
    )
    assert (
        handler_mod._infer_instrument_from_message(
            user_message="trade eurusd=x",
            selected_market="forex",
        )
        == "EURUSD"
    )
    assert (
        handler_mod._infer_instrument_from_message(
            user_message="Let's do es=f.",
            selected_market="futures",
        )
        == "ES"
    )


def test_infer_instrument_does_not_false_positive_on_short_alias(monkeypatch) -> None:
    monkeypatch.setattr(
        handler_mod,
        "get_pre_strategy_market_instrument_map",
        _sample_market_catalog,
    )

    inferred = handler_mod._infer_instrument_from_message(
        user_message="I want the best setup this week.",
        selected_market="futures",
    )
    assert inferred is None


def test_validate_patch_rejects_injection_like_values(monkeypatch) -> None:
    monkeypatch.setattr(handler_mod, "get_pre_strategy_valid_values", _sample_valid_values)

    handler = PreStrategyHandler()
    validated = handler._validate_patch(
        {
            "target_market": " stock; DROP TABLE users; ",
            "target_instrument": "<script>SPY</script>",
            "opportunity_frequency_bucket": "daily<script>",
            "holding_period_bucket": "intraday",
        }
    )
    assert validated == {"holding_period_bucket": "intraday"}


def test_dynamic_state_handles_empty_market_catalog(monkeypatch) -> None:
    monkeypatch.setattr(
        skills_mod,
        "get_pre_strategy_market_instrument_map",
        lambda: {},
    )

    state = skills_mod.build_pre_strategy_dynamic_state(
        missing_fields=["target_market", "target_instrument"],
        collected_fields={},
        kyc_profile={},
    )

    assert "available_markets: none - no local market parquet data found" in state
    assert "market_symbol_catalog:" not in state
    assert "instrument_symbol_map (instrument:market_data|tradingview):" not in state
