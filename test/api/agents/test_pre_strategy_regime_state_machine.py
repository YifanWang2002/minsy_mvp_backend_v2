from __future__ import annotations

from uuid import uuid4

from apps.api.agents.handler_protocol import PhaseContext, RuntimePolicy
from apps.api.agents.handlers.pre_strategy_handler import PreStrategyHandler
from apps.api.agents.phases import Phase


def test_fallback_strategy_family_prompt_requires_regime_ready() -> None:
    handler = PreStrategyHandler()
    artifacts = {
        Phase.PRE_STRATEGY.value: {
            "profile": {
                "target_market": "crypto",
                "target_instrument": "BTCUSD",
                "opportunity_frequency_bucket": "daily",
                "holding_period_bucket": "swing_days",
            },
            "runtime": {
                "instrument_data_status": "local_ready",
                "instrument_data_market": "crypto",
                "instrument_data_symbol": "BTCUSD",
                "instrument_available_locally": True,
                "regime_snapshot_status": "pending",
            },
            "missing_fields": ["strategy_family_choice"],
        }
    }
    ctx = PhaseContext(
        user_id=uuid4(),
        session_id=uuid4(),
        session_artifacts=artifacts,
        runtime_policy=RuntimePolicy(),
    )

    payload = handler.build_fallback_choice_prompt(
        missing_fields=["strategy_family_choice"],
        ctx=ctx,
    )

    assert payload is None


def test_fallback_strategy_family_prompt_uses_three_options_after_regime_ready() -> None:
    handler = PreStrategyHandler()
    artifacts = {
        Phase.PRE_STRATEGY.value: {
            "profile": {
                "target_market": "crypto",
                "target_instrument": "BTCUSD",
                "opportunity_frequency_bucket": "daily",
                "holding_period_bucket": "swing_days",
            },
            "runtime": {
                "instrument_data_status": "local_ready",
                "instrument_data_market": "crypto",
                "instrument_data_symbol": "BTCUSD",
                "instrument_available_locally": True,
                "regime_snapshot_status": "ready",
                "regime_family_subtitles": {
                    "trend_continuation": "Recommended: trend remains strong.",
                    "mean_reversion": "Less preferred: midpoint pullbacks weak.",
                    "volatility_regime": "Less preferred: volatility stable.",
                },
            },
            "missing_fields": ["strategy_family_choice"],
        }
    }
    ctx = PhaseContext(
        user_id=uuid4(),
        session_id=uuid4(),
        session_artifacts=artifacts,
        runtime_policy=RuntimePolicy(),
    )

    payload = handler.build_fallback_choice_prompt(
        missing_fields=["strategy_family_choice"],
        ctx=ctx,
    )

    assert isinstance(payload, dict)
    assert payload["type"] == "choice_prompt"
    assert payload["choice_id"] == "strategy_family_choice"
    assert [option["id"] for option in payload["options"]] == [
        "trend_continuation",
        "mean_reversion",
        "volatility_regime",
    ]
    assert all(option.get("subtitle") for option in payload["options"])
