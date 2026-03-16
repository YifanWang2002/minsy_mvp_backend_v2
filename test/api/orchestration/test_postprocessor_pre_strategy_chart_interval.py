from __future__ import annotations

from apps.api.agents.phases import Phase
from apps.api.orchestration import ChatOrchestrator


def test_pre_strategy_chart_interval_follows_timeframe_plan_primary() -> None:
    orchestrator = ChatOrchestrator(None)  # type: ignore[arg-type]
    artifacts = {
        Phase.PRE_STRATEGY.value: {
            "profile": {
                "target_market": "crypto",
                "target_instrument": "BTCUSD",
            },
            "runtime": {
                "timeframe_plan": {
                    "primary": "1h",
                    "secondary": "4h",
                    "mapping_reason": "test",
                }
            },
        }
    }

    payloads = orchestrator._ensure_pre_strategy_chart_payload(
        phase_before=Phase.PRE_STRATEGY.value,
        artifacts=artifacts,
        genui_payloads=[
            {
                "type": "choice_prompt",
                "choice_id": "strategy_family_choice",
                "question": "Which family?",
                "options": [
                    {"id": "trend_continuation", "label": "Trend", "subtitle": "..."},
                    {"id": "mean_reversion", "label": "MR", "subtitle": "..."},
                ],
            }
        ],
        instrument_before="BTCUSD",
    )

    assert payloads[0]["type"] == "tradingview_chart"
    assert payloads[0]["symbol"] == "BINANCE:BTCUSDT"
    assert payloads[0]["interval"] == "60"


def test_pre_strategy_chart_not_inserted_when_interval_default_and_instrument_unchanged() -> None:
    orchestrator = ChatOrchestrator(None)  # type: ignore[arg-type]
    artifacts = {
        Phase.PRE_STRATEGY.value: {
            "profile": {
                "target_market": "us_stocks",
                "target_instrument": "SPY",
            },
            "runtime": {},
        }
    }
    existing = [
        {
            "type": "choice_prompt",
            "choice_id": "opportunity_frequency_bucket",
            "question": "How often?",
            "options": [
                {"id": "daily", "label": "Daily", "subtitle": "..."},
                {"id": "few_per_week", "label": "Weekly", "subtitle": "..."},
            ],
        }
    ]

    payloads = orchestrator._ensure_pre_strategy_chart_payload(
        phase_before=Phase.PRE_STRATEGY.value,
        artifacts=artifacts,
        genui_payloads=existing,
        instrument_before="SPY",
    )

    assert payloads == existing
