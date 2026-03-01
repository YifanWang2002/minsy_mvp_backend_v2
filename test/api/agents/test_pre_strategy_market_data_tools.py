from __future__ import annotations

from uuid import uuid4

from apps.api.agents.handler_protocol import PhaseContext, RuntimePolicy
from apps.api.agents.handlers.pre_strategy_handler import PreStrategyHandler
from apps.api.agents.phases import Phase


def test_000_pre_strategy_build_prompt_exposes_market_data_tools() -> None:
    handler = PreStrategyHandler()
    ctx = PhaseContext(
        user_id=uuid4(),
        session_id=uuid4(),
        session_artifacts={
            Phase.PRE_STRATEGY.value: {
                "profile": {},
                "missing_fields": list(handler.required_fields),
            }
        },
        language="en",
        runtime_policy=RuntimePolicy(),
    )

    prompt = handler.build_prompt(ctx, "I want to trade SPY")
    assert prompt.tools is not None
    assert len(prompt.tools) == 1

    tool_def = prompt.tools[0]
    assert tool_def["server_label"] == "market_data"
    assert tuple(tool_def["allowed_tools"]) == (
        "check_symbol_available",
        "get_symbol_data_coverage",
        "market_data_detect_missing_ranges",
        "market_data_fetch_missing_ranges",
        "market_data_get_sync_job",
    )
    assert "download consent is mandatory" in prompt.instructions.lower()
    assert "1-2 minutes" in prompt.instructions
    assert "market_data_fetch_missing_ranges" in prompt.instructions
