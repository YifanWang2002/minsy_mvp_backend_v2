from __future__ import annotations

from uuid import uuid4

from src.agents.handler_protocol import PhaseContext, RuntimePolicy
from src.agents.handlers.stub_handler import StubHandler


def test_stub_handler_loads_strategy_stage_markdown() -> None:
    handler = StubHandler("strategy")
    ctx = PhaseContext(
        user_id=uuid4(),
        session_artifacts={},
        language="en",
        runtime_policy=RuntimePolicy(phase_stage="schema_only"),
    )

    prompt = handler.build_prompt(ctx, "Help me design a strategy")

    assert "Minsy Strategy Agent" in prompt.instructions
    assert "[STAGE_MARKER_STRATEGY_SCHEMA_ONLY]" in prompt.instructions
