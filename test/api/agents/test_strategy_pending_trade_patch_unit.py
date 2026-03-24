from __future__ import annotations

from uuid import uuid4

import pytest

from apps.api.agents.handler_protocol import PhaseContext, RuntimePolicy
from apps.api.agents.handlers.strategy_handler import StrategyHandler
from apps.api.agents.phases import Phase


def _pending_patch_payload(strategy_id: str) -> dict[str, object]:
    return {
        "strategy_id": strategy_id,
        "patch_ops": [
            {
                "op": "replace",
                "path": "/trade/long/exits/0/stop/value",
                "value": 0.012,
            }
        ],
        "source_trade": {
            "job_id": str(uuid4()),
            "trade_index": 5,
            "trade_uid": "5:entry:exit",
        },
    }


def test_strategy_handler_validate_patch_accepts_pending_trade_patch() -> None:
    handler = StrategyHandler()
    strategy_id = str(uuid4())

    validated = handler._validate_patch(  # noqa: SLF001
        {"pending_trade_patch": _pending_patch_payload(strategy_id)}
    )

    assert "pending_trade_patch" in validated
    payload = validated["pending_trade_patch"]
    assert payload["strategy_id"] == strategy_id
    assert isinstance(payload["patch_ops"], list)


@pytest.mark.asyncio
async def test_strategy_handler_post_process_can_store_and_clear_pending_patch() -> (
    None
):
    handler = StrategyHandler()
    strategy_id = str(uuid4())
    ctx = PhaseContext(
        user_id=uuid4(),
        session_artifacts={
            Phase.STRATEGY.value: {
                "profile": {"strategy_id": strategy_id},
                "missing_fields": [],
            }
        },
        runtime_policy=RuntimePolicy(phase_stage="artifact_ops"),
        turn_context={},
    )

    stored = await handler.post_process(
        ctx,
        raw_patches=[{"pending_trade_patch": _pending_patch_payload(strategy_id)}],
        db=object(),
    )
    profile = stored.artifacts[Phase.STRATEGY.value]["profile"]
    assert "pending_trade_patch" in profile

    cleared = await handler.post_process(
        PhaseContext(
            user_id=ctx.user_id,
            session_artifacts=stored.artifacts,
            runtime_policy=RuntimePolicy(phase_stage="artifact_ops"),
            turn_context={},
        ),
        raw_patches=[{"clear_pending_trade_patch": True}],
        db=object(),
    )
    profile_after_clear = cleared.artifacts[Phase.STRATEGY.value]["profile"]
    assert "pending_trade_patch" not in profile_after_clear
