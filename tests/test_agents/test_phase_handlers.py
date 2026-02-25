from __future__ import annotations

from uuid import uuid4

import pytest

from src.agents.handler_protocol import PhaseContext, RuntimePolicy
from src.agents.handler_registry import get_handler
from src.agents.handlers.deployment_handler import DeploymentHandler
from src.agents.handlers.strategy_handler import StrategyHandler
from src.agents.handlers.stress_test_handler import StressTestHandler
from src.agents.phases import Phase


def test_handler_registry_uses_real_handlers_for_later_phases() -> None:
    assert isinstance(get_handler(Phase.STRATEGY.value), StrategyHandler)
    assert isinstance(get_handler(Phase.STRESS_TEST.value), StressTestHandler)
    assert isinstance(get_handler(Phase.DEPLOYMENT.value), DeploymentHandler)


@pytest.mark.asyncio
async def test_strategy_handler_collects_strategy_id_without_forcing_transition() -> (
    None
):
    handler = StrategyHandler()
    strategy_id = str(uuid4())
    ctx = PhaseContext(
        user_id=uuid4(),
        session_artifacts={
            Phase.STRATEGY.value: {"profile": {}, "missing_fields": ["strategy_id"]},
            Phase.PRE_STRATEGY.value: {
                "profile": {
                    "target_market": "crypto",
                    "target_instrument": "BTCUSDT",
                }
            },
        },
        language="en",
        runtime_policy=RuntimePolicy(),
    )

    prompt = handler.build_prompt(ctx, "save this strategy")
    assert any(tool.get("server_label") == "strategy" for tool in (prompt.tools or []))

    result = await handler.post_process(ctx, [{"strategy_id": strategy_id}], object())
    assert result.completed is False
    assert result.next_phase is None
    assert result.missing_fields == []


@pytest.mark.asyncio
async def test_strategy_handler_can_auto_advance_to_deployment_on_ai_confirmation() -> (
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
            },
            Phase.PRE_STRATEGY.value: {
                "profile": {
                    "target_market": "crypto",
                    "target_instrument": "BTCUSDT",
                }
            },
        },
        language="en",
        runtime_policy=RuntimePolicy(),
    )

    result = await handler.post_process(
        ctx,
        [
            {
                "strategy_confirmed": True,
                "strategy_name": "ignored_by_handler",
            }
        ],
        object(),
    )
    assert result.completed is True
    assert result.next_phase == Phase.DEPLOYMENT.value
    assert result.transition_reason == "strategy_ai_confirmed_to_deployment"

    strategy_profile = result.artifacts[Phase.STRATEGY.value]["profile"]
    deployment_profile = result.artifacts[Phase.DEPLOYMENT.value]["profile"]
    deployment_runtime = result.artifacts[Phase.DEPLOYMENT.value]["runtime"]

    assert strategy_profile["strategy_confirmed"] is True
    assert isinstance(strategy_profile.get("strategy_last_confirmed_at"), str)
    assert deployment_profile["strategy_id"] == strategy_id
    assert deployment_profile["deployment_status"] == "ready"
    assert deployment_runtime["strategy_id"] == strategy_id
    assert deployment_runtime["deployment_status"] == "ready"


@pytest.mark.asyncio
async def test_stress_test_handler_done_and_failed_return_to_strategy() -> None:
    handler = StressTestHandler()
    strategy_id = str(uuid4())
    job_id = str(uuid4())

    ctx_done = PhaseContext(
        user_id=uuid4(),
        session_artifacts={
            Phase.STRATEGY.value: {"profile": {"strategy_id": strategy_id}},
            Phase.STRESS_TEST.value: {"profile": {}, "missing_fields": []},
        },
        language="en",
        runtime_policy=RuntimePolicy(),
    )
    done_result = await handler.post_process(
        ctx_done,
        [{"backtest_job_id": job_id, "backtest_status": "done"}],
        object(),
    )
    assert done_result.completed is True
    assert done_result.next_phase == Phase.STRATEGY.value
    assert done_result.phase_status.get("stress_test_decision") == "hold"

    ctx_failed = PhaseContext(
        user_id=uuid4(),
        session_artifacts={
            Phase.STRATEGY.value: {"profile": {"strategy_id": strategy_id}},
            Phase.STRESS_TEST.value: {
                "profile": {"backtest_job_id": job_id},
                "missing_fields": [],
            },
        },
        language="en",
        runtime_policy=RuntimePolicy(),
    )
    failed_result = await handler.post_process(
        ctx_failed,
        [{"backtest_status": "failed", "backtest_error_code": "BACKTEST_RUN_ERROR"}],
        object(),
    )
    assert failed_result.completed is True
    assert failed_result.next_phase == Phase.STRATEGY.value


@pytest.mark.asyncio
async def test_deployment_handler_keeps_phase_on_deployed_status() -> None:
    handler = DeploymentHandler()
    ctx = PhaseContext(
        user_id=uuid4(),
        session_artifacts={
            Phase.DEPLOYMENT.value: {
                "profile": {},
                "missing_fields": ["deployment_status"],
            },
        },
        language="en",
        runtime_policy=RuntimePolicy(),
    )

    result = await handler.post_process(
        ctx, [{"deployment_status": "deployed"}], object()
    )
    assert result.completed is False
    assert result.next_phase is None


@pytest.mark.asyncio
async def test_deployment_handler_blocked_returns_to_strategy() -> None:
    handler = DeploymentHandler()
    ctx = PhaseContext(
        user_id=uuid4(),
        session_artifacts={
            Phase.DEPLOYMENT.value: {
                "profile": {},
                "missing_fields": ["deployment_status"],
            },
        },
        language="en",
        runtime_policy=RuntimePolicy(),
    )

    result = await handler.post_process(
        ctx, [{"deployment_status": "blocked"}], object()
    )
    assert result.completed is True
    assert result.next_phase == Phase.STRATEGY.value
