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
async def test_strategy_handler_collects_strategy_id_without_forcing_transition() -> None:
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
async def test_stress_test_handler_done_and_failed_transitions() -> None:
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
    assert done_result.completed is False
    assert done_result.next_phase is None

    deploy_result = await handler.post_process(
        ctx_done,
        [{"stress_test_decision": "deploy"}],
        object(),
    )
    assert deploy_result.completed is True
    assert deploy_result.next_phase == Phase.DEPLOYMENT.value

    iterate_result = await handler.post_process(
        ctx_done,
        [{"stress_test_decision": "iterate"}],
        object(),
    )
    assert iterate_result.completed is True
    assert iterate_result.next_phase == Phase.STRATEGY.value

    ctx_failed = PhaseContext(
        user_id=uuid4(),
        session_artifacts={
            Phase.STRATEGY.value: {"profile": {"strategy_id": strategy_id}},
            Phase.STRESS_TEST.value: {"profile": {"backtest_job_id": job_id}, "missing_fields": []},
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
async def test_deployment_handler_transitions_to_completed() -> None:
    handler = DeploymentHandler()
    ctx = PhaseContext(
        user_id=uuid4(),
        session_artifacts={
            Phase.DEPLOYMENT.value: {"profile": {}, "missing_fields": ["deployment_status"]},
        },
        language="en",
        runtime_policy=RuntimePolicy(),
    )

    result = await handler.post_process(ctx, [{"deployment_status": "deployed"}], object())
    assert result.completed is True
    assert result.next_phase == Phase.COMPLETED.value
