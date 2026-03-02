from __future__ import annotations

import json
from types import SimpleNamespace
from uuid import uuid4

from apps.api.agents.handler_protocol import PhaseContext
from apps.api.agents.handlers import deployment_handler as deployment_handler_module
from apps.api.agents.handlers.deployment_handler import DeploymentHandler
from apps.api.agents.phases import Phase


def _mcp_success_call(name: str, data: dict[str, object]) -> dict[str, object]:
    return {
        "type": "mcp_call",
        "name": name,
        "status": "success",
        "output": json.dumps({"tool": name, "ok": True, "data": data}),
    }


def _mcp_flat_success_call(name: str, data: dict[str, object]) -> dict[str, object]:
    payload = {"tool": name, "ok": True, "category": "trading", **data}
    return {
        "type": "mcp_call",
        "name": name,
        "status": "success",
        "output": json.dumps(payload),
    }


async def test_deployment_handler_blocks_without_broker_and_offers_builtin_sandbox() -> None:
    handler = DeploymentHandler()
    phase_artifacts = handler.init_artifacts()
    phase_artifacts["profile"].update(
        {
            "strategy_name": "Sandbox Momentum",
            "strategy_market": "crypto",
            "strategy_primary_symbol": "BTCUSD",
        }
    )
    artifacts = {Phase.DEPLOYMENT.value: phase_artifacts}
    ctx = PhaseContext(
        user_id=uuid4(),
        session_artifacts=artifacts,
        turn_context={
            "mcp_tool_calls": [
                _mcp_success_call(
                    "trading_check_deployment_readiness",
                    {
                        "status": "no_broker",
                        "strategy_scope": {
                            "market": "crypto",
                            "symbols": ["BTCUSD"],
                        },
                        "accounts": [],
                        "matched_accounts": [],
                        "matched_broker_account_ids": [],
                        "blockers": ["No active paper broker account is connected."],
                    },
                )
            ]
        },
    )

    result = await handler.post_process(ctx, [], SimpleNamespace())
    updated = result.artifacts[Phase.DEPLOYMENT.value]

    assert result.completed is False
    assert result.next_phase is None
    assert result.phase_status["deployment_status"] == "blocked"
    assert result.missing_fields == ["selected_broker_account_id"]
    assert updated["profile"]["broker_readiness_status"] == "no_broker"

    choice_prompt = handler.build_fallback_choice_prompt(
        missing_fields=result.missing_fields,
        ctx=PhaseContext(user_id=ctx.user_id, session_artifacts=result.artifacts),
    )
    assert choice_prompt is not None
    assert choice_prompt["choice_id"] == "selected_broker_account_id"
    option_ids = {option["id"] for option in choice_prompt["options"]}
    assert "create_builtin_sandbox" in option_ids
    assert "open_broker_connectors" in option_ids


async def test_deployment_handler_auto_selects_single_ready_broker_then_waits_for_confirmation() -> None:
    handler = DeploymentHandler()
    broker_id = "6b3f4e8b-e0d4-4cb1-911f-0119985c31e8"
    artifacts = {
        Phase.DEPLOYMENT.value: {
            "profile": {
                "strategy_name": "SPY Trend",
                "strategy_market": "us_stocks",
                "strategy_primary_symbol": "SPY",
            },
            "missing_fields": list(handler.required_fields),
            "runtime": {},
        }
    }
    ctx = PhaseContext(
        user_id=uuid4(),
        session_artifacts=artifacts,
        turn_context={
            "mcp_tool_calls": [
                _mcp_success_call(
                    "trading_check_deployment_readiness",
                    {
                        "status": "ready",
                        "preferred_broker_account_id": broker_id,
                        "default_broker_account_id": broker_id,
                        "matched_broker_account_ids": [broker_id],
                        "matched_accounts": [
                            {
                                "broker_account_id": broker_id,
                                "label": "ALPACA",
                                "provider": "alpaca",
                                "exchange_id": "alpaca",
                                "is_default": True,
                            }
                        ],
                        "accounts": [
                            {
                                "broker_account_id": broker_id,
                                "label": "ALPACA",
                                "provider": "alpaca",
                                "exchange_id": "alpaca",
                                "is_default": True,
                            }
                        ],
                        "blockers": [],
                    },
                )
            ]
        },
    )

    result = await handler.post_process(ctx, [], SimpleNamespace())
    updated = result.artifacts[Phase.DEPLOYMENT.value]

    assert updated["profile"]["selected_broker_account_id"] == broker_id
    assert updated["profile"]["selected_broker_label"] == "ALPACA"
    assert updated["profile"]["broker_readiness_status"] == "ready"
    assert updated["profile"]["deployment_status"] == "ready"
    assert result.missing_fields == ["deployment_confirmation_status"]

    choice_prompt = handler.build_fallback_choice_prompt(
        missing_fields=result.missing_fields,
        ctx=PhaseContext(user_id=ctx.user_id, session_artifacts=result.artifacts),
    )
    assert choice_prompt is not None
    assert choice_prompt["choice_id"] == "deployment_confirmation_status"
    option_ids = {option["id"] for option in choice_prompt["options"]}
    assert option_ids == {"confirmed", "needs_changes"}


async def test_deployment_handler_requires_explicit_choice_when_multiple_brokers_match() -> None:
    handler = DeploymentHandler()
    broker_one = "6b3f4e8b-e0d4-4cb1-911f-0119985c31e8"
    broker_two = "ca4084b8-d30c-44ff-b278-cba45fd01332"
    artifacts = {
        Phase.DEPLOYMENT.value: {
            "profile": {
                "strategy_name": "BTC Breakout",
                "strategy_market": "crypto",
                "strategy_primary_symbol": "BTCUSD",
            },
            "missing_fields": list(handler.required_fields),
            "runtime": {},
        }
    }
    ctx = PhaseContext(
        user_id=uuid4(),
        session_artifacts=artifacts,
        turn_context={
            "mcp_tool_calls": [
                _mcp_success_call(
                    "trading_check_deployment_readiness",
                    {
                        "status": "needs_choice",
                        "preferred_broker_account_id": broker_one,
                        "matched_broker_account_ids": [broker_one, broker_two],
                        "matched_accounts": [
                            {
                                "broker_account_id": broker_one,
                                "label": "CCXT / BINANCE",
                                "provider": "ccxt",
                                "exchange_id": "binance",
                                "is_default": True,
                            },
                            {
                                "broker_account_id": broker_two,
                                "label": "Built-in Sandbox",
                                "provider": "sandbox",
                                "exchange_id": "sandbox",
                                "is_default": False,
                            },
                        ],
                        "accounts": [],
                        "blockers": [],
                    },
                )
            ]
        },
    )

    result = await handler.post_process(ctx, [], SimpleNamespace())

    assert result.missing_fields == ["selected_broker_account_id"]
    updated = result.artifacts[Phase.DEPLOYMENT.value]
    assert updated["profile"].get("selected_broker_account_id") is None
    assert updated["profile"]["broker_readiness_status"] == "needs_choice"

    choice_prompt = handler.build_fallback_choice_prompt(
        missing_fields=result.missing_fields,
        ctx=PhaseContext(user_id=ctx.user_id, session_artifacts=result.artifacts),
    )
    assert choice_prompt is not None
    assert choice_prompt["choice_id"] == "selected_broker_account_id"
    assert len(choice_prompt["options"]) == 2


async def test_deployment_handler_sets_auto_execute_after_confirmation() -> None:
    handler = DeploymentHandler()
    broker_id = "6b3f4e8b-e0d4-4cb1-911f-0119985c31e8"
    artifacts = {
        Phase.DEPLOYMENT.value: {
            "profile": {
                "strategy_name": "ETH Mean Reversion",
                "strategy_market": "crypto",
                "strategy_primary_symbol": "ETHUSD",
                "broker_readiness_status": "ready",
                "selected_broker_account_id": broker_id,
                "selected_broker_label": "Built-in Sandbox",
                "deployment_status": "ready",
                "deployment_confirmation_status": "pending",
            },
            "missing_fields": ["deployment_confirmation_status"],
            "runtime": {},
        }
    }
    ctx = PhaseContext(
        user_id=uuid4(),
        session_artifacts=artifacts,
        turn_context={"mcp_tool_calls": []},
    )

    result = await handler.post_process(
        ctx,
        [{"deployment_confirmation_status": "confirmed"}],
        SimpleNamespace(),
    )

    updated = result.artifacts[Phase.DEPLOYMENT.value]
    assert result.missing_fields == []
    assert updated["profile"]["deployment_confirmation_status"] == "confirmed"
    assert updated["runtime"]["auto_execute_pending"] is True
    assert updated["runtime"]["deployment_summary_snapshot"]["selected_broker"] == (
        "Built-in Sandbox"
    )


async def test_deployment_handler_marks_deployed_from_flat_mcp_payload() -> None:
    handler = DeploymentHandler()
    broker_id = "6b3f4e8b-e0d4-4cb1-911f-0119985c31e8"
    deployment_id = "ca4084b8-d30c-44ff-b278-cba45fd01332"
    artifacts = {
        Phase.DEPLOYMENT.value: {
            "profile": {
                "strategy_name": "ETH Mean Reversion",
                "strategy_market": "crypto",
                "strategy_primary_symbol": "ETHUSD",
                "broker_readiness_status": "ready",
                "selected_broker_account_id": broker_id,
                "selected_broker_label": "Built-in Sandbox",
                "deployment_status": "ready",
                "deployment_confirmation_status": "confirmed",
            },
            "missing_fields": [],
            "runtime": {},
        }
    }
    ctx = PhaseContext(
        user_id=uuid4(),
        session_artifacts=artifacts,
        turn_context={
            "mcp_tool_calls": [
                _mcp_flat_success_call(
                    "trading_create_paper_deployment",
                    {
                        "deployment": {
                            "deployment_id": deployment_id,
                            "status": "active",
                        },
                        "resolved_broker_account_id": broker_id,
                    },
                )
            ]
        },
    )

    result = await handler.post_process(ctx, [], SimpleNamespace())

    updated = result.artifacts[Phase.DEPLOYMENT.value]
    assert updated["profile"]["deployment_status"] == "deployed"
    assert updated["profile"]["deployment_confirmation_status"] == "confirmed"
    assert updated["profile"]["latest_deployment_id"] == deployment_id
    assert updated["runtime"]["deployment_status"] == "deployed"
    assert updated["runtime"]["latest_deployment_id"] == deployment_id
    assert updated["runtime"]["active_deployments"] == [
        {"deployment_id": deployment_id, "status": "active"}
    ]


async def test_deployment_handler_choice_selection_can_create_builtin_sandbox(
    monkeypatch,
) -> None:
    handler = DeploymentHandler()
    broker_id = "6b3f4e8b-e0d4-4cb1-911f-0119985c31e8"

    async def _fake_ensure_builtin_sandbox_account(db, *, user_id):
        del db, user_id
        return SimpleNamespace(
            id=broker_id,
            status="active",
            is_default=False,
            capabilities={"supported_markets": ["us_stocks", "crypto"]},
        )

    monkeypatch.setattr(
        deployment_handler_module,
        "ensure_builtin_sandbox_account",
        _fake_ensure_builtin_sandbox_account,
    )

    artifacts = {
        Phase.DEPLOYMENT.value: {
            "profile": {
                "strategy_name": "SPY Trend",
                "strategy_market": "us_stocks",
                "strategy_primary_symbol": "SPY",
                "broker_readiness_status": "no_broker",
                "deployment_status": "blocked",
            },
            "missing_fields": ["selected_broker_account_id"],
            "runtime": {},
        }
    }
    ctx = PhaseContext(
        user_id=uuid4(),
        session_artifacts=artifacts,
        turn_context={
            "choice_selection": {
                "choice_id": "selected_broker_account_id",
                "selected_option_id": "create_builtin_sandbox",
                "selected_option_label": "Use built-in sandbox",
            },
            "mcp_tool_calls": [],
        },
    )

    result = await handler.post_process(ctx, [], SimpleNamespace())

    updated = result.artifacts[Phase.DEPLOYMENT.value]
    assert updated["profile"]["selected_broker_account_id"] == broker_id
    assert updated["profile"]["selected_broker_label"] == "Built-in Sandbox"
    assert updated["profile"]["broker_readiness_status"] == "ready"
    assert result.missing_fields == ["deployment_confirmation_status"]


async def test_deployment_handler_choice_selection_confirm_sets_confirmed_and_auto_execute() -> None:
    handler = DeploymentHandler()
    broker_id = "6b3f4e8b-e0d4-4cb1-911f-0119985c31e8"
    artifacts = {
        Phase.DEPLOYMENT.value: {
            "profile": {
                "strategy_name": "ETH Mean Reversion",
                "strategy_market": "crypto",
                "strategy_primary_symbol": "ETHUSD",
                "broker_readiness_status": "ready",
                "selected_broker_account_id": broker_id,
                "selected_broker_label": "Built-in Sandbox",
                "deployment_status": "ready",
                "deployment_confirmation_status": "pending",
            },
            "missing_fields": ["deployment_confirmation_status"],
            "runtime": {},
        }
    }
    ctx = PhaseContext(
        user_id=uuid4(),
        session_artifacts=artifacts,
        turn_context={
            "choice_selection": {
                "choice_id": "deployment_confirmation_status",
                "selected_option_id": "confirmed",
                "selected_option_label": "Confirm deployment",
            },
            "mcp_tool_calls": [],
        },
    )

    result = await handler.post_process(ctx, [], SimpleNamespace())

    updated = result.artifacts[Phase.DEPLOYMENT.value]
    assert result.missing_fields == []
    assert updated["profile"]["deployment_confirmation_status"] == "confirmed"
    assert updated["runtime"]["auto_execute_pending"] is True
