from __future__ import annotations

import json
from decimal import Decimal
from types import SimpleNamespace
from uuid import uuid4

from apps.mcp.domains.trading import tools as trading_tools


class _FakeDbSession:
    async def __aenter__(self) -> "_FakeDbSession":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        del exc_type, exc, tb
        return False

    async def scalar(self, stmt):
        del stmt
        return SimpleNamespace(id=uuid4())


async def _fake_new_db_session() -> _FakeDbSession:
    return _FakeDbSession()


def _fake_broker_account(
    *,
    provider: str,
    supported_markets: list[str],
    is_default: bool,
) -> SimpleNamespace:
    broker_id = uuid4()
    exchange_id = "sandbox" if provider == "sandbox" else provider
    return SimpleNamespace(
        id=broker_id,
        provider=provider,
        exchange_id=exchange_id,
        account_uid=f"{provider}-{broker_id.hex[:8]}",
        mode="paper",
        status="active",
        is_default=is_default,
        is_sandbox=(provider == "sandbox"),
        capabilities={"supported_markets": supported_markets},
        metadata_={},
        validation_metadata={},
    )


async def test_trading_list_broker_accounts_returns_default_broker_and_capabilities(
    monkeypatch,
) -> None:
    claims = SimpleNamespace(user_id=uuid4(), session_id=uuid4())
    alpaca = _fake_broker_account(
        provider="alpaca",
        supported_markets=["us_stocks", "crypto"],
        is_default=True,
    )
    sandbox = _fake_broker_account(
        provider="sandbox",
        supported_markets=["us_stocks", "crypto"],
        is_default=False,
    )

    monkeypatch.setattr(trading_tools, "_resolve_context_claims", lambda ctx: claims)
    monkeypatch.setattr(trading_tools, "_new_db_session", _fake_new_db_session)

    async def _fake_list_user_broker_accounts(db, *, user_id, active_only=False):
        del db, user_id, active_only
        return [alpaca, sandbox]

    monkeypatch.setattr(
        trading_tools,
        "list_user_broker_accounts",
        _fake_list_user_broker_accounts,
    )

    result = await trading_tools.trading_list_broker_accounts(ctx=object())
    payload = json.loads(result)

    assert payload["ok"] is True
    assert payload["count"] == 2
    assert payload["default_broker_account_id"] == str(alpaca.id)
    assert payload["accounts"][0]["derived_capabilities"]["supports_us_stocks"] is True


async def test_trading_check_deployment_readiness_reports_needs_choice_with_multiple_matches(
    monkeypatch,
) -> None:
    claims = SimpleNamespace(user_id=uuid4(), session_id=uuid4())
    broker_one = _fake_broker_account(
        provider="ccxt",
        supported_markets=["crypto"],
        is_default=True,
    )
    broker_two = _fake_broker_account(
        provider="sandbox",
        supported_markets=["us_stocks", "crypto"],
        is_default=False,
    )

    monkeypatch.setattr(trading_tools, "_resolve_context_claims", lambda ctx: claims)
    monkeypatch.setattr(trading_tools, "_new_db_session", _fake_new_db_session)

    async def _fake_load_owned_strategy_for_context(
        *,
        db,
        user_id,
        session_id,
        strategy_id,
    ):
        del db, user_id, session_id, strategy_id
        return (
            SimpleNamespace(
                id=uuid4(),
                name="BTC Momentum",
                timeframe="1h",
                symbols=["BTCUSD"],
                dsl_payload={
                    "universe": {
                        "market": "crypto",
                        "tickers": ["BTCUSD"],
                    },
                    "timeframe": "1h",
                },
            ),
            None,
        )

    async def _fake_list_user_broker_accounts(db, *, user_id, active_only=False):
        del db, user_id, active_only
        return [broker_one, broker_two]

    monkeypatch.setattr(
        trading_tools,
        "_load_owned_strategy_for_context",
        _fake_load_owned_strategy_for_context,
    )
    monkeypatch.setattr(
        trading_tools,
        "list_user_broker_accounts",
        _fake_list_user_broker_accounts,
    )

    result = await trading_tools.trading_check_deployment_readiness(ctx=object())
    payload = json.loads(result)

    assert payload["ok"] is True
    assert payload["status"] == "needs_choice"
    assert len(payload["matched_accounts"]) == 2
    assert payload["strategy_scope"]["market"] == "crypto"


async def test_trading_create_builtin_sandbox_returns_structured_broker_summary(
    monkeypatch,
) -> None:
    claims = SimpleNamespace(user_id=uuid4(), session_id=uuid4())
    sandbox = _fake_broker_account(
        provider="sandbox",
        supported_markets=["us_stocks", "crypto"],
        is_default=True,
    )

    monkeypatch.setattr(trading_tools, "_resolve_context_claims", lambda ctx: claims)
    monkeypatch.setattr(trading_tools, "_new_db_session", _fake_new_db_session)

    async def _fake_ensure_builtin_sandbox_account(db, *, user_id):
        del db, user_id
        return sandbox

    monkeypatch.setattr(
        trading_tools,
        "ensure_builtin_sandbox_account",
        _fake_ensure_builtin_sandbox_account,
    )

    result = await trading_tools.trading_create_builtin_sandbox_broker_account(ctx=object())
    payload = json.loads(result)

    assert payload["ok"] is True
    assert payload["broker_account"]["broker_account_id"] == str(sandbox.id)
    assert payload["broker_account"]["label"] == "Built-in Sandbox"


async def test_trading_create_paper_deployment_defaults_capital_to_auto_resolution(
    monkeypatch,
) -> None:
    claims = SimpleNamespace(user_id=uuid4(), session_id=uuid4())
    captured: dict[str, Decimal] = {}
    strategy_id = uuid4()
    broker_account_id = uuid4()
    deployment_id = uuid4()

    monkeypatch.setattr(trading_tools, "_resolve_context_claims", lambda ctx: claims)
    monkeypatch.setattr(trading_tools, "_new_db_session", _fake_new_db_session)

    async def _fake_resolve_strategy_id_for_context(
        *,
        db,
        user_id,
        session_id,
        strategy_id,
    ):
        del db, user_id, session_id, strategy_id
        return (claims.session_id, None)

    async def _fake_resolve_broker_account_id_for_context(
        *,
        db,
        user_id,
        broker_account_id,
    ):
        del db, user_id, broker_account_id
        return (claims.user_id, None)

    async def _fake_create_deployment(
        *,
        db,
        strategy_id,
        broker_account_id,
        user_id,
        mode,
        capital_allocated,
        risk_limits,
        runtime_state,
    ):
        del db, strategy_id, broker_account_id, user_id, mode, risk_limits, runtime_state
        captured["capital_allocated"] = capital_allocated
        return SimpleNamespace(id=deployment_id)

    monkeypatch.setattr(
        trading_tools,
        "_resolve_strategy_id_for_context",
        _fake_resolve_strategy_id_for_context,
    )
    monkeypatch.setattr(
        trading_tools,
        "_resolve_broker_account_id_for_context",
        _fake_resolve_broker_account_id_for_context,
    )
    monkeypatch.setattr(trading_tools.deployment_ops, "create_deployment", _fake_create_deployment)
    monkeypatch.setattr(
        trading_tools.deployment_ops,
        "serialize_deployment",
        lambda deployment: {"deployment_id": str(deployment.id)},
    )

    result = await trading_tools.trading_create_paper_deployment(ctx=object())
    payload = json.loads(result)

    assert payload["ok"] is True
    assert payload["resolved_strategy_id"] == str(claims.session_id)
    assert payload["resolved_broker_account_id"] == str(claims.user_id)
    assert captured["capital_allocated"] == Decimal("0")


async def test_trading_create_builtin_sandbox_accepts_account_config_patch(
    monkeypatch,
) -> None:
    claims = SimpleNamespace(user_id=uuid4(), session_id=uuid4())
    sandbox = _fake_broker_account(
        provider="sandbox",
        supported_markets=["us_stocks", "crypto"],
        is_default=True,
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(trading_tools, "_resolve_context_claims", lambda ctx: claims)
    monkeypatch.setattr(trading_tools, "_new_db_session", _fake_new_db_session)

    async def _fake_ensure_builtin_sandbox_account(db, *, user_id, metadata=None):
        del db, user_id
        captured["metadata"] = metadata
        sandbox.metadata_ = metadata or {}
        return sandbox

    monkeypatch.setattr(
        trading_tools,
        "ensure_builtin_sandbox_account",
        _fake_ensure_builtin_sandbox_account,
    )

    result = await trading_tools.trading_create_builtin_sandbox_broker_account(
        starting_cash="25000",
        fee_bps="3",
        slippage_bps="1.5",
        ctx=object(),
    )
    payload = json.loads(result)

    assert payload["ok"] is True
    assert captured["metadata"] == {
        "starting_cash": "25000",
        "fee_bps": "3",
        "slippage_bps": "1.5",
    }
