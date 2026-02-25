from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from uuid import uuid4

import pytest
from mcp.server.fastmcp import FastMCP
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.engine.strategy import EXAMPLE_PATH, load_strategy_payload, upsert_strategy_dsl
from src.mcp.context_auth import McpContextClaims
from src.mcp.trading import tools as trading_tools
from src.models.broker_account import BrokerAccount
from src.models.deployment import Deployment
from src.models.deployment_run import DeploymentRun
from src.models.order import Order
from src.models.position import Position
from src.models.session import Session as AgentSession
from src.models.user import User


class _SessionContext:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def __aenter__(self) -> AsyncSession:
        return self._session

    async def __aexit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001
        return False


def _extract_payload(call_result: object) -> dict[str, Any]:
    if isinstance(call_result, tuple) and len(call_result) == 2:
        maybe_result = call_result[1]
        if isinstance(maybe_result, dict):
            raw = maybe_result.get("result")
            if isinstance(raw, str):
                return json.loads(raw)
    raise AssertionError(f"Unexpected call result: {call_result!r}")


def _claims_for(*, user_id: Any, session_id: Any | None) -> McpContextClaims:
    now = datetime.now(UTC)
    return McpContextClaims(
        user_id=user_id,
        session_id=session_id,
        issued_at=now,
        expires_at=now + timedelta(minutes=5),
        trace_id="test-trace",
        phase="deployment",
    )


async def _create_fixture_bundle(
    db_session: AsyncSession,
    *,
    email: str,
    with_runtime_rows: bool = True,
    create_deployment_row: bool = True,
) -> dict[str, Any]:
    user = User(email=email, password_hash="hash", name=email)
    db_session.add(user)
    await db_session.flush()

    session = AgentSession(
        user_id=user.id,
        current_phase="deployment",
        status="active",
        artifacts={},
        metadata_={},
    )
    db_session.add(session)
    await db_session.flush()

    payload = load_strategy_payload(EXAMPLE_PATH)
    payload["universe"] = {"market": "stocks", "tickers": ["AAPL"]}
    payload["timeframe"] = "1m"
    persistence = await upsert_strategy_dsl(
        db_session,
        session_id=session.id,
        dsl_payload=payload,
        auto_commit=False,
    )
    strategy = persistence.strategy

    broker_account = BrokerAccount(
        user_id=user.id,
        provider="alpaca",
        mode="paper",
        encrypted_credentials="enc.test.credentials",
        status="active",
        metadata_={},
        validation_metadata={},
    )
    db_session.add(broker_account)
    await db_session.flush()

    deployment: Deployment | None = None
    if create_deployment_row:
        deployment = Deployment(
            strategy_id=strategy.id,
            user_id=user.id,
            mode="paper",
            status="pending",
            risk_limits={},
            capital_allocated=Decimal("1000"),
        )
        db_session.add(deployment)
        await db_session.flush()

        deployment_run = DeploymentRun(
            deployment_id=deployment.id,
            strategy_id=strategy.id,
            broker_account_id=broker_account.id,
            status="stopped",
            runtime_state={},
        )
        db_session.add(deployment_run)
        await db_session.flush()

        if with_runtime_rows:
            order = Order(
                deployment_id=deployment.id,
                provider_order_id=f"paper-{uuid4().hex}",
                client_order_id=f"coid-{uuid4().hex}",
                symbol="AAPL",
                side="buy",
                type="market",
                qty=Decimal("1"),
                price=Decimal("100"),
                status="accepted",
                metadata_={},
                submitted_at=datetime.now(UTC),
            )
            position = Position(
                deployment_id=deployment.id,
                symbol="AAPL",
                side="long",
                qty=Decimal("1"),
                avg_entry_price=Decimal("100"),
                mark_price=Decimal("101"),
                unrealized_pnl=Decimal("1"),
                realized_pnl=Decimal("0"),
            )
            db_session.add(order)
            db_session.add(position)

    await db_session.commit()
    return {
        "user": user,
        "session": session,
        "deployment": deployment,
        "strategy": strategy,
        "broker_account": broker_account,
    }


@pytest.mark.asyncio
async def test_trading_tools_list_and_lifecycle(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = await _create_fixture_bundle(
        db_session,
        email="trading_mcp_lifecycle@example.com",
        with_runtime_rows=False,
    )

    async def _fake_new_db_session() -> _SessionContext:
        return _SessionContext(db_session)

    monkeypatch.setattr(trading_tools, "_new_db_session", _fake_new_db_session)
    monkeypatch.setattr(
        trading_tools,
        "_resolve_context_claims",
        lambda _ctx: _claims_for(
            user_id=bundle["user"].id,
            session_id=bundle["session"].id,
        ),
    )
    monkeypatch.setattr(settings, "paper_trading_enqueue_on_start", False)

    mcp = FastMCP("test-trading-tools-lifecycle")
    trading_tools.register_trading_tools(mcp)

    listed = _extract_payload(await mcp.call_tool("trading_list_deployments", {}))
    assert listed["ok"] is True
    assert listed["count"] == 1
    assert listed["deployments"][0]["status"] == "pending"

    deployment_id = str(bundle["deployment"].id)
    started = _extract_payload(
        await mcp.call_tool(
            "trading_start_deployment",
            {"deployment_id": deployment_id},
        )
    )
    assert started["ok"] is True
    assert started["deployment"]["status"] == "active"
    assert started["deployment"]["run"]["status"] == "starting"

    paused = _extract_payload(
        await mcp.call_tool(
            "trading_pause_deployment",
            {"deployment_id": deployment_id},
        )
    )
    assert paused["ok"] is True
    assert paused["deployment"]["status"] == "paused"
    assert paused["deployment"]["run"]["status"] == "paused"

    stopped = _extract_payload(
        await mcp.call_tool(
            "trading_stop_deployment",
            {"deployment_id": deployment_id},
        )
    )
    assert stopped["ok"] is True
    assert stopped["deployment"]["status"] == "stopped"
    assert stopped["deployment"]["run"]["status"] == "stopped"


@pytest.mark.asyncio
async def test_trading_tools_orders_positions_and_invalid_uuid(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = await _create_fixture_bundle(
        db_session,
        email="trading_mcp_rows@example.com",
        with_runtime_rows=True,
    )

    async def _fake_new_db_session() -> _SessionContext:
        return _SessionContext(db_session)

    monkeypatch.setattr(trading_tools, "_new_db_session", _fake_new_db_session)
    monkeypatch.setattr(
        trading_tools,
        "_resolve_context_claims",
        lambda _ctx: _claims_for(
            user_id=bundle["user"].id,
            session_id=bundle["session"].id,
        ),
    )

    mcp = FastMCP("test-trading-tools-rows")
    trading_tools.register_trading_tools(mcp)
    deployment_id = str(bundle["deployment"].id)

    orders = _extract_payload(
        await mcp.call_tool(
            "trading_get_orders",
            {"deployment_id": deployment_id},
        )
    )
    assert orders["ok"] is True
    assert orders["count"] == 1
    assert orders["orders"][0]["symbol"] == "AAPL"

    positions = _extract_payload(
        await mcp.call_tool(
            "trading_get_positions",
            {"deployment_id": deployment_id},
        )
    )
    assert positions["ok"] is True
    assert positions["count"] == 1
    assert positions["positions"][0]["symbol"] == "AAPL"

    invalid_uuid = _extract_payload(
        await mcp.call_tool(
            "trading_get_orders",
            {"deployment_id": "not-a-uuid"},
        )
    )
    assert invalid_uuid["ok"] is False
    assert invalid_uuid["error"]["code"] == "INVALID_UUID"


@pytest.mark.asyncio
async def test_trading_create_paper_deployment_auto_resolves_context(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = await _create_fixture_bundle(
        db_session,
        email="trading_mcp_create_auto@example.com",
        with_runtime_rows=False,
        create_deployment_row=False,
    )

    async def _fake_new_db_session() -> _SessionContext:
        return _SessionContext(db_session)

    monkeypatch.setattr(trading_tools, "_new_db_session", _fake_new_db_session)
    monkeypatch.setattr(
        trading_tools,
        "_resolve_context_claims",
        lambda _ctx: _claims_for(
            user_id=bundle["user"].id,
            session_id=bundle["session"].id,
        ),
    )

    mcp = FastMCP("test-trading-create-auto")
    trading_tools.register_trading_tools(mcp)

    created = _extract_payload(
        await mcp.call_tool(
            "trading_create_paper_deployment",
            {
                "capital_allocated": "15000",
                "risk_limits": {"order_qty": 0.5},
                "runtime_state": {"source": "mcp_test"},
            },
        )
    )
    assert created["ok"] is True
    assert created["auto_started"] is False
    assert created["resolved_strategy_id"] == str(bundle["strategy"].id)
    assert created["resolved_broker_account_id"] == str(bundle["broker_account"].id)
    assert created["deployment"]["status"] == "pending"
    assert created["deployment"]["run"]["status"] == "stopped"


@pytest.mark.asyncio
async def test_trading_create_paper_deployment_can_auto_start(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = await _create_fixture_bundle(
        db_session,
        email="trading_mcp_create_start@example.com",
        with_runtime_rows=False,
        create_deployment_row=False,
    )

    async def _fake_new_db_session() -> _SessionContext:
        return _SessionContext(db_session)

    monkeypatch.setattr(trading_tools, "_new_db_session", _fake_new_db_session)
    monkeypatch.setattr(
        trading_tools,
        "_resolve_context_claims",
        lambda _ctx: _claims_for(
            user_id=bundle["user"].id,
            session_id=bundle["session"].id,
        ),
    )
    monkeypatch.setattr(settings, "paper_trading_enqueue_on_start", False)

    mcp = FastMCP("test-trading-create-start")
    trading_tools.register_trading_tools(mcp)

    created = _extract_payload(
        await mcp.call_tool(
            "trading_create_paper_deployment",
            {
                "strategy_id": str(bundle["strategy"].id),
                "broker_account_id": str(bundle["broker_account"].id),
                "auto_start": True,
            },
        )
    )
    assert created["ok"] is True
    assert created["auto_started"] is True
    assert created["deployment"]["status"] == "active"
    assert created["deployment"]["run"]["status"] == "starting"


@pytest.mark.asyncio
async def test_trading_tools_require_context(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = await _create_fixture_bundle(
        db_session,
        email="trading_mcp_missing_context@example.com",
        with_runtime_rows=False,
    )

    async def _fake_new_db_session() -> _SessionContext:
        return _SessionContext(db_session)

    monkeypatch.setattr(trading_tools, "_new_db_session", _fake_new_db_session)
    monkeypatch.setattr(trading_tools, "_resolve_context_claims", lambda _ctx: None)

    mcp = FastMCP("test-trading-tools-missing-context")
    trading_tools.register_trading_tools(mcp)

    missing_context = _extract_payload(await mcp.call_tool("trading_list_deployments", {}))
    assert missing_context["ok"] is False
    assert missing_context["error"]["code"] == "MISSING_CONTEXT"
    assert str(bundle["deployment"].id)


@pytest.mark.asyncio
async def test_trading_tools_reject_foreign_deployment_access(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner_bundle = await _create_fixture_bundle(
        db_session,
        email="trading_mcp_owner@example.com",
        with_runtime_rows=True,
    )
    outsider = User(
        email="trading_mcp_outsider@example.com",
        password_hash="hash",
        name="outsider",
    )
    db_session.add(outsider)
    await db_session.commit()

    async def _fake_new_db_session() -> _SessionContext:
        return _SessionContext(db_session)

    monkeypatch.setattr(trading_tools, "_new_db_session", _fake_new_db_session)
    monkeypatch.setattr(
        trading_tools,
        "_resolve_context_claims",
        lambda _ctx: _claims_for(
            user_id=outsider.id,
            session_id=None,
        ),
    )

    mcp = FastMCP("test-trading-tools-foreign-owner")
    trading_tools.register_trading_tools(mcp)

    forbidden = _extract_payload(
        await mcp.call_tool(
            "trading_get_orders",
            {"deployment_id": str(owner_bundle["deployment"].id)},
        )
    )
    assert forbidden["ok"] is False
    assert forbidden["error"]["code"] == "DEPLOYMENT_NOT_FOUND"
