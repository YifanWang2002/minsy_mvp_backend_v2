"""Trading MCP tools for deployment lifecycle and runtime visibility."""

from __future__ import annotations

from decimal import Decimal
from typing import Any
from uuid import UUID

from fastapi import HTTPException
from mcp.server.fastmcp import Context, FastMCP
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from src.api.routers import deployments as deployments_router
from src.api.schemas.requests import DeploymentCreateRequest
from src.config import settings
from src.mcp._utils import log_mcp_tool_result, to_json, utc_now_iso
from src.mcp.context_auth import (
    McpContextClaims,
    decode_mcp_context_token,
    extract_mcp_context_token,
)
from src.models import database as db_module
from src.models.broker_account import BrokerAccount
from src.models.deployment import Deployment
from src.models.order import Order
from src.models.position import Position
from src.models.strategy import Strategy
from src.models.user import User
from src.workers.paper_trading_tasks import enqueue_paper_trading_runtime

_VALID_DEPLOYMENT_STATUS: frozenset[str] = frozenset(
    {"pending", "active", "paused", "stopped", "error"}
)

TOOL_NAMES: tuple[str, ...] = (
    "trading_ping",
    "trading_capabilities",
    "trading_create_paper_deployment",
    "trading_list_deployments",
    "trading_start_deployment",
    "trading_pause_deployment",
    "trading_stop_deployment",
    "trading_get_positions",
    "trading_get_orders",
)


def _payload(
    *,
    tool: str,
    ok: bool,
    data: dict[str, Any] | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
) -> str:
    body: dict[str, Any] = {
        "category": "trading",
        "tool": tool,
        "ok": ok,
        "timestamp_utc": utc_now_iso(),
    }
    resolved_error_code: str | None = None
    resolved_error_message: str | None = None
    if isinstance(data, dict) and data:
        body.update(data)
    if not ok:
        resolved_error_code = error_code or "UNKNOWN_ERROR"
        resolved_error_message = error_message or "Unknown error"
        body["error"] = {
            "code": resolved_error_code,
            "message": resolved_error_message,
        }
    log_mcp_tool_result(
        category="trading",
        tool=tool,
        ok=ok,
        error_code=resolved_error_code,
        error_message=resolved_error_message,
    )
    return to_json(body)


def _parse_uuid(value: str, field_name: str) -> UUID:
    try:
        return UUID(value)
    except ValueError as exc:
        raise ValueError(f"Invalid {field_name}: {value}") from exc


def _serialize_model(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json")
    if hasattr(model, "dict"):
        return model.dict()
    if isinstance(model, dict):
        return dict(model)
    return {"value": str(model)}


def _resolve_context_claims(ctx: Context | None) -> McpContextClaims | None:
    if ctx is None:
        return None
    try:
        request = ctx.request_context.request
    except Exception:  # noqa: BLE001
        return None
    headers = getattr(request, "headers", None)
    token = extract_mcp_context_token(headers)
    if token is None:
        return None
    return decode_mcp_context_token(token)


def _require_context_user_id(
    *,
    tool: str,
    ctx: Context | None,
) -> tuple[UUID | None, str | None]:
    try:
        claims = _resolve_context_claims(ctx)
    except ValueError as exc:
        return None, _payload(
            tool=tool,
            ok=False,
            error_code="INVALID_INPUT",
            error_message=str(exc),
        )
    if claims is None:
        return None, _payload(
            tool=tool,
            ok=False,
            error_code="MISSING_CONTEXT",
            error_message=(
                "Missing MCP context token. "
                "This tool requires user-scoped context from orchestrator."
            ),
        )
    return claims.user_id, None


def _http_error_code_message(exc: HTTPException) -> tuple[str, str]:
    detail = exc.detail
    if isinstance(detail, dict):
        code_raw = detail.get("code")
        message_raw = detail.get("message")
        code = str(code_raw).strip() if code_raw else f"HTTP_{exc.status_code}"
        message = (
            str(message_raw).strip()
            if message_raw
            else str(detail).strip() or f"HTTP {exc.status_code}"
        )
        return code, message
    text = str(detail).strip()
    return f"HTTP_{exc.status_code}", text or f"HTTP {exc.status_code}"


async def _new_db_session():
    if db_module.AsyncSessionLocal is None:
        # MCP worker process only needs a ready pool; schema is managed by API startup/migrations.
        await db_module.init_postgres(ensure_schema=False)
    assert db_module.AsyncSessionLocal is not None
    return db_module.AsyncSessionLocal()


def _normalize_optional_dict(value: Any, *, field_name: str) -> tuple[dict[str, Any], str | None]:
    if value is None:
        return {}, None
    if isinstance(value, dict):
        return dict(value), None
    return {}, f"{field_name} must be a JSON object."


def _normalize_capital_allocated(value: Any) -> tuple[Decimal, str | None]:
    if value is None:
        return Decimal("10000"), None
    try:
        capital = Decimal(str(value))
    except Exception:  # noqa: BLE001
        return Decimal("0"), "capital_allocated must be a valid decimal number."
    if capital < 0:
        return Decimal("0"), "capital_allocated must be >= 0."
    return capital, None


async def _resolve_strategy_id_for_context(
    *,
    db: Any,
    user_id: UUID,
    session_id: UUID | None,
    strategy_id: str,
) -> tuple[UUID | None, str | None]:
    strategy_text = strategy_id.strip()
    if strategy_text:
        try:
            strategy_uuid = _parse_uuid(strategy_text, "strategy_id")
        except ValueError as exc:
            return None, str(exc)
        owned = await db.scalar(
            select(Strategy.id).where(
                Strategy.id == strategy_uuid,
                Strategy.user_id == user_id,
            )
        )
        if owned is None:
            return None, "Strategy not found for current user."
        return strategy_uuid, None

    if session_id is not None:
        session_scoped = await db.scalar(
            select(Strategy.id)
            .where(
                Strategy.user_id == user_id,
                Strategy.session_id == session_id,
            )
            .order_by(Strategy.updated_at.desc(), Strategy.created_at.desc())
            .limit(1)
        )
        if session_scoped is not None:
            return session_scoped, None

    latest = await db.scalar(
        select(Strategy.id)
        .where(Strategy.user_id == user_id)
        .order_by(Strategy.updated_at.desc(), Strategy.created_at.desc())
        .limit(1)
    )
    if latest is None:
        return None, "No strategy found for current user."
    return latest, None


async def _resolve_broker_account_id_for_context(
    *,
    db: Any,
    user_id: UUID,
    broker_account_id: str,
) -> tuple[UUID | None, str | None]:
    account_text = broker_account_id.strip()
    if account_text:
        try:
            account_uuid = _parse_uuid(account_text, "broker_account_id")
        except ValueError as exc:
            return None, str(exc)
        owned = await db.scalar(
            select(BrokerAccount.id).where(
                BrokerAccount.id == account_uuid,
                BrokerAccount.user_id == user_id,
                BrokerAccount.mode == "paper",
            )
        )
        if owned is None:
            return None, "Broker account not found for current user."
        return account_uuid, None

    candidate = await db.scalar(
        select(BrokerAccount.id)
        .where(
            BrokerAccount.user_id == user_id,
            BrokerAccount.mode == "paper",
            BrokerAccount.status == "active",
        )
        .order_by(BrokerAccount.updated_at.desc(), BrokerAccount.created_at.desc())
        .limit(1)
    )
    if candidate is None:
        return None, "No active paper broker account found for current user."
    return candidate, None


def trading_ping() -> str:
    """Health probe for trading MCP domain."""
    return _payload(
        tool="trading_ping",
        ok=True,
        data={"status": "ready"},
    )


def trading_capabilities() -> str:
    """Advertise currently available trading-domain capabilities."""
    return _payload(
        tool="trading_capabilities",
        ok=True,
        data={
            "available": True,
            "message": "Trading MCP tools are enabled for paper deployment workflows.",
            "supported_tools": list(TOOL_NAMES),
            "require_user_context": True,
            "supported_mode": "paper",
        },
    )


async def trading_create_paper_deployment(
    strategy_id: str = "",
    broker_account_id: str = "",
    capital_allocated: str = "10000",
    risk_limits: dict[str, Any] | None = None,
    runtime_state: dict[str, Any] | None = None,
    auto_start: bool = False,
    ctx: Context | None = None,
) -> str:
    """Create one paper deployment for current user (optionally auto-start)."""

    try:
        claims = _resolve_context_claims(ctx)
    except ValueError as exc:
        return _payload(
            tool="trading_create_paper_deployment",
            ok=False,
            error_code="INVALID_INPUT",
            error_message=str(exc),
        )
    if claims is None:
        return _payload(
            tool="trading_create_paper_deployment",
            ok=False,
            error_code="MISSING_CONTEXT",
            error_message=(
                "Missing MCP context token. "
                "This tool requires user-scoped context from orchestrator."
            ),
        )

    normalized_risk_limits, risk_error = _normalize_optional_dict(
        risk_limits,
        field_name="risk_limits",
    )
    if risk_error is not None:
        return _payload(
            tool="trading_create_paper_deployment",
            ok=False,
            error_code="INVALID_INPUT",
            error_message=risk_error,
        )

    normalized_runtime_state, runtime_error = _normalize_optional_dict(
        runtime_state,
        field_name="runtime_state",
    )
    if runtime_error is not None:
        return _payload(
            tool="trading_create_paper_deployment",
            ok=False,
            error_code="INVALID_INPUT",
            error_message=runtime_error,
        )

    normalized_capital, capital_error = _normalize_capital_allocated(capital_allocated)
    if capital_error is not None:
        return _payload(
            tool="trading_create_paper_deployment",
            ok=False,
            error_code="INVALID_INPUT",
            error_message=capital_error,
        )

    try:
        async with await _new_db_session() as db:
            user = await db.scalar(select(User).where(User.id == claims.user_id))
            if user is None:
                return _payload(
                    tool="trading_create_paper_deployment",
                    ok=False,
                    error_code="USER_NOT_FOUND",
                    error_message="User not found for MCP context.",
                )

            resolved_strategy_id, strategy_error = await _resolve_strategy_id_for_context(
                db=db,
                user_id=claims.user_id,
                session_id=claims.session_id,
                strategy_id=strategy_id,
            )
            if strategy_error is not None or resolved_strategy_id is None:
                return _payload(
                    tool="trading_create_paper_deployment",
                    ok=False,
                    error_code="STRATEGY_NOT_FOUND",
                    error_message=strategy_error or "Strategy not found.",
                )

            resolved_broker_account_id, account_error = (
                await _resolve_broker_account_id_for_context(
                    db=db,
                    user_id=claims.user_id,
                    broker_account_id=broker_account_id,
                )
            )
            if account_error is not None or resolved_broker_account_id is None:
                return _payload(
                    tool="trading_create_paper_deployment",
                    ok=False,
                    error_code="BROKER_ACCOUNT_NOT_FOUND",
                    error_message=account_error or "Broker account not found.",
                )

            create_payload = DeploymentCreateRequest(
                strategy_id=resolved_strategy_id,
                broker_account_id=resolved_broker_account_id,
                mode="paper",
                capital_allocated=normalized_capital,
                risk_limits=normalized_risk_limits,
                runtime_state=normalized_runtime_state,
            )
            created = await deployments_router.create_deployment(
                payload=create_payload,
                user=user,
                db=db,
            )
            deployment_payload = _serialize_model(created)

            queued_task_id: str | None = None
            started = False
            if bool(auto_start):
                loaded_deployment = await deployments_router._load_owned_deployment(
                    db,
                    deployment_id=created.deployment_id,
                    user_id=claims.user_id,
                )
                started_deployment = await deployments_router._apply_status_transition(
                    db,
                    deployment=loaded_deployment,
                    target_status="active",
                )
                deployment_payload = _serialize_model(
                    deployments_router._serialize_deployment(started_deployment)
                )
                if settings.paper_trading_enqueue_on_start:
                    queued_task_id = enqueue_paper_trading_runtime(started_deployment.id)
                started = True
    except HTTPException as exc:
        code, message = _http_error_code_message(exc)
        return _payload(
            tool="trading_create_paper_deployment",
            ok=False,
            error_code=code,
            error_message=message,
        )
    except Exception as exc:  # noqa: BLE001
        return _payload(
            tool="trading_create_paper_deployment",
            ok=False,
            error_code="TRADING_CREATE_DEPLOYMENT_ERROR",
            error_message=f"{type(exc).__name__}: {exc}",
        )

    return _payload(
        tool="trading_create_paper_deployment",
        ok=True,
        data={
            "deployment": deployment_payload,
            "auto_started": started,
            "queued_task_id": queued_task_id,
            "resolved_strategy_id": str(resolved_strategy_id),
            "resolved_broker_account_id": str(resolved_broker_account_id),
        },
    )


async def trading_list_deployments(
    status: str = "",
    limit: int = 50,
    ctx: Context | None = None,
) -> str:
    """List current user's deployments."""
    user_id, error_payload = _require_context_user_id(tool="trading_list_deployments", ctx=ctx)
    if error_payload is not None or user_id is None:
        return error_payload or _payload(
            tool="trading_list_deployments",
            ok=False,
            error_code="MISSING_CONTEXT",
            error_message="Missing user context.",
        )

    normalized_status = status.strip().lower()
    if normalized_status and normalized_status not in _VALID_DEPLOYMENT_STATUS:
        return _payload(
            tool="trading_list_deployments",
            ok=False,
            error_code="INVALID_INPUT",
            error_message=(
                f"Invalid status filter: {status}. "
                f"Use one of {sorted(_VALID_DEPLOYMENT_STATUS)}."
            ),
        )

    try:
        safe_limit = min(max(int(limit), 1), 200)
    except (TypeError, ValueError):
        safe_limit = 50

    try:
        async with await _new_db_session() as db:
            stmt = (
                select(Deployment)
                .options(
                    selectinload(Deployment.deployment_runs),
                    selectinload(Deployment.strategy),
                )
                .where(Deployment.user_id == user_id)
                .order_by(Deployment.created_at.desc())
                .limit(safe_limit)
            )
            if normalized_status:
                stmt = stmt.where(Deployment.status == normalized_status)
            rows = list((await db.scalars(stmt)).all())
            deployments = [
                _serialize_model(deployments_router._serialize_deployment(row))
                for row in rows
            ]
    except HTTPException as exc:
        code, message = _http_error_code_message(exc)
        return _payload(
            tool="trading_list_deployments",
            ok=False,
            error_code=code,
            error_message=message,
        )
    except Exception as exc:  # noqa: BLE001
        return _payload(
            tool="trading_list_deployments",
            ok=False,
            error_code="TRADING_LIST_ERROR",
            error_message=f"{type(exc).__name__}: {exc}",
        )

    return _payload(
        tool="trading_list_deployments",
        ok=True,
        data={
            "count": len(deployments),
            "deployments": deployments,
        },
    )


async def _transition_deployment(
    *,
    tool: str,
    deployment_id: str,
    target_status: str,
    ctx: Context | None,
) -> str:
    try:
        deployment_uuid = _parse_uuid(deployment_id, "deployment_id")
    except ValueError as exc:
        return _payload(
            tool=tool,
            ok=False,
            error_code="INVALID_UUID",
            error_message=str(exc),
        )

    user_id, error_payload = _require_context_user_id(tool=tool, ctx=ctx)
    if error_payload is not None or user_id is None:
        return error_payload or _payload(
            tool=tool,
            ok=False,
            error_code="MISSING_CONTEXT",
            error_message="Missing user context.",
        )

    try:
        async with await _new_db_session() as db:
            deployment = await deployments_router._load_owned_deployment(
                db,
                deployment_id=deployment_uuid,
                user_id=user_id,
            )
            updated = await deployments_router._apply_status_transition(
                db,
                deployment=deployment,
                target_status=target_status,
            )

            queued_task_id: str | None = None
            if target_status == "active" and settings.paper_trading_enqueue_on_start:
                queued_task_id = enqueue_paper_trading_runtime(updated.id)
            deployment_payload = _serialize_model(
                deployments_router._serialize_deployment(updated)
            )
    except HTTPException as exc:
        code, message = _http_error_code_message(exc)
        return _payload(
            tool=tool,
            ok=False,
            error_code=code,
            error_message=message,
        )
    except Exception as exc:  # noqa: BLE001
        return _payload(
            tool=tool,
            ok=False,
            error_code="TRADING_TRANSITION_ERROR",
            error_message=f"{type(exc).__name__}: {exc}",
        )

    return _payload(
        tool=tool,
        ok=True,
        data={
            "deployment": deployment_payload,
            "queued_task_id": queued_task_id,
        },
    )


async def trading_start_deployment(
    deployment_id: str,
    ctx: Context | None = None,
) -> str:
    """Start one deployment (pending/paused -> active)."""
    return await _transition_deployment(
        tool="trading_start_deployment",
        deployment_id=deployment_id,
        target_status="active",
        ctx=ctx,
    )


async def trading_pause_deployment(
    deployment_id: str,
    ctx: Context | None = None,
) -> str:
    """Pause one deployment (active -> paused)."""
    return await _transition_deployment(
        tool="trading_pause_deployment",
        deployment_id=deployment_id,
        target_status="paused",
        ctx=ctx,
    )


async def trading_stop_deployment(
    deployment_id: str,
    ctx: Context | None = None,
) -> str:
    """Stop one deployment."""
    return await _transition_deployment(
        tool="trading_stop_deployment",
        deployment_id=deployment_id,
        target_status="stopped",
        ctx=ctx,
    )


async def trading_get_positions(
    deployment_id: str,
    limit: int = 200,
    ctx: Context | None = None,
) -> str:
    """List positions for one deployment."""
    try:
        deployment_uuid = _parse_uuid(deployment_id, "deployment_id")
    except ValueError as exc:
        return _payload(
            tool="trading_get_positions",
            ok=False,
            error_code="INVALID_UUID",
            error_message=str(exc),
        )

    user_id, error_payload = _require_context_user_id(tool="trading_get_positions", ctx=ctx)
    if error_payload is not None or user_id is None:
        return error_payload or _payload(
            tool="trading_get_positions",
            ok=False,
            error_code="MISSING_CONTEXT",
            error_message="Missing user context.",
        )

    try:
        safe_limit = min(max(int(limit), 1), 500)
    except (TypeError, ValueError):
        safe_limit = 200

    try:
        async with await _new_db_session() as db:
            await deployments_router._load_owned_deployment(
                db,
                deployment_id=deployment_uuid,
                user_id=user_id,
            )
            rows = list(
                (
                    await db.scalars(
                        select(Position)
                        .where(Position.deployment_id == deployment_uuid)
                        .order_by(Position.updated_at.desc())
                        .limit(safe_limit),
                    )
                ).all()
            )
            positions = [
                _serialize_model(deployments_router._serialize_position(position))
                for position in rows
            ]
    except HTTPException as exc:
        code, message = _http_error_code_message(exc)
        return _payload(
            tool="trading_get_positions",
            ok=False,
            error_code=code,
            error_message=message,
        )
    except Exception as exc:  # noqa: BLE001
        return _payload(
            tool="trading_get_positions",
            ok=False,
            error_code="TRADING_POSITIONS_ERROR",
            error_message=f"{type(exc).__name__}: {exc}",
        )

    return _payload(
        tool="trading_get_positions",
        ok=True,
        data={
            "deployment_id": str(deployment_uuid),
            "count": len(positions),
            "positions": positions,
        },
    )


async def trading_get_orders(
    deployment_id: str,
    limit: int = 200,
    ctx: Context | None = None,
) -> str:
    """List orders for one deployment."""
    try:
        deployment_uuid = _parse_uuid(deployment_id, "deployment_id")
    except ValueError as exc:
        return _payload(
            tool="trading_get_orders",
            ok=False,
            error_code="INVALID_UUID",
            error_message=str(exc),
        )

    user_id, error_payload = _require_context_user_id(tool="trading_get_orders", ctx=ctx)
    if error_payload is not None or user_id is None:
        return error_payload or _payload(
            tool="trading_get_orders",
            ok=False,
            error_code="MISSING_CONTEXT",
            error_message="Missing user context.",
        )

    try:
        safe_limit = min(max(int(limit), 1), 500)
    except (TypeError, ValueError):
        safe_limit = 200

    try:
        async with await _new_db_session() as db:
            await deployments_router._load_owned_deployment(
                db,
                deployment_id=deployment_uuid,
                user_id=user_id,
            )
            rows = list(
                (
                    await db.scalars(
                        select(Order)
                        .where(Order.deployment_id == deployment_uuid)
                        .order_by(Order.submitted_at.desc())
                        .limit(safe_limit),
                    )
                ).all()
            )
            orders = [
                _serialize_model(deployments_router._serialize_order(order))
                for order in rows
            ]
    except HTTPException as exc:
        code, message = _http_error_code_message(exc)
        return _payload(
            tool="trading_get_orders",
            ok=False,
            error_code=code,
            error_message=message,
        )
    except Exception as exc:  # noqa: BLE001
        return _payload(
            tool="trading_get_orders",
            ok=False,
            error_code="TRADING_ORDERS_ERROR",
            error_message=f"{type(exc).__name__}: {exc}",
        )

    return _payload(
        tool="trading_get_orders",
        ok=True,
        data={
            "deployment_id": str(deployment_uuid),
            "count": len(orders),
            "orders": orders,
        },
    )


def register_trading_tools(mcp: FastMCP) -> None:
    """Register trading-domain tools."""
    mcp.tool()(trading_ping)
    mcp.tool()(trading_capabilities)
    mcp.tool()(trading_create_paper_deployment)
    mcp.tool()(trading_list_deployments)
    mcp.tool()(trading_start_deployment)
    mcp.tool()(trading_pause_deployment)
    mcp.tool()(trading_stop_deployment)
    mcp.tool()(trading_get_positions)
    mcp.tool()(trading_get_orders)
