"""Stress MCP tools for black swan / Monte Carlo / sensitivity / optimization."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from mcp.server.fastmcp import Context, FastMCP

from src.engine.stress import (
    StressInputError,
    StressJobNotFoundError,
    StressStrategyNotFoundError,
    create_stress_job,
    execute_stress_job,
    get_optimization_pareto_points,
    get_stress_job_view,
    list_black_swan_windows,
    schedule_stress_job,
)
from src.mcp._utils import log_mcp_tool_result, to_json, utc_now_iso
from src.mcp.context_auth import (
    McpContextClaims,
    decode_mcp_context_token,
    extract_mcp_context_token,
)
from src.models import database as db_module

TOOL_NAMES: tuple[str, ...] = (
    "stress_ping",
    "stress_capabilities",
    "stress_black_swan_list_windows",
    "stress_black_swan_create_job",
    "stress_black_swan_get_job",
    "stress_monte_carlo_create_job",
    "stress_monte_carlo_get_job",
    "stress_param_sensitivity_create_job",
    "stress_param_sensitivity_get_job",
    "stress_optimize_create_job",
    "stress_optimize_get_job",
    "stress_optimize_get_pareto",
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
        "category": "stress",
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
        category="stress",
        tool=tool,
        ok=ok,
        error_code=resolved_error_code,
        error_message=resolved_error_message,
    )
    return to_json(body)


def _parse_uuid(value: str, field_name: str) -> UUID:
    try:
        return UUID(str(value).strip())
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Invalid {field_name}: {value}") from exc


async def _new_db_session():
    if db_module.AsyncSessionLocal is None:
        await db_module.init_postgres(ensure_schema=False)
    assert db_module.AsyncSessionLocal is not None
    return db_module.AsyncSessionLocal()


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


def _view_to_payload(view: Any) -> dict[str, Any]:
    return {
        "stress_job_id": str(view.job_id),
        "strategy_id": str(view.strategy_id),
        "base_backtest_job_id": str(view.base_backtest_job_id) if view.base_backtest_job_id else None,
        "job_type": view.job_type,
        "status": view.status,
        "progress": view.progress,
        "current_step": view.current_step,
        "config": view.config,
        "summary": view.summary,
        "items": list(view.items),
        "trials": list(view.trials),
        "error": view.error,
        "submitted_at": view.submitted_at.isoformat(),
        "completed_at": view.completed_at.isoformat() if view.completed_at else None,
    }


async def _create_job(
    *,
    tool: str,
    job_type: str,
    strategy_id: str,
    backtest_job_id: str,
    config: dict[str, Any],
    run_async: bool,
    ctx: Context | None,
) -> str:
    try:
        claims = _resolve_context_claims(ctx)
        strategy_uuid = _parse_uuid(strategy_id, "strategy_id") if strategy_id.strip() else None
        backtest_uuid = _parse_uuid(backtest_job_id, "backtest_job_id") if backtest_job_id.strip() else None
    except ValueError as exc:
        return _payload(
            tool=tool,
            ok=False,
            error_code="INVALID_INPUT",
            error_message=str(exc),
        )

    try:
        async with await _new_db_session() as db:
            receipt = await create_stress_job(
                db,
                job_type=job_type,  # type: ignore[arg-type]
                strategy_id=strategy_uuid,
                backtest_job_id=backtest_uuid,
                config=config,
                user_id=claims.user_id if claims is not None else None,
                auto_commit=True,
            )

            queued_task_id: str | None = None
            if run_async:
                queued_task_id = await schedule_stress_job(receipt.job_id)
                view = await get_stress_job_view(
                    db,
                    job_id=receipt.job_id,
                    user_id=claims.user_id if claims is not None else None,
                )
            else:
                view = await execute_stress_job(
                    db,
                    job_id=receipt.job_id,
                    auto_commit=True,
                )

            data = _view_to_payload(view)
            data["queued_task_id"] = queued_task_id
            data["run_async"] = bool(run_async)
            return _payload(tool=tool, ok=True, data=data)
    except StressStrategyNotFoundError as exc:
        return _payload(
            tool=tool,
            ok=False,
            error_code="STRATEGY_NOT_FOUND",
            error_message=str(exc),
        )
    except StressInputError as exc:
        return _payload(
            tool=tool,
            ok=False,
            error_code="INVALID_INPUT",
            error_message=str(exc),
        )
    except Exception as exc:  # noqa: BLE001
        return _payload(
            tool=tool,
            ok=False,
            error_code="STRESS_CREATE_ERROR",
            error_message=f"{type(exc).__name__}: {exc}",
        )


async def _get_job(
    *,
    tool: str,
    stress_job_id: str,
    expect_job_type: str,
    ctx: Context | None,
) -> str:
    try:
        claims = _resolve_context_claims(ctx)
        job_uuid = _parse_uuid(stress_job_id, "stress_job_id")
    except ValueError as exc:
        return _payload(
            tool=tool,
            ok=False,
            error_code="INVALID_INPUT",
            error_message=str(exc),
        )

    try:
        async with await _new_db_session() as db:
            view = await get_stress_job_view(
                db,
                job_id=job_uuid,
                user_id=claims.user_id if claims is not None else None,
            )
    except StressJobNotFoundError as exc:
        return _payload(
            tool=tool,
            ok=False,
            error_code="NOT_FOUND",
            error_message=str(exc),
        )
    except Exception as exc:  # noqa: BLE001
        return _payload(
            tool=tool,
            ok=False,
            error_code="STRESS_GET_ERROR",
            error_message=f"{type(exc).__name__}: {exc}",
        )

    if view.job_type != expect_job_type:
        return _payload(
            tool=tool,
            ok=False,
            error_code="INVALID_INPUT",
            error_message=(
                f"stress_job_id={stress_job_id} is job_type={view.job_type}, "
                f"expected={expect_job_type}"
            ),
        )
    return _payload(tool=tool, ok=True, data=_view_to_payload(view))


def stress_ping() -> str:
    """Health probe for stress domain."""
    return _payload(tool="stress_ping", ok=True, data={"status": "ready"})


def stress_capabilities() -> str:
    """Expose available stress tools."""
    return _payload(
        tool="stress_capabilities",
        ok=True,
        data={
            "available": True,
            "supported_tools": list(TOOL_NAMES),
            "run_async_default": True,
            "job_types": ["black_swan", "monte_carlo", "param_scan", "optimization"],
        },
    )


def stress_black_swan_list_windows(market: str = "us_stocks") -> str:
    """List predefined black-swan windows for one market."""

    try:
        windows = list_black_swan_windows(market=market)
    except Exception as exc:  # noqa: BLE001
        return _payload(
            tool="stress_black_swan_list_windows",
            ok=False,
            error_code="INVALID_INPUT",
            error_message=f"{type(exc).__name__}: {exc}",
        )

    return _payload(
        tool="stress_black_swan_list_windows",
        ok=True,
        data={
            "market": market,
            "count": len(windows),
            "windows": windows,
        },
    )


async def stress_black_swan_create_job(
    strategy_id: str = "",
    backtest_job_id: str = "",
    window_set: str = "default",
    metrics: list[str] | None = None,
    custom_windows: list[dict[str, Any]] | None = None,
    run_async: bool = True,
    ctx: Context | None = None,
) -> str:
    """Create black-swan stress job."""

    config = {
        "window_set": window_set,
        "metrics": list(metrics or []),
        "custom_windows": list(custom_windows or []),
    }
    return await _create_job(
        tool="stress_black_swan_create_job",
        job_type="black_swan",
        strategy_id=strategy_id,
        backtest_job_id=backtest_job_id,
        config=config,
        run_async=run_async,
        ctx=ctx,
    )


async def stress_black_swan_get_job(
    stress_job_id: str,
    ctx: Context | None = None,
) -> str:
    """Get black-swan stress job."""

    return await _get_job(
        tool="stress_black_swan_get_job",
        stress_job_id=stress_job_id,
        expect_job_type="black_swan",
        ctx=ctx,
    )


async def stress_monte_carlo_create_job(
    strategy_id: str = "",
    backtest_job_id: str = "",
    num_trials: int = 2000,
    horizon_bars: int = 252,
    method: str = "block_bootstrap",
    ruin_threshold_pct: float = -30.0,
    run_async: bool = True,
    ctx: Context | None = None,
) -> str:
    """Create Monte Carlo stress job."""

    config = {
        "num_trials": int(num_trials),
        "horizon_bars": int(horizon_bars),
        "method": method,
        "ruin_threshold_pct": float(ruin_threshold_pct),
    }
    return await _create_job(
        tool="stress_monte_carlo_create_job",
        job_type="monte_carlo",
        strategy_id=strategy_id,
        backtest_job_id=backtest_job_id,
        config=config,
        run_async=run_async,
        ctx=ctx,
    )


async def stress_monte_carlo_get_job(
    stress_job_id: str,
    ctx: Context | None = None,
) -> str:
    """Get Monte Carlo stress job."""

    return await _get_job(
        tool="stress_monte_carlo_get_job",
        stress_job_id=stress_job_id,
        expect_job_type="monte_carlo",
        ctx=ctx,
    )


async def stress_param_sensitivity_create_job(
    strategy_id: str,
    scan_pct: float = 10.0,
    steps_per_side: int = 3,
    target_params: list[str] | None = None,
    metric_set: str = "return,sharpe,max_dd,stability",
    run_async: bool = True,
    ctx: Context | None = None,
) -> str:
    """Create parameter sensitivity scan job."""

    config = {
        "scan_pct": float(scan_pct),
        "steps_per_side": int(steps_per_side),
        "target_params": list(target_params or []),
        "metric_set": metric_set,
    }
    return await _create_job(
        tool="stress_param_sensitivity_create_job",
        job_type="param_scan",
        strategy_id=strategy_id,
        backtest_job_id="",
        config=config,
        run_async=run_async,
        ctx=ctx,
    )


async def stress_param_sensitivity_get_job(
    stress_job_id: str,
    ctx: Context | None = None,
) -> str:
    """Get parameter sensitivity scan job."""

    return await _get_job(
        tool="stress_param_sensitivity_get_job",
        stress_job_id=stress_job_id,
        expect_job_type="param_scan",
        ctx=ctx,
    )


async def stress_optimize_create_job(
    strategy_id: str,
    method: str = "random",
    search_space: dict[str, Any] | None = None,
    budget: int = 40,
    objectives: list[str] | None = None,
    constraints: dict[str, Any] | None = None,
    run_async: bool = True,
    ctx: Context | None = None,
) -> str:
    """Create optimization stress job."""

    config = {
        "method": method,
        "search_space": dict(search_space or {}),
        "budget": int(budget),
        "objectives": list(objectives or []),
        "constraints": dict(constraints or {}),
    }
    return await _create_job(
        tool="stress_optimize_create_job",
        job_type="optimization",
        strategy_id=strategy_id,
        backtest_job_id="",
        config=config,
        run_async=run_async,
        ctx=ctx,
    )


async def stress_optimize_get_job(
    stress_job_id: str,
    ctx: Context | None = None,
) -> str:
    """Get optimization stress job."""

    return await _get_job(
        tool="stress_optimize_get_job",
        stress_job_id=stress_job_id,
        expect_job_type="optimization",
        ctx=ctx,
    )


async def stress_optimize_get_pareto(
    stress_job_id: str,
    x_metric: str,
    y_metric: str,
    ctx: Context | None = None,
) -> str:
    """Project optimization trials into x/y metric points."""

    try:
        claims = _resolve_context_claims(ctx)
        job_uuid = _parse_uuid(stress_job_id, "stress_job_id")
    except ValueError as exc:
        return _payload(
            tool="stress_optimize_get_pareto",
            ok=False,
            error_code="INVALID_INPUT",
            error_message=str(exc),
        )

    try:
        async with await _new_db_session() as db:
            points = await get_optimization_pareto_points(
                db,
                job_id=job_uuid,
                x_metric=x_metric,
                y_metric=y_metric,
                user_id=claims.user_id if claims is not None else None,
            )
    except StressJobNotFoundError as exc:
        return _payload(
            tool="stress_optimize_get_pareto",
            ok=False,
            error_code="NOT_FOUND",
            error_message=str(exc),
        )
    except StressInputError as exc:
        return _payload(
            tool="stress_optimize_get_pareto",
            ok=False,
            error_code="INVALID_INPUT",
            error_message=str(exc),
        )
    except Exception as exc:  # noqa: BLE001
        return _payload(
            tool="stress_optimize_get_pareto",
            ok=False,
            error_code="STRESS_OPTIMIZE_PARETO_ERROR",
            error_message=f"{type(exc).__name__}: {exc}",
        )

    return _payload(
        tool="stress_optimize_get_pareto",
        ok=True,
        data={
            "stress_job_id": stress_job_id,
            "x_metric": x_metric,
            "y_metric": y_metric,
            "points": points,
        },
    )


def register_stress_tools(mcp: FastMCP) -> None:
    """Register stress tools."""

    mcp.tool()(stress_ping)
    mcp.tool()(stress_capabilities)
    mcp.tool()(stress_black_swan_list_windows)
    mcp.tool()(stress_black_swan_create_job)
    mcp.tool()(stress_black_swan_get_job)
    mcp.tool()(stress_monte_carlo_create_job)
    mcp.tool()(stress_monte_carlo_get_job)
    mcp.tool()(stress_param_sensitivity_create_job)
    mcp.tool()(stress_param_sensitivity_get_job)
    mcp.tool()(stress_optimize_create_job)
    mcp.tool()(stress_optimize_get_job)
    mcp.tool()(stress_optimize_get_pareto)
