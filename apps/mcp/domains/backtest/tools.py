"""Backtest MCP tools: create job and query unified job view."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from mcp.server.fastmcp import Context, FastMCP

from packages.domain.backtest import (
    BacktestBarLimitExceededError,
    BacktestJobNotFoundError,
    BacktestStrategyNotFoundError,
    create_backtest_job,
    get_backtest_job_view,
    schedule_backtest_job,
)
from packages.domain.backtest.analytics import (
    build_backtest_overview,
    compute_entry_hour_pnl_heatmap,
    compute_entry_weekday_pnl,
    compute_exit_reason_breakdown,
    compute_holding_period_pnl_bins,
    compute_long_short_breakdown,
    compute_monthly_return_table,
    compute_rolling_metrics,
    compute_underwater_curve,
)
from apps.mcp.auth.context_auth import (
    McpContextClaims,
    decode_mcp_context_token,
    extract_mcp_context_token,
)
from apps.mcp.common.utils import log_mcp_tool_result, to_json, utc_now_iso
from packages.infra.db import session as db_module

TOOL_NAMES: tuple[str, ...] = (
    "backtest_create_job",
    "backtest_get_job",
    "backtest_entry_hour_pnl_heatmap",
    "backtest_entry_weekday_pnl",
    "backtest_monthly_return_table",
    "backtest_holding_period_pnl_bins",
    "backtest_long_short_breakdown",
    "backtest_exit_reason_breakdown",
    "backtest_underwater_curve",
    "backtest_rolling_metrics",
)

_MAX_SAMPLE_TRADES = 5
_MAX_SAMPLE_EVENTS = 5
_MAX_CURVE_POINTS = 240


def _payload(
    *,
    tool: str,
    ok: bool,
    data: dict[str, Any] | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
) -> str:
    body: dict[str, Any] = {
        "category": "backtest",
        "tool": tool,
        "ok": ok,
        "timestamp_utc": utc_now_iso(),
    }
    resolved_error_code: str | None = None
    resolved_error_message: str | None = None
    if data:
        body.update(data)
    if not ok:
        resolved_error_code = error_code or "UNKNOWN_ERROR"
        resolved_error_message = error_message or "Unknown error"
        body["error"] = {
            "code": resolved_error_code,
            "message": resolved_error_message,
        }
    log_mcp_tool_result(
        category="backtest",
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


async def _new_db_session():
    if db_module.AsyncSessionLocal is None:
        # MCP worker process only needs a ready pool; schema is managed by API startup/migrations.
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


def _view_payload(view: Any) -> dict[str, Any]:
    result_ready = view.result is not None
    data: dict[str, Any] = {
        "job_id": str(view.job_id),
        "strategy_id": str(view.strategy_id),
        "status": view.status,
        "progress": view.progress,
        "current_step": view.current_step,
        "submitted_at": view.submitted_at.isoformat(),
        "completed_at": view.completed_at.isoformat() if view.completed_at else None,
        "result_ready": result_ready,
        "error": view.error,
    }

    if view.status == "done":
        data["result"] = build_backtest_overview(
            view.result,
            sample_trades=_MAX_SAMPLE_TRADES,
            sample_events=_MAX_SAMPLE_EVENTS,
        )
    elif view.status == "failed":
        data["result"] = {
            "error": view.error or {"code": "BACKTEST_FAILED", "message": "Backtest failed"}
        }
    else:
        data["result"] = None
    return data


def _build_job_stub(view: Any) -> dict[str, Any]:
    return {
        "job_id": str(view.job_id),
        "strategy_id": str(view.strategy_id),
        "status": view.status,
        "progress": view.progress,
        "current_step": view.current_step,
        "result_ready": view.status == "done" and isinstance(view.result, dict),
        "error": view.error,
        "submitted_at": view.submitted_at.isoformat(),
        "completed_at": view.completed_at.isoformat() if view.completed_at else None,
    }


def _require_completed_result(
    *,
    tool: str,
    view: Any,
) -> tuple[dict[str, Any] | None, str | None]:
    if view.status == "failed":
        return None, _payload(
            tool=tool,
            ok=False,
            data=_build_job_stub(view),
            error_code="BACKTEST_FAILED",
            error_message=(view.error or {}).get("message", "Backtest failed."),
        )
    if view.status != "done" or not isinstance(view.result, dict):
        return None, _payload(
            tool=tool,
            ok=False,
            data=_build_job_stub(view),
            error_code="JOB_NOT_READY",
            error_message="Backtest job is not completed yet.",
        )
    return view.result, None


def register_backtest_tools(mcp: FastMCP) -> None:
    """Register backtest tools."""

    @mcp.tool()
    async def backtest_create_job(
        strategy_id: str,
        start_date: str = "",
        end_date: str = "",
        initial_capital: float = 100000.0,
        commission_rate: float = 0.0,
        slippage_bps: float = 0.0,
        run_now: bool = False,
        ctx: Context | None = None,
    ) -> str:
        try:
            strategy_uuid = _parse_uuid(strategy_id, "strategy_id")
        except ValueError as exc:
            return _payload(
                tool="backtest_create_job",
                ok=False,
                error_code="INVALID_UUID",
                error_message=str(exc),
            )
        try:
            claims = _resolve_context_claims(ctx)
        except ValueError as exc:
            return _payload(
                tool="backtest_create_job",
                ok=False,
                error_code="INVALID_INPUT",
                error_message=str(exc),
            )

        try:
            async with await _new_db_session() as db:
                receipt = await create_backtest_job(
                    db,
                    strategy_id=strategy_uuid,
                    start_date=start_date,
                    end_date=end_date,
                    initial_capital=initial_capital,
                    commission_rate=commission_rate,
                    slippage_bps=slippage_bps,
                    user_id=claims.user_id if claims is not None else None,
                    auto_commit=True,
                )
                # Keep accepting run_now for backward compatibility, but execution is always queued.
                await schedule_backtest_job(receipt.job_id)
                view = await get_backtest_job_view(
                    db,
                    job_id=receipt.job_id,
                    user_id=claims.user_id if claims is not None else None,
                )
        except BacktestStrategyNotFoundError as exc:
            return _payload(
                tool="backtest_create_job",
                ok=False,
                error_code="STRATEGY_NOT_FOUND",
                error_message=str(exc),
            )
        except BacktestBarLimitExceededError as exc:
            return _payload(
                tool="backtest_create_job",
                ok=False,
                error_code="BACKTEST_BAR_LIMIT_EXCEEDED",
                error_message=str(exc),
            )
        except Exception as exc:  # noqa: BLE001
            # During incremental refactors, compatibility wrappers can cause
            # class-identity drift across import paths. Fall back to class-name
            # matching so API error contracts remain stable.
            if type(exc).__name__ == "BacktestBarLimitExceededError":
                return _payload(
                    tool="backtest_create_job",
                    ok=False,
                    error_code="BACKTEST_BAR_LIMIT_EXCEEDED",
                    error_message=str(exc),
                )
            return _payload(
                tool="backtest_create_job",
                ok=False,
                error_code="BACKTEST_CREATE_ERROR",
                error_message=f"{type(exc).__name__}: {exc}",
            )

        return _payload(
            tool="backtest_create_job",
            ok=True,
            data=_view_payload(view),
        )

    @mcp.tool()
    async def backtest_get_job(job_id: str, ctx: Context | None = None) -> str:
        try:
            job_uuid = _parse_uuid(job_id, "job_id")
        except ValueError as exc:
            return _payload(
                tool="backtest_get_job",
                ok=False,
                error_code="INVALID_UUID",
                error_message=str(exc),
            )
        try:
            claims = _resolve_context_claims(ctx)
        except ValueError as exc:
            return _payload(
                tool="backtest_get_job",
                ok=False,
                error_code="INVALID_INPUT",
                error_message=str(exc),
            )

        try:
            async with await _new_db_session() as db:
                view = await get_backtest_job_view(
                    db,
                    job_id=job_uuid,
                    user_id=claims.user_id if claims is not None else None,
                )
        except BacktestJobNotFoundError as exc:
            return _payload(
                tool="backtest_get_job",
                ok=False,
                error_code="JOB_NOT_FOUND",
                error_message=str(exc),
            )
        except Exception as exc:  # noqa: BLE001
            return _payload(
                tool="backtest_get_job",
                ok=False,
                error_code="BACKTEST_GET_JOB_ERROR",
                error_message=f"{type(exc).__name__}: {exc}",
            )

        return _payload(
            tool="backtest_get_job",
            ok=True,
            data=_view_payload(view),
        )

    async def _load_completed_result(
        tool: str,
        job_id: str,
        *,
        ctx: Context | None = None,
    ) -> tuple[dict[str, Any] | None, str | None]:
        try:
            job_uuid = _parse_uuid(job_id, "job_id")
        except ValueError as exc:
            return None, _payload(
                tool=tool,
                ok=False,
                error_code="INVALID_UUID",
                error_message=str(exc),
            )
        try:
            claims = _resolve_context_claims(ctx)
        except ValueError as exc:
            return None, _payload(
                tool=tool,
                ok=False,
                error_code="INVALID_INPUT",
                error_message=str(exc),
            )

        try:
            async with await _new_db_session() as db:
                view = await get_backtest_job_view(
                    db,
                    job_id=job_uuid,
                    user_id=claims.user_id if claims is not None else None,
                )
        except BacktestJobNotFoundError as exc:
            return None, _payload(
                tool=tool,
                ok=False,
                error_code="JOB_NOT_FOUND",
                error_message=str(exc),
            )
        except Exception as exc:  # noqa: BLE001
            return None, _payload(
                tool=tool,
                ok=False,
                error_code="BACKTEST_GET_JOB_ERROR",
                error_message=f"{type(exc).__name__}: {exc}",
            )

        result, error_payload = _require_completed_result(tool=tool, view=view)
        if error_payload is not None:
            return None, error_payload
        assert result is not None
        return result, None

    @mcp.tool()
    async def backtest_entry_hour_pnl_heatmap(
        job_id: str,
        ctx: Context | None = None,
    ) -> str:
        result, error_payload = await _load_completed_result(
            "backtest_entry_hour_pnl_heatmap",
            job_id,
            ctx=ctx,
        )
        if error_payload is not None:
            return error_payload
        return _payload(
            tool="backtest_entry_hour_pnl_heatmap",
            ok=True,
            data=compute_entry_hour_pnl_heatmap(result),
        )

    @mcp.tool()
    async def backtest_entry_weekday_pnl(
        job_id: str,
        ctx: Context | None = None,
    ) -> str:
        result, error_payload = await _load_completed_result(
            "backtest_entry_weekday_pnl",
            job_id,
            ctx=ctx,
        )
        if error_payload is not None:
            return error_payload
        return _payload(
            tool="backtest_entry_weekday_pnl",
            ok=True,
            data=compute_entry_weekday_pnl(result),
        )

    @mcp.tool()
    async def backtest_monthly_return_table(
        job_id: str,
        ctx: Context | None = None,
    ) -> str:
        result, error_payload = await _load_completed_result(
            "backtest_monthly_return_table",
            job_id,
            ctx=ctx,
        )
        if error_payload is not None:
            return error_payload
        return _payload(
            tool="backtest_monthly_return_table",
            ok=True,
            data=compute_monthly_return_table(result),
        )

    @mcp.tool()
    async def backtest_holding_period_pnl_bins(
        job_id: str,
        ctx: Context | None = None,
    ) -> str:
        result, error_payload = await _load_completed_result(
            "backtest_holding_period_pnl_bins",
            job_id,
            ctx=ctx,
        )
        if error_payload is not None:
            return error_payload
        return _payload(
            tool="backtest_holding_period_pnl_bins",
            ok=True,
            data=compute_holding_period_pnl_bins(result),
        )

    @mcp.tool()
    async def backtest_long_short_breakdown(
        job_id: str,
        ctx: Context | None = None,
    ) -> str:
        result, error_payload = await _load_completed_result(
            "backtest_long_short_breakdown",
            job_id,
            ctx=ctx,
        )
        if error_payload is not None:
            return error_payload
        return _payload(
            tool="backtest_long_short_breakdown",
            ok=True,
            data=compute_long_short_breakdown(result),
        )

    @mcp.tool()
    async def backtest_exit_reason_breakdown(
        job_id: str,
        ctx: Context | None = None,
    ) -> str:
        result, error_payload = await _load_completed_result(
            "backtest_exit_reason_breakdown",
            job_id,
            ctx=ctx,
        )
        if error_payload is not None:
            return error_payload
        return _payload(
            tool="backtest_exit_reason_breakdown",
            ok=True,
            data=compute_exit_reason_breakdown(result),
        )

    @mcp.tool()
    async def backtest_underwater_curve(
        job_id: str,
        max_points: int = _MAX_CURVE_POINTS,
        ctx: Context | None = None,
    ) -> str:
        result, error_payload = await _load_completed_result(
            "backtest_underwater_curve",
            job_id,
            ctx=ctx,
        )
        if error_payload is not None:
            return error_payload
        cap = max(10, min(1000, int(max_points)))
        return _payload(
            tool="backtest_underwater_curve",
            ok=True,
            data=compute_underwater_curve(result, max_points=cap),
        )

    @mcp.tool()
    async def backtest_rolling_metrics(
        job_id: str,
        window_bars: int = 0,
        max_points: int = _MAX_CURVE_POINTS,
        ctx: Context | None = None,
    ) -> str:
        result, error_payload = await _load_completed_result(
            "backtest_rolling_metrics",
            job_id,
            ctx=ctx,
        )
        if error_payload is not None:
            return error_payload
        requested_window = max(0, int(window_bars))
        cap = max(10, min(1000, int(max_points)))
        return _payload(
            tool="backtest_rolling_metrics",
            ok=True,
            data=compute_rolling_metrics(
                result,
                window_bars=requested_window,
                max_points=cap,
            ),
        )
