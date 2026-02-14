"""Backtest MCP tools: create job and query unified job view."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from mcp.server.fastmcp import FastMCP

from src.engine.backtest import (
    BacktestJobNotFoundError,
    BacktestStrategyNotFoundError,
    create_backtest_job,
    get_backtest_job_view,
    schedule_backtest_job,
)
from src.mcp._utils import to_json, utc_now_iso
from src.models import database as db_module

TOOL_NAMES: tuple[str, ...] = (
    "backtest_create_job",
    "backtest_get_job",
)

_MAX_SAMPLE_TRADES = 5
_MAX_SAMPLE_EVENTS = 5


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
    if data:
        body.update(data)
    if not ok:
        body["error"] = {
            "code": error_code or "UNKNOWN_ERROR",
            "message": error_message or "Unknown error",
        }
    return to_json(body)


def _parse_uuid(value: str, field_name: str) -> UUID:
    try:
        return UUID(value)
    except ValueError as exc:
        raise ValueError(f"Invalid {field_name}: {value}") from exc


async def _new_db_session():
    if db_module.AsyncSessionLocal is None:
        await db_module.init_postgres()
    assert db_module.AsyncSessionLocal is not None
    return db_module.AsyncSessionLocal()


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
        data["result"] = _summarize_result_payload(view.result)
    elif view.status == "failed":
        data["result"] = {
            "error": view.error or {"code": "BACKTEST_FAILED", "message": "Backtest failed"}
        }
    else:
        data["result"] = None
    return data


def _summarize_result_payload(result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {
            "summary": {},
            "performance": {},
            "counts": {"trades": 0, "equity_points": 0, "events": 0},
            "sample_trades": [],
            "sample_events": [],
            "result_truncated": True,
        }

    summary = result.get("summary")
    if not isinstance(summary, dict):
        summary = {}

    performance = result.get("performance")
    if not isinstance(performance, dict):
        performance = {}

    trades = result.get("trades")
    if not isinstance(trades, list):
        trades = []

    events = result.get("events")
    if not isinstance(events, list):
        events = []

    equity_curve = result.get("equity_curve")
    if not isinstance(equity_curve, list):
        equity_curve = []

    output: dict[str, Any] = {
        "market": result.get("market"),
        "symbol": result.get("symbol"),
        "timeframe": result.get("timeframe"),
        "summary": summary,
        "performance": performance,
        "counts": {
            "trades": len(trades),
            "equity_points": len(equity_curve),
            "events": len(events),
        },
        "sample_trades": trades[:_MAX_SAMPLE_TRADES],
        "sample_events": events[:_MAX_SAMPLE_EVENTS],
        "started_at": result.get("started_at"),
        "finished_at": result.get("finished_at"),
        "result_truncated": True,
    }
    return output


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
            async with await _new_db_session() as db:
                receipt = await create_backtest_job(
                    db,
                    strategy_id=strategy_uuid,
                    start_date=start_date,
                    end_date=end_date,
                    initial_capital=initial_capital,
                    commission_rate=commission_rate,
                    slippage_bps=slippage_bps,
                    auto_commit=True,
                )
                # Keep accepting run_now for backward compatibility, but execution is always queued.
                await schedule_backtest_job(receipt.job_id)
                view = await get_backtest_job_view(db, job_id=receipt.job_id)
        except BacktestStrategyNotFoundError as exc:
            return _payload(
                tool="backtest_create_job",
                ok=False,
                error_code="STRATEGY_NOT_FOUND",
                error_message=str(exc),
            )
        except Exception as exc:  # noqa: BLE001
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
    async def backtest_get_job(job_id: str) -> str:
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
            async with await _new_db_session() as db:
                view = await get_backtest_job_view(db, job_id=job_uuid)
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
