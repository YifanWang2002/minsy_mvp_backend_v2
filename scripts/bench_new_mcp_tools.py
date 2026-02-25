from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from uuid import uuid4

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mcp.server.fastmcp import FastMCP

from src.config import settings
from src.engine.data.data_loader import DataLoader
from src.engine.execution.adapters.base import OhlcvBar
from src.engine.market_data import sync_service
from src.engine.strategy import EXAMPLE_PATH, load_strategy_payload, upsert_strategy_dsl
from src.mcp.market_data import tools as market_tools
from src.mcp.stress import tools as stress_tools
from src.models import database as db_module
from src.models.session import Session
from src.models.user import User

MARKET_TOOLS = {
    "market_data_detect_missing_ranges",
    "market_data_fetch_missing_ranges",
    "market_data_get_sync_job",
}

STRESS_TOOLS = {
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
}

ALL_TOOLS = sorted(MARKET_TOOLS | STRESS_TOOLS)


def _extract_payload(call_result: object) -> dict[str, Any]:
    if isinstance(call_result, tuple) and len(call_result) == 2:
        maybe_result = call_result[1]
        if isinstance(maybe_result, dict):
            raw = maybe_result.get("result")
            if isinstance(raw, str):
                return json.loads(raw)
    raise RuntimeError(f"unexpected mcp result: {call_result!r}")


async def _prepare_strategy() -> tuple[str, str]:
    if not settings.postgres_db.endswith("_bench"):
        settings.postgres_db = f"{settings.postgres_db}_bench"

    await db_module.init_postgres(ensure_schema=True)
    assert db_module.AsyncSessionLocal is not None

    async with db_module.AsyncSessionLocal() as db:
        user = User(
            email=f"bench_{uuid4().hex}@example.com",
            password_hash="hash",
            name="bench-user",
        )
        db.add(user)
        await db.flush()

        session = Session(
            user_id=user.id,
            current_phase="stress_test",
            status="active",
            artifacts={},
            metadata_={},
        )
        db.add(session)
        await db.flush()

        payload = load_strategy_payload(EXAMPLE_PATH)
        payload["universe"] = {"market": "crypto", "tickers": ["BTCUSD"]}
        payload["timeframe"] = "1d"

        persistence = await upsert_strategy_dsl(
            db,
            session_id=session.id,
            dsl_payload=payload,
            auto_commit=False,
        )
        await db.commit()
        return str(persistence.strategy.id), str(session.id)


def _fake_provider_builder_factory() -> Any:
    start = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    end = datetime(2024, 1, 3, 0, 0, tzinfo=UTC)

    class _FakeProvider:
        async def fetch_ohlcv(self, **kwargs: Any) -> list[OhlcvBar]:
            timeframe = str(kwargs.get("timeframe", "1Min"))
            since = kwargs.get("since")
            limit = max(1, int(kwargs.get("limit", 500)))

            if isinstance(since, datetime):
                cursor = since.astimezone(UTC)
            else:
                cursor = start

            if "5" in timeframe and "Min" in timeframe:
                step = timedelta(minutes=5)
            else:
                step = timedelta(minutes=1)

            if cursor > end:
                return []

            rows: list[OhlcvBar] = []
            current = cursor
            for _ in range(limit):
                if current > end:
                    break
                base = Decimal("100") + Decimal(str((current - start).total_seconds() / 60.0 * 0.01))
                rows.append(
                    OhlcvBar(
                        timestamp=current,
                        open=base,
                        high=base + Decimal("0.5"),
                        low=base - Decimal("0.5"),
                        close=base + Decimal("0.1"),
                        volume=Decimal("10"),
                    )
                )
                current += step
            return rows

    async def _builder(_provider_name: str):
        provider = _FakeProvider()

        async def _close() -> None:
            return None

        return provider, _close

    return _builder


def _cleanup_benchmark_parquet_files(*, market: str, symbol: str) -> None:
    """Delete synthetic parquet shards produced by benchmark runs."""
    try:
        loader = DataLoader()
        market_key = loader.normalize_market(market)
    except Exception:  # noqa: BLE001
        return

    market_dir = loader.data_dir / market_key
    if not market_dir.exists():
        return

    for parquet_file in market_dir.glob(f"{symbol}_*.parquet"):
        try:
            parquet_file.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            continue


async def _call_tool(mcp: FastMCP, tool: str, arguments: dict[str, Any]) -> tuple[dict[str, Any], float]:
    t0 = time.perf_counter()
    payload = _extract_payload(await mcp.call_tool(tool, arguments))
    elapsed = time.perf_counter() - t0
    return payload, elapsed


async def run_one(tool: str) -> dict[str, Any]:
    if tool not in ALL_TOOLS:
        raise ValueError(f"Unsupported tool: {tool}")

    strategy_id, _session_id = await _prepare_strategy()
    # Use an isolated symbol per run so fetch/get benchmarks are deterministic
    # even if local parquet data from previous runs already exists.
    market_symbol = f"BTCBENCH{uuid4().hex[:6].upper()}"

    market_mcp = FastMCP("bench-market")
    market_tools.register_market_data_tools(market_mcp)

    stress_mcp = FastMCP("bench-stress")
    stress_tools.register_stress_tools(stress_mcp)

    result: dict[str, Any]
    elapsed: float

    old_builder = sync_service._build_provider_client
    sync_service._build_provider_client = _fake_provider_builder_factory()
    try:
        if tool == "market_data_detect_missing_ranges":
            result, elapsed = await _call_tool(
                market_mcp,
                tool,
                {
                    "market": "crypto",
                    "symbol": market_symbol,
                    "timeframe": "1m",
                    "start_date": "2024-01-01T00:00:00Z",
                    "end_date": "2024-01-03T00:00:00Z",
                },
            )
        elif tool == "market_data_fetch_missing_ranges":
            result, elapsed = await _call_tool(
                market_mcp,
                tool,
                {
                    "provider": "alpaca",
                    "market": "crypto",
                    "symbol": market_symbol,
                    "timeframe": "1m",
                    "start_date": "2024-01-01T00:00:00Z",
                    "end_date": "2024-01-03T00:00:00Z",
                    "run_async": False,
                },
            )
        elif tool == "market_data_get_sync_job":
            created, _ = await _call_tool(
                market_mcp,
                "market_data_fetch_missing_ranges",
                {
                    "provider": "alpaca",
                    "market": "crypto",
                    "symbol": market_symbol,
                    "timeframe": "1m",
                    "start_date": "2024-01-01T00:00:00Z",
                    "end_date": "2024-01-03T00:00:00Z",
                    "run_async": False,
                },
            )
            result, elapsed = await _call_tool(
                market_mcp,
                tool,
                {"sync_job_id": created["sync_job_id"]},
            )
        elif tool == "stress_black_swan_list_windows":
            result, elapsed = await _call_tool(stress_mcp, tool, {"market": "crypto"})
        elif tool == "stress_black_swan_create_job":
            result, elapsed = await _call_tool(
                stress_mcp,
                tool,
                {
                    "strategy_id": strategy_id,
                    "window_set": "custom",
                    "custom_windows": [
                        {
                            "window_id": "w1",
                            "label": "Week 1",
                            "start": "2024-01-01T00:00:00Z",
                            "end": "2024-01-07T00:00:00Z",
                        },
                        {
                            "window_id": "w2",
                            "label": "Week 2",
                            "start": "2024-01-08T00:00:00Z",
                            "end": "2024-01-14T00:00:00Z",
                        },
                    ],
                    "run_async": False,
                },
            )
        elif tool == "stress_black_swan_get_job":
            created, _ = await _call_tool(
                stress_mcp,
                "stress_black_swan_create_job",
                {
                    "strategy_id": strategy_id,
                    "window_set": "custom",
                    "custom_windows": [
                        {
                            "window_id": "w1",
                            "label": "Week 1",
                            "start": "2024-01-01T00:00:00Z",
                            "end": "2024-01-07T00:00:00Z",
                        }
                    ],
                    "run_async": False,
                },
            )
            result, elapsed = await _call_tool(
                stress_mcp,
                tool,
                {"stress_job_id": created["stress_job_id"]},
            )
        elif tool == "stress_monte_carlo_create_job":
            result, elapsed = await _call_tool(
                stress_mcp,
                tool,
                {
                    "strategy_id": strategy_id,
                    "num_trials": 2000,
                    "horizon_bars": 252,
                    "method": "block_bootstrap",
                    "ruin_threshold_pct": -30.0,
                    "run_async": False,
                },
            )
        elif tool == "stress_monte_carlo_get_job":
            created, _ = await _call_tool(
                stress_mcp,
                "stress_monte_carlo_create_job",
                {
                    "strategy_id": strategy_id,
                    "num_trials": 1000,
                    "horizon_bars": 126,
                    "run_async": False,
                },
            )
            result, elapsed = await _call_tool(
                stress_mcp,
                tool,
                {"stress_job_id": created["stress_job_id"]},
            )
        elif tool == "stress_param_sensitivity_create_job":
            result, elapsed = await _call_tool(
                stress_mcp,
                tool,
                {
                    "strategy_id": strategy_id,
                    "scan_pct": 10.0,
                    "steps_per_side": 2,
                    "target_params": ["ema_9.period", "ema_21.period"],
                    "run_async": False,
                },
            )
        elif tool == "stress_param_sensitivity_get_job":
            created, _ = await _call_tool(
                stress_mcp,
                "stress_param_sensitivity_create_job",
                {
                    "strategy_id": strategy_id,
                    "scan_pct": 10.0,
                    "steps_per_side": 1,
                    "target_params": ["ema_9.period"],
                    "run_async": False,
                },
            )
            result, elapsed = await _call_tool(
                stress_mcp,
                tool,
                {"stress_job_id": created["stress_job_id"]},
            )
        elif tool == "stress_optimize_create_job":
            result, elapsed = await _call_tool(
                stress_mcp,
                tool,
                {
                    "strategy_id": strategy_id,
                    "method": "random",
                    "budget": 20,
                    "search_space": {
                        "ema_9.period": {"min": 6, "max": 20, "step": 1},
                        "ema_21.period": {"min": 15, "max": 40, "step": 1},
                    },
                    "objectives": ["max_return", "min_drawdown", "max_stability"],
                    "constraints": {"max_drawdown_pct": 40},
                    "run_async": False,
                },
            )
        elif tool == "stress_optimize_get_job":
            created, _ = await _call_tool(
                stress_mcp,
                "stress_optimize_create_job",
                {
                    "strategy_id": strategy_id,
                    "method": "random",
                    "budget": 10,
                    "search_space": {
                        "ema_9.period": {"min": 6, "max": 15, "step": 1},
                        "ema_21.period": {"min": 16, "max": 35, "step": 1},
                    },
                    "run_async": False,
                },
            )
            result, elapsed = await _call_tool(
                stress_mcp,
                tool,
                {"stress_job_id": created["stress_job_id"]},
            )
        elif tool == "stress_optimize_get_pareto":
            created, _ = await _call_tool(
                stress_mcp,
                "stress_optimize_create_job",
                {
                    "strategy_id": strategy_id,
                    "method": "random",
                    "budget": 10,
                    "search_space": {
                        "ema_9.period": {"min": 6, "max": 15, "step": 1},
                        "ema_21.period": {"min": 16, "max": 35, "step": 1},
                    },
                    "run_async": False,
                },
            )
            result, elapsed = await _call_tool(
                stress_mcp,
                tool,
                {
                    "stress_job_id": created["stress_job_id"],
                    "x_metric": "total_return_pct",
                    "y_metric": "max_drawdown_pct",
                },
            )
        else:
            raise ValueError(tool)
    finally:
        sync_service._build_provider_client = old_builder
        _cleanup_benchmark_parquet_files(market="crypto", symbol=market_symbol)

    preview_keys = [
        "ok",
        "status",
        "sync_job_id",
        "stress_job_id",
        "job_type",
        "progress",
        "rows_written",
        "local_coverage_pct",
        "count",
    ]
    preview = {key: result.get(key) for key in preview_keys if key in result}

    return {
        "tool": tool,
        "duration_sec": round(elapsed, 6),
        "preview": preview,
        "payload": result,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark one new MCP tool call")
    parser.add_argument("--tool", required=True, choices=ALL_TOOLS)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.dirname(__file__)))
    output = asyncio.run(run_one(args.tool))
    if args.pretty:
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(output, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
