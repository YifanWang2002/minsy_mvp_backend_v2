"""Stress job orchestration service (black swan / MC / sensitivity / optimization)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import UUID

import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.config import settings
from src.engine.backtest.engine import EventDrivenBacktestEngine
from src.engine.backtest.types import BacktestConfig, BacktestResult
from src.engine.data import DataLoader
from src.engine.optimization import (
    build_metric_points,
    build_search_space,
    compute_objective_score,
    constraints_satisfied,
    generate_bayes_like_candidates,
    generate_grid_candidates,
    generate_random_candidates,
    pareto_front_indices,
)
from src.engine.strategy import parse_strategy_payload
from src.engine.strategy.parser import build_parsed_strategy
from src.engine.strategy.param_mutation import apply_param_values, list_tunable_params
from src.engine.stress.black_swan import run_black_swan_analysis, serialize_window_result
from src.engine.stress.monte_carlo import run_monte_carlo
from src.engine.stress.param_sensitivity import scan_param_sensitivity
from src.engine.stress.scenario_windows import list_windows, resolve_window_set, serialize_windows
from src.engine.stress.statistics import (
    annualized_stability_score,
    conditional_value_at_risk,
    confidence_intervals,
    histogram,
    percentile_summary,
    risk_of_ruin,
    value_at_risk,
)
from src.models import database as db_module
from src.models.backtest import BacktestJob
from src.models.optimization_trial import OptimizationTrial
from src.models.strategy import Strategy
from src.models.stress_job import StressJob
from src.models.stress_job_item import StressJobItem
from src.util.logger import logger

StressJobType = Literal["black_swan", "monte_carlo", "param_scan", "optimization"]

_INTERNAL_TO_EXTERNAL_STATUS: dict[str, str] = {
    "queued": "pending",
    "running": "running",
    "completed": "done",
    "failed": "failed",
    "cancelled": "failed",
}


@dataclass(frozen=True, slots=True)
class StressJobReceipt:
    """Create-job response."""

    job_id: UUID
    strategy_id: UUID
    job_type: StressJobType
    status: str
    progress: int


@dataclass(frozen=True, slots=True)
class StressJobView:
    """Job read-model for MCP responses."""

    job_id: UUID
    strategy_id: UUID
    base_backtest_job_id: UUID | None
    job_type: StressJobType
    status: str
    progress: int
    current_step: str | None
    config: dict[str, Any]
    summary: dict[str, Any]
    items: tuple[dict[str, Any], ...]
    trials: tuple[dict[str, Any], ...]
    error: dict[str, Any] | None
    submitted_at: datetime
    completed_at: datetime | None


class StressJobNotFoundError(LookupError):
    """Raised when stress job does not exist."""


class StressInputError(ValueError):
    """Raised on invalid stress input."""


class StressStrategyNotFoundError(LookupError):
    """Raised when strategy cannot be resolved for create request."""


@dataclass(frozen=True, slots=True)
class _BacktestRunContext:
    """Preloaded data context reused across parameter variants."""

    market: str
    symbol: str
    timeframe: str
    start: datetime
    end: datetime
    frame: pd.DataFrame


def list_black_swan_windows(*, market: str) -> list[dict[str, Any]]:
    """Public helper for MCP list-windows tool."""

    return serialize_windows(list_windows(market=market))


async def create_stress_job(
    db: AsyncSession,
    *,
    job_type: StressJobType,
    strategy_id: UUID | None,
    backtest_job_id: UUID | None,
    config: dict[str, Any] | None,
    user_id: UUID | None,
    auto_commit: bool = True,
) -> StressJobReceipt:
    """Create one queued stress job."""

    if strategy_id is None and backtest_job_id is None:
        raise StressInputError("Provide strategy_id or backtest_job_id")

    if strategy_id is not None:
        strategy = await db.scalar(select(Strategy).where(Strategy.id == strategy_id))
    else:
        strategy = None

    base_job: BacktestJob | None = None
    if backtest_job_id is not None:
        base_job = await db.scalar(select(BacktestJob).where(BacktestJob.id == backtest_job_id))
        if base_job is None:
            raise StressStrategyNotFoundError(f"Backtest job not found: {backtest_job_id}")
        if strategy is None:
            strategy = await db.scalar(select(Strategy).where(Strategy.id == base_job.strategy_id))

    if strategy is None:
        raise StressStrategyNotFoundError(f"Strategy not found: {strategy_id}")
    if user_id is not None and strategy.user_id != user_id:
        raise StressStrategyNotFoundError(f"Strategy not found: {strategy.id}")

    normalized_type = _normalize_job_type(job_type)
    normalized_config = _normalize_create_config(job_type=normalized_type, config=config or {})

    job = StressJob(
        user_id=strategy.user_id,
        strategy_id=strategy.id,
        base_backtest_job_id=base_job.id if base_job is not None else None,
        job_type=normalized_type,
        status="queued",
        progress=0,
        current_step="queued",
        config=normalized_config,
        summary={},
        error_message=None,
    )
    db.add(job)
    await db.flush()

    if auto_commit:
        await db.commit()
        await db.refresh(job)

    return StressJobReceipt(
        job_id=job.id,
        strategy_id=job.strategy_id,
        job_type=normalized_type,
        status=_to_external_status(job.status),
        progress=int(job.progress),
    )


async def execute_stress_job_with_fresh_session(job_id: UUID) -> StressJobView:
    """Execute one stress job with dedicated DB session."""

    if db_module.AsyncSessionLocal is None:
        await db_module.init_postgres(ensure_schema=False)
    assert db_module.AsyncSessionLocal is not None

    async with db_module.AsyncSessionLocal() as session:
        return await execute_stress_job(session, job_id=job_id, auto_commit=True)


def _enqueue_stress_job(job_id: UUID) -> str:
    from src.workers.stress_tasks import enqueue_stress_job

    return enqueue_stress_job(job_id)


async def schedule_stress_job(job_id: UUID) -> str:
    """Schedule one stress job on Celery queue."""

    task_id = _enqueue_stress_job(job_id)
    logger.info("[stress] enqueued job_id=%s celery_task_id=%s", job_id, task_id)
    return task_id


async def execute_stress_job(
    db: AsyncSession,
    *,
    job_id: UUID,
    auto_commit: bool = True,
) -> StressJobView:
    """Execute queued stress job and persist result artifacts."""

    job = await _load_job_for_run(db, job_id=job_id)
    if job.status == "completed":
        return _to_view(job)

    strategy = await db.scalar(select(Strategy).where(Strategy.id == job.strategy_id))
    if strategy is None:
        raise StressJobNotFoundError(f"strategy not found for stress job: {job.strategy_id}")

    payload = strategy.dsl_payload if isinstance(strategy.dsl_payload, dict) else {}
    job.status = "running"
    job.progress = 5
    job.current_step = "initializing"
    job.error_message = None
    await _commit_if_requested(db, auto_commit=auto_commit)

    try:
        if job.job_type == "black_swan":
            await _run_black_swan_job(db=db, job=job, strategy_payload=payload)
        elif job.job_type == "monte_carlo":
            await _run_monte_carlo_job(db=db, job=job, strategy_payload=payload)
        elif job.job_type == "param_scan":
            await _run_param_scan_job(db=db, job=job, strategy_payload=payload)
        elif job.job_type == "optimization":
            await _run_optimization_job(db=db, job=job, strategy_payload=payload)
        else:
            raise StressInputError(f"Unsupported job_type: {job.job_type}")

        job.status = "completed"
        job.progress = 100
        job.current_step = "completed"
        job.completed_at = datetime.now(UTC)
        job.error_message = None
    except Exception as exc:  # noqa: BLE001
        job.status = "failed"
        job.progress = 100
        job.current_step = "failed"
        job.completed_at = datetime.now(UTC)
        job.error_message = f"{type(exc).__name__}: {exc}"

    await _commit_if_requested(db, auto_commit=auto_commit)
    refreshed = await _load_job_for_run(db, job_id=job.id)
    return _to_view(refreshed)


async def get_stress_job_view(
    db: AsyncSession,
    *,
    job_id: UUID,
    user_id: UUID | None = None,
) -> StressJobView:
    """Load one stress job view with optional user ownership check."""

    job = await _load_job_for_run(db, job_id=job_id)
    if user_id is not None and job.user_id != user_id:
        raise StressJobNotFoundError(f"stress job not found: {job_id}")
    return _to_view(job)


async def get_optimization_pareto_points(
    db: AsyncSession,
    *,
    job_id: UUID,
    x_metric: str,
    y_metric: str,
    user_id: UUID | None = None,
) -> list[dict[str, Any]]:
    """Project optimization trials into x/y metric points."""

    view = await get_stress_job_view(db, job_id=job_id, user_id=user_id)
    if view.job_type != "optimization":
        raise StressInputError("stress_optimize_get_pareto only supports optimization jobs")
    return build_metric_points(rows=list(view.trials), x_metric=x_metric, y_metric=y_metric)


async def _run_black_swan_job(
    *,
    db: AsyncSession,
    job: StressJob,
    strategy_payload: dict[str, Any],
) -> None:
    config = dict(job.config or {})
    parsed = parse_strategy_payload(strategy_payload)
    market = parsed.universe.market

    windows = resolve_window_set(
        market=market,
        window_set=str(config.get("window_set", "default")),
        custom_windows=config.get("custom_windows") if isinstance(config.get("custom_windows"), list) else None,
    )

    metrics_threshold = dict(config.get("thresholds") or {})
    min_return_pct = float(metrics_threshold.get("min_return_pct", -20.0))
    max_drawdown_pct = float(metrics_threshold.get("max_drawdown_pct", 35.0))

    results, aggregate = run_black_swan_analysis(
        strategy_payload=strategy_payload,
        windows=windows,
        min_return_pct=min_return_pct,
        max_drawdown_pct=max_drawdown_pct,
        backtest_config=_backtest_config_from_job(job),
    )

    await _delete_job_items(db, job_id=job.id)
    items: list[StressJobItem] = []
    for index, item in enumerate(results):
        items.append(
            StressJobItem(
                job_id=job.id,
                item_type="window",
                item_index=index,
                labels={
                    "window_id": item.window_id,
                    "window_label": item.window_label,
                },
                metrics={
                    "return_pct": item.return_pct,
                    "max_drawdown_pct": item.max_drawdown_pct,
                    "win_rate": item.win_rate,
                    "trade_count": item.trade_count,
                    "sharpe": item.sharpe,
                    "pass": item.pass_window,
                },
                artifacts={
                    "start": item.start,
                    "end": item.end,
                    "error": item.error,
                },
            )
        )
    db.add_all(items)

    job.summary = {
        "window_results": [serialize_window_result(item) for item in results],
        "aggregate_score": aggregate.aggregate_score,
        "pass_fail": aggregate.pass_fail,
        "pass_count": aggregate.pass_count,
        "total_count": aggregate.total_count,
    }
    job.progress = 95
    job.current_step = "finalizing_black_swan"


async def _run_monte_carlo_job(
    *,
    db: AsyncSession,
    job: StressJob,
    strategy_payload: dict[str, Any],
) -> None:
    config = dict(job.config or {})
    num_trials = max(100, int(config.get("num_trials", 2000)))
    horizon_bars = max(20, int(config.get("horizon_bars", 252)))
    method = str(config.get("method", "block_bootstrap")).strip().lower()
    if method not in {"iid_bootstrap", "block_bootstrap", "trade_shuffle"}:
        raise StressInputError(f"Unsupported monte_carlo method: {method}")

    backtest = _run_backtest(strategy_payload=strategy_payload, job=job)
    simulation = run_monte_carlo(
        returns=list(backtest.returns),
        num_trials=num_trials,
        horizon_bars=horizon_bars,
        method=method,  # type: ignore[arg-type]
        seed=_seed_from_config(config),
    )

    final_returns = simulation.final_returns_pct
    distribution = histogram(final_returns.tolist(), bins=40)
    ci = confidence_intervals(final_returns.tolist())
    ruin_threshold = float(config.get("ruin_threshold_pct", -30.0))
    ruin_prob = risk_of_ruin(final_returns.tolist(), ruin_threshold_pct=ruin_threshold)

    sample_trials = final_returns[: min(200, len(final_returns))]
    await _delete_job_items(db, job_id=job.id)
    db.add_all(
        [
            StressJobItem(
                job_id=job.id,
                item_type="trial",
                item_index=index,
                labels={"trial_no": index + 1},
                metrics={"final_return_pct": float(value)},
                artifacts={},
            )
            for index, value in enumerate(sample_trials)
        ]
    )

    job.summary = {
        "distribution": distribution,
        "percentiles": percentile_summary(final_returns.tolist()),
        "ci": ci,
        "risk_of_ruin": ruin_prob,
        "tail_metrics": {
            "var_95": value_at_risk(final_returns.tolist(), confidence=0.95),
            "cvar_95": conditional_value_at_risk(final_returns.tolist(), confidence=0.95),
        },
        "avg_path": [float(value) for value in simulation.avg_path.tolist()],
        "method": simulation.method,
        "num_trials": simulation.num_trials,
        "horizon_bars": simulation.horizon_bars,
    }
    job.progress = 95
    job.current_step = "finalizing_monte_carlo"


async def _run_param_scan_job(
    *,
    db: AsyncSession,
    job: StressJob,
    strategy_payload: dict[str, Any],
) -> None:
    config = dict(job.config or {})
    scan_pct = float(config.get("scan_pct", 10.0))
    steps_per_side = max(1, int(config.get("steps_per_side", 3)))
    target_params = config.get("target_params") if isinstance(config.get("target_params"), list) else None

    params = list_tunable_params(strategy_payload, target_params=target_params)
    if not params:
        raise StressInputError("No tunable params found for param sensitivity scan")

    context = _build_backtest_context(strategy_payload=strategy_payload, job=job)
    base_backtest = _run_backtest(
        strategy_payload=strategy_payload,
        job=job,
        context=context,
    )
    base_metrics = _metrics_from_backtest(base_backtest)

    def _evaluator(candidate_payload: dict[str, Any]) -> dict[str, Any]:
        result = _run_backtest(
            strategy_payload=candidate_payload,
            job=job,
            context=context,
        )
        metrics = _metrics_from_backtest(result)
        return metrics

    variants, fragile_rank, stability_score = scan_param_sensitivity(
        base_payload=strategy_payload,
        params=params,
        scan_pct=scan_pct,
        steps_per_side=steps_per_side,
        evaluator=_evaluator,
    )

    await _delete_job_items(db, job_id=job.id)
    db.add_all(
        [
            StressJobItem(
                job_id=job.id,
                item_type="param_variant",
                item_index=index,
                labels={
                    "param_key": item.key,
                    "factor_id": item.factor_id,
                    "param_name": item.param_name,
                    "pct_delta": item.pct_delta,
                },
                metrics=item.metrics,
                artifacts={"value": item.value},
            )
            for index, item in enumerate(variants)
        ]
    )

    job.summary = {
        "base_metrics": base_metrics,
        "fragile_params_rank": fragile_rank,
        "param_response_curves": _curves_from_variants(variants),
        "stability_score": stability_score,
    }
    job.progress = 95
    job.current_step = "finalizing_param_scan"


async def _run_optimization_job(
    *,
    db: AsyncSession,
    job: StressJob,
    strategy_payload: dict[str, Any],
) -> None:
    config = dict(job.config or {})
    method = str(config.get("method", "random")).strip().lower()
    if method not in {"grid", "random", "bayes"}:
        raise StressInputError("method must be one of: grid, random, bayes")

    budget = max(1, int(config.get("budget", 40)))
    objectives = [
        str(item).strip().lower()
        for item in (config.get("objectives") or [])
        if isinstance(item, str) and item.strip()
    ]
    constraints = config.get("constraints") if isinstance(config.get("constraints"), dict) else {}

    tunables = list_tunable_params(strategy_payload, target_params=None)
    dimensions = build_search_space(
        raw_search_space=config.get("search_space") if isinstance(config.get("search_space"), dict) else None,
        tunable_params=tunables,
    )

    seed = _seed_from_config(config)
    if method == "grid":
        candidates = generate_grid_candidates(dimensions=dimensions, budget=budget)
    elif method == "random":
        candidates = generate_random_candidates(dimensions=dimensions, budget=budget, seed=seed)
    else:
        candidates = generate_bayes_like_candidates(dimensions=dimensions, budget=budget, seed=seed)

    await _delete_job_items(db, job_id=job.id)
    await _delete_optimization_trials(db, job_id=job.id)

    trial_rows: list[dict[str, Any]] = []
    persisted_trials: list[OptimizationTrial] = []
    context = _build_backtest_context(strategy_payload=strategy_payload, job=job)

    for trial_no, params in enumerate(candidates, start=1):
        mutated_payload = apply_param_values(strategy_payload, values=params)
        result = _run_backtest(
            strategy_payload=mutated_payload,
            job=job,
            context=context,
        )
        metrics = _metrics_from_backtest(result)
        metrics["stability_score"] = annualized_stability_score(result.returns)

        objective_values, objective_score = compute_objective_score(
            metrics=metrics,
            objectives=objectives,
        )
        passed_constraints = constraints_satisfied(metrics=metrics, constraints=constraints)

        row = {
            "trial_no": trial_no,
            "params": params,
            "metrics": metrics,
            "objective_values": objective_values,
            "objective_score": float(objective_score),
            "passed_constraints": passed_constraints,
        }
        trial_rows.append(row)

        persisted_trials.append(
            OptimizationTrial(
                job_id=job.id,
                trial_no=trial_no,
                method=method,
                params=params,
                metrics={
                    **metrics,
                    "objective_score": float(objective_score),
                    "passed_constraints": passed_constraints,
                },
                stability_score=float(metrics.get("stability_score", 0.0)),
                is_pareto=False,
            )
        )

        if trial_no % 5 == 0:
            job.progress = min(90, int((trial_no / len(candidates)) * 80) + 10)
            job.current_step = f"optimization_trial_{trial_no}_of_{len(candidates)}"
            await db.flush()

    if not trial_rows:
        raise StressInputError("No optimization candidates generated")

    objective_keys = list(trial_rows[0]["objective_values"].keys()) if trial_rows else []
    pareto_indices = pareto_front_indices(rows=trial_rows, objective_keys=objective_keys)

    pareto_points: list[dict[str, Any]] = []
    for index, row in enumerate(trial_rows):
        is_pareto = index in pareto_indices
        persisted_trials[index].is_pareto = is_pareto
        if is_pareto:
            pareto_points.append(
                {
                    "trial_no": row["trial_no"],
                    "params": row["params"],
                    "metrics": row["metrics"],
                    "objective_values": row["objective_values"],
                }
            )

    db.add_all(persisted_trials)
    db.add_all(
        [
            StressJobItem(
                job_id=job.id,
                item_type="pareto_point",
                item_index=index,
                labels={"trial_no": point["trial_no"]},
                metrics=point["metrics"],
                artifacts={
                    "params": point["params"],
                    "objective_values": point["objective_values"],
                },
            )
            for index, point in enumerate(pareto_points)
        ]
    )

    feasible = [row for row in trial_rows if row["passed_constraints"]]
    ranked = sorted(
        feasible or trial_rows,
        key=lambda item: float(item.get("objective_score", 0.0)),
        reverse=True,
    )
    best = ranked[0]

    top_trials = ranked[: min(8, len(ranked))]
    pareto_frontier = [
        {
            "trial_no": row["trial_no"],
            "params": row["params"],
            "metrics": row["metrics"],
            "objective_values": row["objective_values"],
        }
        for row in pareto_points
    ]

    stability_frontier = sorted(
        [
            {
                "trial_no": row["trial_no"],
                "params": row["params"],
                "stability_score": float(row["metrics"].get("stability_score", 0.0)),
                "total_return_pct": float(row["metrics"].get("total_return_pct", 0.0)),
            }
            for row in trial_rows
        ],
        key=lambda item: (item["stability_score"], item["total_return_pct"]),
        reverse=True,
    )[: min(10, len(trial_rows))]

    job.summary = {
        "best_trials": top_trials,
        "pareto_frontier": pareto_frontier,
        "stability_frontier": stability_frontier,
        "recommended_params": best["params"],
        "recommended_trial_no": best["trial_no"],
        "method": method,
        "budget": budget,
    }
    job.progress = 95
    job.current_step = "finalizing_optimization"


def _metrics_from_backtest(result: BacktestResult) -> dict[str, Any]:
    performance = result.performance if isinstance(result.performance, dict) else {}
    metrics = performance.get("metrics") if isinstance(performance.get("metrics"), dict) else {}

    return {
        "total_return_pct": float(result.summary.total_return_pct),
        "max_drawdown_pct": abs(float(result.summary.max_drawdown_pct)),
        "win_rate": float(result.summary.win_rate),
        "total_trades": int(result.summary.total_trades),
        "final_equity": float(result.summary.final_equity),
        "sharpe": float(metrics.get("sharpe") or 0.0),
    }


def _curves_from_variants(variants: list[Any]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in variants:
        grouped.setdefault(item.key, []).append(
            {
                "pct_delta": item.pct_delta,
                "value": item.value,
                "metrics": item.metrics,
            }
        )

    output: list[dict[str, Any]] = []
    for key, points in grouped.items():
        output.append(
            {
                "param_key": key,
                "points": sorted(points, key=lambda row: float(row["pct_delta"])),
            }
        )
    output.sort(key=lambda row: str(row["param_key"]))
    return output


def _build_backtest_context(
    *,
    strategy_payload: dict[str, Any],
    job: StressJob,
) -> _BacktestRunContext:
    parsed = build_parsed_strategy(strategy_payload)
    if not parsed.universe.tickers:
        raise StressInputError("Strategy universe.tickers cannot be empty")

    symbol = parsed.universe.tickers[0]
    market = parsed.universe.market
    timeframe = parsed.universe.timeframe
    loader = DataLoader()

    config = dict(job.config or {})
    maybe_start = str(config.get("start_date", "")).strip()
    maybe_end = str(config.get("end_date", "")).strip()
    if maybe_start and maybe_end:
        start = _parse_datetime(maybe_start)
        end = _parse_datetime(maybe_end)
    else:
        metadata = loader.get_symbol_metadata(market, symbol)
        available = metadata.get("available_timerange", {}) if isinstance(metadata, dict) else {}
        start = _parse_datetime(str(available.get("start")))
        end = _parse_datetime(str(available.get("end")))

    frame = loader.load(
        market=market,
        symbol=symbol,
        timeframe=timeframe,
        start_date=start,
        end_date=end,
    )
    return _BacktestRunContext(
        market=market,
        symbol=symbol,
        timeframe=timeframe,
        start=start,
        end=end,
        frame=frame,
    )


def _run_backtest(
    *,
    strategy_payload: dict[str, Any],
    job: StressJob,
    context: _BacktestRunContext | None = None,
) -> BacktestResult:
    # Stress mutations only tweak numeric params on an already validated strategy.
    # Keep factor ids stable even when period/fast/slow values are perturbed.
    parsed = build_parsed_strategy(strategy_payload)
    resolved_context = context or _build_backtest_context(
        strategy_payload=strategy_payload,
        job=job,
    )

    return EventDrivenBacktestEngine(
        strategy=parsed,
        data=resolved_context.frame,
        config=_backtest_config_from_job(job),
    ).run()


def _backtest_config_from_job(job: StressJob) -> BacktestConfig:
    config = dict(job.config or {})
    return BacktestConfig(
        initial_capital=float(config.get("initial_capital", 100000.0)),
        commission_rate=float(config.get("commission_rate", 0.0)),
        slippage_bps=float(config.get("slippage_bps", 0.0)),
        record_bar_events=False,
        performance_series_max_points=max(500, int(config.get("performance_series_max_points", 3000))),
    )


def _seed_from_config(config: dict[str, Any]) -> int:
    raw = config.get("seed", settings.stress_default_seed)
    try:
        return int(raw)
    except Exception:  # noqa: BLE001
        return int(settings.stress_default_seed)


def _normalize_job_type(value: str) -> StressJobType:
    normalized = str(value).strip().lower()
    if normalized not in {"black_swan", "monte_carlo", "param_scan", "optimization"}:
        raise StressInputError(f"Unsupported job_type: {value}")
    return normalized  # type: ignore[return-value]


def _normalize_create_config(*, job_type: StressJobType, config: dict[str, Any]) -> dict[str, Any]:
    base = dict(config)
    if job_type == "black_swan":
        base.setdefault("window_set", "default")
        base.setdefault(
            "thresholds",
            {
                "min_return_pct": -20.0,
                "max_drawdown_pct": 35.0,
            },
        )
    elif job_type == "monte_carlo":
        base.setdefault("num_trials", settings.stress_monte_carlo_default_trials)
        base.setdefault("horizon_bars", settings.stress_monte_carlo_default_horizon_bars)
        base.setdefault("method", settings.stress_monte_carlo_default_method)
        base.setdefault("ruin_threshold_pct", -30.0)
        base.setdefault("seed", settings.stress_default_seed)
    elif job_type == "param_scan":
        base.setdefault("scan_pct", 10.0)
        base.setdefault("steps_per_side", 3)
        base.setdefault("target_params", [])
    elif job_type == "optimization":
        base.setdefault("method", "random")
        base.setdefault("budget", settings.optimization_default_budget)
        base.setdefault("objectives", ["max_return", "min_drawdown", "max_stability"])
        base.setdefault("constraints", {})
        base.setdefault("search_space", {})
        base.setdefault("seed", settings.stress_default_seed)
    return base


def _to_view(job: StressJob) -> StressJobView:
    items = tuple(
        {
            "item_type": item.item_type,
            "item_index": int(item.item_index),
            "labels": dict(item.labels or {}),
            "metrics": dict(item.metrics or {}),
            "artifacts": dict(item.artifacts or {}),
        }
        for item in sorted(job.items or [], key=lambda row: (row.item_type, row.item_index))
    )
    trials = tuple(
        {
            "trial_no": int(item.trial_no),
            "method": item.method,
            "params": dict(item.params or {}),
            "metrics": dict(item.metrics or {}),
            "stability_score": float(item.stability_score) if item.stability_score is not None else 0.0,
            "is_pareto": bool(item.is_pareto),
        }
        for item in sorted(job.optimization_trials or [], key=lambda row: row.trial_no)
    )

    error_payload = None
    if job.status == "failed":
        error_payload = {
            "code": "JOB_FAILED",
            "message": job.error_message or "Stress job failed",
        }

    return StressJobView(
        job_id=job.id,
        strategy_id=job.strategy_id,
        base_backtest_job_id=job.base_backtest_job_id,
        job_type=_normalize_job_type(job.job_type),
        status=_to_external_status(job.status),
        progress=int(job.progress),
        current_step=job.current_step,
        config=dict(job.config or {}),
        summary=dict(job.summary or {}),
        items=items,
        trials=trials,
        error=error_payload,
        submitted_at=_ensure_utc(job.submitted_at),
        completed_at=_ensure_utc(job.completed_at) if job.completed_at else None,
    )


async def _load_job_for_run(db: AsyncSession, *, job_id: UUID) -> StressJob:
    job = await db.scalar(
        select(StressJob)
        .options(
            selectinload(StressJob.items),
            selectinload(StressJob.optimization_trials),
        )
        .where(StressJob.id == job_id)
    )
    if job is None:
        raise StressJobNotFoundError(f"stress job not found: {job_id}")
    return job


async def _delete_job_items(db: AsyncSession, *, job_id: UUID) -> None:
    rows = list(
        (
            await db.scalars(
                select(StressJobItem).where(StressJobItem.job_id == job_id)
            )
        ).all()
    )
    for item in rows:
        await db.delete(item)


async def _delete_optimization_trials(db: AsyncSession, *, job_id: UUID) -> None:
    rows = list(
        (
            await db.scalars(
                select(OptimizationTrial).where(OptimizationTrial.job_id == job_id)
            )
        ).all()
    )
    for item in rows:
        await db.delete(item)


def _parse_datetime(value: str) -> datetime:
    text = str(value).strip()
    if not text:
        raise StressInputError("datetime value cannot be empty")
    normalized = text.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    return _ensure_utc(parsed)


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _to_external_status(status: str) -> str:
    return _INTERNAL_TO_EXTERNAL_STATUS.get(status, status)


async def _commit_if_requested(db: AsyncSession, *, auto_commit: bool) -> None:
    if not auto_commit:
        return
    await db.commit()
