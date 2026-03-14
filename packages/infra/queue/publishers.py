"""Celery-backed implementations for engine queue ports."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from packages.domain.ports.queue_ports import JobQueuePorts, configure_job_queue_ports
from packages.infra.observability.logger import logger
from packages.infra.queue.celery_app import celery_app
from packages.shared_settings.schema.settings import settings


def enqueue_paper_trading_runtime(deployment_id: UUID | str) -> str:
    result = celery_app.send_task(
        "paper_trading.run_deployment_runtime",
        args=(str(deployment_id),),
        queue="paper_trading",
    )
    return str(result.id)


def enqueue_execute_manual_trade_action(
    action_id: UUID | str,
    *,
    countdown_seconds: int | None = None,
) -> str | None:
    kwargs: dict[str, Any] = {}
    if countdown_seconds is not None and countdown_seconds > 0:
        kwargs["countdown"] = countdown_seconds
    try:
        result = celery_app.send_task(
            "paper_trading.execute_manual_trade_action",
            args=(str(action_id),),
            queue=settings.paper_trading_manual_action_queue,
            **kwargs,
        )
    except Exception:  # noqa: BLE001
        logger.exception(
            "[queue.publishers] enqueue manual trade action failed action_id=%s",
            action_id,
        )
        return None
    return str(result.id)


def enqueue_market_data_refresh(
    *,
    market: str,
    symbol: str,
    requested_timeframe: str | None = None,
    min_bars: int | None = None,
) -> str:
    result = celery_app.send_task(
        "market_data.refresh_symbol",
        args=(market, symbol, requested_timeframe, min_bars),
        queue="market_data",
    )
    return str(result.id)


def enqueue_reconcile_billing_usage_event(payload: dict[str, Any]) -> str | None:
    try:
        result = celery_app.send_task(
            "maintenance.reconcile_billing_usage_event",
            args=(payload,),
            queue="maintenance",
        )
    except Exception:  # noqa: BLE001
        logger.exception(
            "[queue.publishers] enqueue billing usage reconcile failed payload_keys=%s",
            sorted(payload.keys()),
        )
        return None
    return str(result.id)


def enqueue_execute_approved_open(request_id: UUID | str) -> str | None:
    try:
        result = celery_app.send_task(
            "trade_approval.execute_approved_open",
            args=(str(request_id),),
            queue="trade_approval",
        )
    except Exception:  # noqa: BLE001
        logger.exception(
            "[queue.publishers] enqueue trade approval failed request_id=%s",
            request_id,
        )
        return None
    return str(result.id)


def enqueue_backtest_job(job_id: UUID | str) -> str:
    result = celery_app.send_task(
        "backtest.execute_job",
        args=(str(job_id),),
        queue="backtest",
    )
    return str(result.id)


def enqueue_stress_job(job_id: UUID | str) -> str:
    result = celery_app.send_task(
        "stress.execute_job",
        args=(str(job_id),),
        queue="stress",
    )
    return str(result.id)


def enqueue_market_data_sync_job(job_id: UUID | str) -> str:
    result = celery_app.send_task(
        "market_data.sync_missing_ranges",
        args=(str(job_id),),
        queue="market_data",
    )
    return str(result.id)


def enqueue_market_data_incremental_import_job(job_id: UUID | str) -> str:
    result = celery_app.send_task(
        "market_data.import_incremental_batch",
        args=(str(job_id),),
        queue="market_data",
    )
    return str(result.id)


def configure_default_job_queue_ports() -> None:
    """Wire domain queue ports to Celery enqueue functions."""

    configure_job_queue_ports(
        JobQueuePorts(
            enqueue_backtest_job=lambda job_id: enqueue_backtest_job(job_id),
            enqueue_stress_job=lambda job_id: enqueue_stress_job(job_id),
            enqueue_market_data_sync_job=lambda job_id: enqueue_market_data_sync_job(job_id),
        )
    )
