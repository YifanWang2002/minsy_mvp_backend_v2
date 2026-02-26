"""Celery-backed implementations for engine queue ports."""

from __future__ import annotations

from uuid import UUID

from packages.domain.ports.queue_ports import JobQueuePorts, configure_job_queue_ports
from packages.infra.observability.logger import logger
from packages.infra.queue.celery_app import celery_app


def enqueue_paper_trading_runtime(deployment_id: UUID | str) -> str:
    result = celery_app.send_task(
        "paper_trading.run_deployment_runtime",
        args=(str(deployment_id),),
        queue="paper_trading",
    )
    return str(result.id)


def enqueue_market_data_refresh(*, market: str, symbol: str) -> str:
    result = celery_app.send_task(
        "market_data.refresh_symbol",
        args=(market, symbol),
        queue="market_data",
    )
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


def configure_default_job_queue_ports() -> None:
    """Wire domain queue ports to Celery enqueue functions."""

    configure_job_queue_ports(
        JobQueuePorts(
            enqueue_backtest_job=lambda job_id: enqueue_backtest_job(job_id),
            enqueue_stress_job=lambda job_id: enqueue_stress_job(job_id),
            enqueue_market_data_sync_job=lambda job_id: enqueue_market_data_sync_job(job_id),
        )
    )
