"""Celery application bootstrap for background jobs."""

from __future__ import annotations

from celery import Celery

from src.config import settings

celery_app = Celery(
    "minsy",
    broker=settings.effective_celery_broker_url,
    backend=settings.effective_celery_result_backend,
    include=["src.workers.backtest_tasks"],
)

celery_app.conf.update(
    task_default_queue=settings.celery_task_default_queue,
    task_acks_late=settings.celery_task_acks_late,
    task_track_started=True,
    worker_prefetch_multiplier=settings.celery_worker_prefetch_multiplier,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_time_limit=settings.celery_task_time_limit_seconds,
    task_soft_time_limit=settings.celery_task_soft_time_limit_seconds,
    task_always_eager=settings.celery_task_always_eager,
    broker_connection_retry_on_startup=True,
)
