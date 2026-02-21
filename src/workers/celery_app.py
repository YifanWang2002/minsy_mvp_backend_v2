"""Celery application bootstrap for background jobs."""

from __future__ import annotations

from datetime import timedelta

from celery import Celery
from celery.schedules import crontab

from src.config import settings
from src.observability.sentry_setup import init_backend_sentry

init_backend_sentry(source="celery")

_beat_schedule: dict[str, dict[str, object]] = {}

if settings.postgres_backup_enabled:
    _beat_schedule["maintenance.postgres_full_backup"] = {
        "task": "maintenance.backup_postgres_full",
        "schedule": crontab(
            hour=settings.postgres_backup_hour_utc,
            minute=settings.postgres_backup_minute_utc,
        ),
    }

if settings.user_email_csv_export_enabled:
    _beat_schedule["maintenance.user_email_csv_export"] = {
        "task": "maintenance.export_user_emails_csv",
        "schedule": timedelta(minutes=settings.user_email_csv_export_interval_minutes),
    }

celery_app = Celery(
    "minsy",
    broker=settings.effective_celery_broker_url,
    backend=settings.effective_celery_result_backend,
    include=["src.workers.backtest_tasks", "src.workers.maintenance_tasks"],
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
    # Celery uses KB for this option; recycle child process after it exceeds cap.
    worker_max_memory_per_child=settings.celery_worker_max_memory_per_child,
    task_always_eager=settings.celery_task_always_eager,
    broker_connection_retry_on_startup=True,
    timezone=settings.celery_timezone,
    enable_utc=settings.celery_timezone.strip().upper() == "UTC",
    beat_schedule=_beat_schedule,
)
