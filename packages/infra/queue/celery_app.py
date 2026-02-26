"""Celery application bootstrap for background jobs.

`MINSY_CELERY_PROFILE` selects task ownership:
- `all` (default): backward-compatible full app (api/tests/legacy entrypoints)
- `cpu`: backtest + stress workers
- `io`: paper trading + market data + approvals + notifications + maintenance workers
- `beat`: scheduler-only app (publishes periodic tasks, no heavy worker queues)
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from datetime import timedelta

from celery import Celery
from celery.schedules import crontab
from kombu import Queue

from packages.infra.observability.sentry import init_backend_sentry
from packages.shared_settings import (
    get_beat_settings,
    get_worker_cpu_settings,
    get_worker_io_settings,
)

_PROFILE_ALL = "all"
_PROFILE_CPU = "cpu"
_PROFILE_IO = "io"
_PROFILE_BEAT = "beat"
_SUPPORTED_PROFILES = {_PROFILE_ALL, _PROFILE_CPU, _PROFILE_IO, _PROFILE_BEAT}

_INCLUDES_CPU: tuple[str, ...] = (
    "apps.worker.cpu.tasks.backtest",
    "apps.worker.cpu.tasks.stress",
)
_INCLUDES_IO: tuple[str, ...] = (
    "apps.worker.io.tasks.market_data",
    "apps.worker.io.tasks.maintenance",
    "apps.worker.io.tasks.notification",
    "apps.worker.io.tasks.paper_trading",
    "apps.worker.io.tasks.trade_approval",
)
_INCLUDES_ALL: tuple[str, ...] = _INCLUDES_CPU + _INCLUDES_IO

_ROUTES_CPU: dict[str, dict[str, str]] = {
    "backtest.*": {"queue": "backtest"},
    "stress.*": {"queue": "stress"},
}
_ROUTES_IO: dict[str, dict[str, str]] = {
    "market_data.*": {"queue": "market_data"},
    "paper_trading.scheduler_tick": {"queue": "paper_trading"},
    "paper_trading.run_deployment_runtime": {"queue": "paper_trading"},
    "paper_trading.*": {"queue": "paper_trading"},
    "maintenance.*": {"queue": "maintenance"},
    "notifications.dispatch_pending": {"queue": "notifications"},
    "notifications.*": {"queue": "notifications"},
    "trade_approval.execute_approved_open": {"queue": "trade_approval"},
    "trade_approval.expire_pending": {"queue": "trade_approval"},
    "trade_approval.*": {"queue": "trade_approval"},
}
_ROUTES_ALL: dict[str, dict[str, str]] = {
    **_ROUTES_CPU,
    **_ROUTES_IO,
}

_QUEUES_CPU: tuple[Queue, ...] = (
    Queue("backtest"),
    Queue("stress"),
)
_QUEUES_IO: tuple[Queue, ...] = (
    Queue("market_data"),
    Queue("paper_trading"),
    Queue("maintenance"),
    Queue("notifications"),
    Queue("trade_approval"),
)
_QUEUES_ALL: tuple[Queue, ...] = (
    Queue("backtest"),
    Queue("market_data"),
    Queue("stress"),
    Queue("paper_trading"),
    Queue("maintenance"),
    Queue("notifications"),
    Queue("trade_approval"),
)


def _resolve_profile() -> str:
    raw = os.getenv("MINSY_CELERY_PROFILE", _PROFILE_ALL).strip().lower()
    if raw in _SUPPORTED_PROFILES:
        return raw
    return _PROFILE_ALL


def _load_settings_for_profile(profile: str):
    if profile == _PROFILE_CPU:
        return get_worker_cpu_settings()
    if profile == _PROFILE_IO:
        return get_worker_io_settings()
    if profile == _PROFILE_BEAT:
        return get_beat_settings()
    return get_worker_io_settings()


_profile = _resolve_profile()
settings = _load_settings_for_profile(_profile)
init_backend_sentry(source="celery")


def _build_beat_schedule(*, enabled: bool) -> dict[str, dict[str, object]]:
    if not enabled:
        return {}

    beat_schedule: dict[str, dict[str, object]] = {}

    if settings.postgres_backup_enabled:
        beat_schedule["maintenance.postgres_full_backup"] = {
            "task": "maintenance.backup_postgres_full",
            "schedule": crontab(
                hour=settings.postgres_backup_hour_utc,
                minute=settings.postgres_backup_minute_utc,
            ),
        }

    if settings.user_email_csv_export_enabled:
        beat_schedule["maintenance.user_email_csv_export"] = {
            "task": "maintenance.export_user_emails_csv",
            "schedule": timedelta(minutes=settings.user_email_csv_export_interval_minutes),
        }

    if settings.backtest_stale_job_cleanup_enabled:
        beat_schedule["maintenance.fail_stale_backtest_jobs"] = {
            "task": "maintenance.fail_stale_backtest_jobs",
            "schedule": timedelta(minutes=settings.backtest_stale_job_cleanup_interval_minutes),
        }

    if settings.paper_trading_enabled:
        beat_schedule["paper_trading.scheduler_tick"] = {
            "task": "paper_trading.scheduler_tick",
            "schedule": timedelta(seconds=settings.paper_trading_loop_interval_seconds),
            "options": {
                # Drop stale scheduler ticks to avoid backlog snowball under load.
                "expires": max(1, int(settings.paper_trading_loop_interval_seconds * 2)),
            },
        }
        beat_schedule["market_data.refresh_active_subscriptions"] = {
            "task": "market_data.refresh_active_subscriptions",
            "schedule": timedelta(
                seconds=settings.market_data_refresh_active_subscriptions_interval_seconds,
            ),
        }

    if settings.notifications_enabled:
        beat_schedule["notifications.dispatch_pending"] = {
            "task": "notifications.dispatch_pending",
            "schedule": timedelta(seconds=settings.notifications_loop_interval_seconds),
            "options": {
                # Skip expired dispatch ticks so notifications do not flood worker slots.
                "expires": max(1, int(settings.notifications_loop_interval_seconds)),
            },
        }

    if settings.trading_approval_enabled:
        beat_schedule["trade_approval.expire_pending"] = {
            "task": "trade_approval.expire_pending",
            "schedule": timedelta(seconds=settings.trading_approval_expire_scan_interval_seconds),
        }

    return beat_schedule


def _select_profile_config(profile: str) -> tuple[Sequence[str], tuple[Queue, ...], dict[str, dict[str, str]], bool]:
    if profile == _PROFILE_CPU:
        return _INCLUDES_CPU, _QUEUES_CPU, _ROUTES_CPU, False
    if profile == _PROFILE_IO:
        return _INCLUDES_IO, _QUEUES_IO, _ROUTES_IO, False
    if profile == _PROFILE_BEAT:
        # Beat publishes to cpu/io queues but does not run heavy worker pools itself.
        return _INCLUDES_IO, _QUEUES_ALL, _ROUTES_ALL, True
    return _INCLUDES_ALL, _QUEUES_ALL, _ROUTES_ALL, True


_include, _task_queues, _task_routes, _enable_beat = _select_profile_config(_profile)
_beat_schedule = _build_beat_schedule(enabled=_enable_beat)

celery_app = Celery(
    f"minsy-{_profile}",
    broker=settings.effective_celery_broker_url,
    backend=settings.effective_celery_result_backend,
    include=list(_include),
)

_task_annotations: dict[str, dict[str, str]] = {}
if settings.market_data_refresh_symbol_rate_limit.strip():
    _task_annotations["market_data.refresh_symbol"] = {
        "rate_limit": settings.market_data_refresh_symbol_rate_limit.strip(),
    }

celery_app.conf.update(
    task_default_queue=settings.celery_task_default_queue,
    task_acks_late=settings.celery_task_acks_late,
    task_reject_on_worker_lost=True,
    task_track_started=True,
    worker_prefetch_multiplier=settings.celery_worker_prefetch_multiplier,
    task_default_retry_delay=settings.paper_trading_broker_retry_backoff_seconds,
    task_annotations=_task_annotations,
    task_publish_retry=True,
    task_publish_retry_policy={
        "max_retries": settings.paper_trading_max_retries,
        "interval_start": 0,
        "interval_step": 0.2,
        "interval_max": 2.0,
    },
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_time_limit=settings.celery_task_time_limit_seconds,
    task_soft_time_limit=settings.celery_task_soft_time_limit_seconds,
    # Celery expects KB here; recycle child process after reaching memory cap.
    worker_max_memory_per_child=settings.celery_worker_max_memory_per_child,
    task_always_eager=settings.celery_task_always_eager,
    broker_connection_retry_on_startup=True,
    timezone=settings.celery_timezone,
    enable_utc=settings.celery_timezone.strip().upper() == "UTC",
    beat_schedule=_beat_schedule,
    task_queues=_task_queues,
    task_routes=_task_routes,
)
