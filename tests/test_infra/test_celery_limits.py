from __future__ import annotations

from src.config import settings
from src.workers.celery_app import celery_app


def test_celery_worker_time_and_memory_limits_match_settings() -> None:
    assert celery_app.conf.task_time_limit == settings.celery_task_time_limit_seconds
    assert celery_app.conf.task_soft_time_limit == settings.celery_task_soft_time_limit_seconds
    assert (
        celery_app.conf.worker_max_memory_per_child
        == settings.celery_worker_max_memory_per_child
    )
