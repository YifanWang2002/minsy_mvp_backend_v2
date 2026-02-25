from __future__ import annotations

from src.config import settings
from src.workers.celery_app import celery_app


def test_celery_includes_trade_approval_tasks_module() -> None:
    include = celery_app.conf.include or ()
    assert "src.workers.trade_approval_tasks" in include


def test_celery_has_trade_approval_queue_and_routes() -> None:
    queues = celery_app.conf.task_queues or ()
    queue_names = {queue.name for queue in queues}
    assert "trade_approval" in queue_names

    routes = celery_app.conf.task_routes or {}
    assert routes["trade_approval.*"]["queue"] == "trade_approval"


def test_celery_trade_approval_expire_schedule_matches_settings() -> None:
    schedule = celery_app.conf.beat_schedule or {}
    key = "trade_approval.expire_pending"
    if settings.trading_approval_enabled:
        assert key in schedule
        assert schedule[key]["task"] == "trade_approval.expire_pending"
    else:
        assert key not in schedule
