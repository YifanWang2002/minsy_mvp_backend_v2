from __future__ import annotations

from src.config import settings
from src.workers.celery_app import celery_app


def test_celery_includes_paper_trading_tasks_module() -> None:
    include = celery_app.conf.include or ()
    assert "src.workers.paper_trading_tasks" in include


def test_celery_has_paper_trading_queue_and_route() -> None:
    queues = celery_app.conf.task_queues or ()
    queue_names = {queue.name for queue in queues}
    assert "paper_trading" in queue_names

    routes = celery_app.conf.task_routes or {}
    assert routes["paper_trading.*"]["queue"] == "paper_trading"


def test_celery_worker_has_time_and_memory_limits() -> None:
    assert celery_app.conf.task_time_limit == settings.celery_task_time_limit_seconds
    assert celery_app.conf.task_soft_time_limit == settings.celery_task_soft_time_limit_seconds
    assert (
        celery_app.conf.worker_max_memory_per_child
        == settings.celery_worker_max_memory_per_child
    )


def test_celery_backtest_stale_cleanup_schedule_matches_settings() -> None:
    schedule = celery_app.conf.beat_schedule or {}
    key = "maintenance.fail_stale_backtest_jobs"
    if settings.backtest_stale_job_cleanup_enabled:
        assert key in schedule
        assert schedule[key]["task"] == "maintenance.fail_stale_backtest_jobs"
    else:
        assert key not in schedule


def test_celery_paper_scheduler_tick_schedule_matches_settings() -> None:
    schedule = celery_app.conf.beat_schedule or {}
    key = "paper_trading.scheduler_tick"
    if settings.paper_trading_enabled:
        assert key in schedule
        assert schedule[key]["task"] == "paper_trading.scheduler_tick"
    else:
        assert key not in schedule


def test_celery_market_data_refresh_schedule_matches_settings() -> None:
    schedule = celery_app.conf.beat_schedule or {}
    key = "market_data.refresh_active_subscriptions"
    if settings.paper_trading_enabled:
        assert key in schedule
        assert schedule[key]["task"] == "market_data.refresh_active_subscriptions"
    else:
        assert key not in schedule
