from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID, uuid4

import pytest

from src.workers import backtest_tasks


def test_enqueue_backtest_job_submits_celery_task(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class _FakeAsyncResult:
        id = "celery-task-xyz"

    class _FakeTask:
        def apply_async(self, args=None, kwargs=None):  # noqa: ANN001
            captured["args"] = args
            captured["kwargs"] = kwargs
            return _FakeAsyncResult()

    monkeypatch.setattr(backtest_tasks, "execute_backtest_job_task", _FakeTask())

    job_id = uuid4()
    task_id = backtest_tasks.enqueue_backtest_job(job_id)

    assert task_id == "celery-task-xyz"
    assert captured["args"] == (str(job_id),)
    assert captured["kwargs"] is None


def test_execute_backtest_job_task_runs_service_with_uuid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_id = uuid4()

    @dataclass
    class _FakeView:
        job_id: UUID
        status: str
        progress: int
        current_step: str | None

    async def _fake_execute(target_job_id: UUID) -> _FakeView:
        assert target_job_id == job_id
        return _FakeView(
            job_id=target_job_id,
            status="done",
            progress=100,
            current_step="done",
        )

    monkeypatch.setattr(backtest_tasks, "execute_backtest_job_with_fresh_session", _fake_execute)

    payload = backtest_tasks.execute_backtest_job_task.run(str(job_id))
    assert payload == {
        "job_id": str(job_id),
        "status": "done",
        "progress": 100,
        "current_step": "done",
    }


def test_execute_backtest_job_task_rejects_invalid_uuid() -> None:
    with pytest.raises(ValueError):
        backtest_tasks.execute_backtest_job_task.run("not-a-uuid")


def test_execute_backtest_job_task_resets_db_resources_per_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_id = uuid4()
    observed = {"close_calls": 0}

    @dataclass
    class _FakeView:
        job_id: UUID
        status: str
        progress: int
        current_step: str | None

    async def _fake_close_postgres() -> None:
        observed["close_calls"] += 1

    async def _fake_execute(target_job_id: UUID) -> _FakeView:
        assert target_job_id == job_id
        return _FakeView(
            job_id=target_job_id,
            status="done",
            progress=100,
            current_step="done",
        )

    monkeypatch.setattr(backtest_tasks.db_module, "close_postgres", _fake_close_postgres)
    monkeypatch.setattr(backtest_tasks, "execute_backtest_job_with_fresh_session", _fake_execute)

    payload = backtest_tasks.execute_backtest_job_task.run(str(job_id))
    assert payload["status"] == "done"
    assert observed["close_calls"] == 2


def test_backtest_task_is_configured_to_avoid_oom_redelivery_loops() -> None:
    assert backtest_tasks.execute_backtest_job_task.acks_late is False
    assert backtest_tasks.execute_backtest_job_task.reject_on_worker_lost is False
