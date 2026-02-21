from __future__ import annotations

import asyncio
import csv
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from src.workers import maintenance_tasks


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def test_prune_old_backups_keeps_latest_by_filename(tmp_path: Path) -> None:
    backup_names = [
        "minsy_20260101_000000.dump",
        "minsy_20260102_000000.dump",
        "minsy_20260103_000000.dump",
    ]
    for name in backup_names:
        (tmp_path / name).write_text("x", encoding="utf-8")

    deleted = maintenance_tasks._prune_old_backups(tmp_path, keep_count=2)

    assert deleted == 1
    remaining = sorted(path.name for path in tmp_path.glob("*.dump"))
    assert remaining == backup_names[1:]


def test_export_user_emails_csv_keeps_history_and_deduplicates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv_path = tmp_path / "emails.csv"
    csv_path.write_text(
        "\n".join(
            [
                "email,first_seen_at_utc",
                "old@example.com,2026-02-17T00:00:00Z",
                "OLD@EXAMPLE.COM,2026-02-17T00:00:00Z",
            ]
        ),
        encoding="utf-8",
    )

    async def _fake_fetch_current_user_emails() -> set[str]:
        return {"new@example.com"}

    monkeypatch.setattr(
        maintenance_tasks,
        "_fetch_current_user_emails",
        _fake_fetch_current_user_emails,
    )
    monkeypatch.setattr(maintenance_tasks.settings, "user_email_csv_path", str(csv_path))

    payload = maintenance_tasks.export_user_emails_csv_task.run()
    rows = _read_csv_rows(csv_path)
    emails = sorted(row["email"] for row in rows)

    assert payload["total_emails"] == 2
    assert payload["newly_added"] == 1
    assert emails == ["new@example.com", "old@example.com"]


def test_backup_postgres_full_task_creates_backup_and_prunes_old(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("minsy_20260101_000000.dump", "minsy_20260102_000000.dump"):
        (tmp_path / name).write_text("old", encoding="utf-8")

    def _fake_run_pg_dump(target_file: Path) -> None:
        target_file.write_bytes(b"new-backup")

    monkeypatch.setattr(maintenance_tasks, "_run_pg_dump", _fake_run_pg_dump)
    monkeypatch.setattr(maintenance_tasks, "_backup_filename", lambda: "minsy_new.dump")
    monkeypatch.setattr(maintenance_tasks.settings, "postgres_backup_dir", str(tmp_path))
    monkeypatch.setattr(maintenance_tasks.settings, "postgres_backup_retention_count", 2)

    payload = maintenance_tasks.backup_postgres_full_task.run()
    backup_files = sorted(path.name for path in tmp_path.glob("*.dump"))

    assert payload["backup_file"].endswith("minsy_new.dump")
    assert payload["size_bytes"] == 10
    assert payload["retention_pruned"] == 1
    assert backup_files == ["minsy_20260102_000000.dump", "minsy_new.dump"]


def test_fail_stale_backtest_jobs_marks_running_jobs_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stale_job = SimpleNamespace(
        id=uuid4(),
        status="running",
        progress=5,
        current_step="running_engine",
        completed_at=None,
        error_message=None,
        results=None,
    )

    class _FakeScalarResult:
        def __init__(self, jobs):
            self._jobs = jobs

        def all(self):
            return self._jobs

    class _FakeSession:
        def __init__(self, jobs):
            self._jobs = jobs
            self.committed = False

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):  # noqa: ANN001
            return False

        async def scalars(self, _query):  # noqa: ANN001
            return _FakeScalarResult(self._jobs)

        async def commit(self):
            self.committed = True

    fake_session = _FakeSession([stale_job])

    class _FakeFactory:
        def __call__(self):
            return fake_session

    async def _noop(*args, **kwargs) -> None:  # noqa: ANN002, ANN003
        return None

    monkeypatch.setattr(maintenance_tasks.db_module, "close_postgres", _noop)
    monkeypatch.setattr(maintenance_tasks.db_module, "init_postgres", _noop)
    monkeypatch.setattr(maintenance_tasks.db_module, "AsyncSessionLocal", _FakeFactory())
    monkeypatch.setattr(maintenance_tasks.settings, "backtest_running_stale_minutes", 30)

    payload = asyncio.run(maintenance_tasks._fail_stale_backtest_jobs_once())

    assert payload["failed_jobs"] == 1
    assert payload["job_ids"] == [str(stale_job.id)]
    assert stale_job.status == "failed"
    assert stale_job.progress == 100
    assert stale_job.current_step == "failed"
    assert stale_job.completed_at is not None
    assert "timed out" in str(stale_job.error_message)
    assert stale_job.results["error"]["code"] == "BACKTEST_STALE_RUNNING"
    assert fake_session.committed is True


def test_fail_stale_backtest_jobs_task_wraps_async_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_cleanup() -> dict[str, int | list[str]]:
        return {
            "threshold_minutes": 30,
            "failed_jobs": 0,
            "job_ids": [],
        }

    monkeypatch.setattr(maintenance_tasks, "_fail_stale_backtest_jobs_once", _fake_cleanup)
    payload = maintenance_tasks.fail_stale_backtest_jobs_task.run()
    assert payload["failed_jobs"] == 0
    assert payload["job_ids"] == []
