from __future__ import annotations

import csv
from pathlib import Path

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
