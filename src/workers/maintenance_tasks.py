"""Celery tasks for recurring maintenance jobs."""

from __future__ import annotations

import asyncio
import csv
import os
import subprocess
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import select

from src.config import settings
from src.models.backtest import BacktestJob
from src.models import database as db_module
from src.models.user import User
from src.util.logger import logger
from src.workers.celery_app import celery_app

_BACKEND_ROOT = Path(__file__).resolve().parents[2]
_EMAIL_CSV_HEADER = ("email", "first_seen_at_utc")


def _resolve_backend_path(path_value: str) -> Path:
    candidate = Path(path_value).expanduser()
    if candidate.is_absolute():
        return candidate
    return _BACKEND_ROOT / candidate


def _normalize_email(email: str) -> str:
    return email.strip().lower()


def _backup_filename() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return f"{settings.postgres_db}_{stamp}.dump"


def _run_pg_dump(target_file: Path) -> None:
    command = [
        settings.postgres_pg_dump_bin,
        "--format=custom",
        "--clean",
        "--if-exists",
        "--no-owner",
        "--no-privileges",
        "--host",
        settings.postgres_host,
        "--port",
        str(settings.postgres_port),
        "--username",
        settings.postgres_user,
        "--dbname",
        settings.postgres_db,
        "--file",
        str(target_file),
    ]
    environment = os.environ.copy()
    if settings.postgres_password:
        environment["PGPASSWORD"] = settings.postgres_password

    try:
        subprocess.run(  # noqa: S603
            command,
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"{settings.postgres_pg_dump_bin} not found. Please install PostgreSQL client tools.",
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError(f"pg_dump failed: {stderr}") from exc


def _prune_old_backups(backup_dir: Path, keep_count: int) -> int:
    backup_files = sorted(backup_dir.glob("*.dump"))
    extra_count = len(backup_files) - keep_count
    if extra_count <= 0:
        return 0

    deleted = 0
    for old_backup in backup_files[:extra_count]:
        with suppress(FileNotFoundError):
            old_backup.unlink()
            deleted += 1
    return deleted


def _read_email_archive(csv_path: Path) -> dict[str, str]:
    if not csv_path.exists():
        return {}

    archive: dict[str, str] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            email_raw = str(row.get("email", "")).strip()
            if not email_raw:
                continue
            normalized = _normalize_email(email_raw)
            first_seen = str(row.get("first_seen_at_utc", "")).strip()
            archive.setdefault(normalized, first_seen)
    return archive


def _write_email_archive(csv_path: Path, archive: dict[str, str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(_EMAIL_CSV_HEADER)
        for email in sorted(archive):
            writer.writerow((email, archive[email]))


async def _fetch_current_user_emails() -> set[str]:
    # Celery creates per-task event loops, so reset shared asyncpg resources.
    with suppress(Exception):
        await db_module.close_postgres()

    await db_module.init_postgres(ensure_schema=False)
    assert db_module.AsyncSessionLocal is not None

    try:
        async with db_module.AsyncSessionLocal() as session:
            emails = await session.scalars(select(User.email))
            return {
                _normalize_email(email)
                for email in emails.all()
                if isinstance(email, str) and email.strip()
            }
    finally:
        with suppress(Exception):
            await db_module.close_postgres()


async def _fail_stale_backtest_jobs_once() -> dict[str, int | list[str]]:
    """Mark stale running backtest jobs as failed."""
    with suppress(Exception):
        await db_module.close_postgres()

    await db_module.init_postgres(ensure_schema=False)
    assert db_module.AsyncSessionLocal is not None

    stale_ids: list[str] = []
    try:
        cutoff = datetime.now(UTC) - timedelta(minutes=settings.backtest_running_stale_minutes)
        async with db_module.AsyncSessionLocal() as session:
            stale_jobs = (
                await session.scalars(
                    select(BacktestJob).where(
                        BacktestJob.status == "running",
                        BacktestJob.updated_at < cutoff,
                    )
                )
            ).all()

            now_utc = datetime.now(UTC)
            for job in stale_jobs:
                stale_ids.append(str(job.id))
                message = (
                    "Backtest job timed out in running state and was auto-failed by cleanup "
                    f"(threshold={settings.backtest_running_stale_minutes}m)."
                )
                job.status = "failed"
                job.progress = 100
                job.current_step = "failed"
                job.completed_at = now_utc
                job.error_message = message
                job.results = {
                    "error": {
                        "code": "BACKTEST_STALE_RUNNING",
                        "message": message,
                    }
                }

            await session.commit()
    finally:
        with suppress(Exception):
            await db_module.close_postgres()

    logger.info(
        "[maintenance] stale backtest cleanup threshold_minutes=%s failed_jobs=%s",
        settings.backtest_running_stale_minutes,
        len(stale_ids),
    )
    return {
        "threshold_minutes": settings.backtest_running_stale_minutes,
        "failed_jobs": len(stale_ids),
        "job_ids": stale_ids,
    }


@celery_app.task(name="maintenance.backup_postgres_full")
def backup_postgres_full_task() -> dict[str, str | int]:
    """Create a full PostgreSQL backup and rotate old backup files."""
    backup_dir = _resolve_backend_path(settings.postgres_backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_file = backup_dir / _backup_filename()

    _run_pg_dump(backup_file)
    deleted_count = _prune_old_backups(backup_dir, settings.postgres_backup_retention_count)
    size_bytes = backup_file.stat().st_size if backup_file.exists() else 0

    logger.info(
        "[maintenance] postgres backup done file=%s size_bytes=%s pruned=%s",
        backup_file,
        size_bytes,
        deleted_count,
    )
    return {
        "backup_file": str(backup_file),
        "size_bytes": size_bytes,
        "retention_pruned": deleted_count,
    }


@celery_app.task(name="maintenance.export_user_emails_csv")
def export_user_emails_csv_task() -> dict[str, str | int]:
    """Persist non-duplicate user emails to CSV without deleting historical rows."""
    current_emails = asyncio.run(_fetch_current_user_emails())
    csv_path = _resolve_backend_path(settings.user_email_csv_path)
    archive = _read_email_archive(csv_path)

    now_iso = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    newly_added = 0
    for email in current_emails:
        if email not in archive:
            archive[email] = now_iso
            newly_added += 1
        elif not archive[email]:
            archive[email] = now_iso

    _write_email_archive(csv_path, archive)
    logger.info(
        "[maintenance] email archive synced file=%s total=%s newly_added=%s",
        csv_path,
        len(archive),
        newly_added,
    )
    return {
        "csv_file": str(csv_path),
        "total_emails": len(archive),
        "newly_added": newly_added,
    }


@celery_app.task(name="maintenance.fail_stale_backtest_jobs")
def fail_stale_backtest_jobs_task() -> dict[str, int | list[str]]:
    """Fail stale backtest jobs stuck in running state."""
    return asyncio.run(_fail_stale_backtest_jobs_once())
