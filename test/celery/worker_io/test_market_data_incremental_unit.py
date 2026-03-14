from __future__ import annotations

import pytest
from pydantic import ValidationError

from apps.worker.io.tasks import market_data as market_data_tasks
from packages.domain.market_data.incremental.local_sync_service import (
    LocalIncrementalSyncResult,
)
from packages.infra.queue import celery_app as celery_module
from packages.shared_settings.schema.settings import Settings


def test_incremental_execution_mode_validation_accepts_known_values() -> None:
    for mode in ("disabled", "local_collector", "remote_importer"):
        value = Settings(
            OPENAI_API_KEY="test-openai-key",
            SECRET_KEY="test-secret-key",
            MARKET_DATA_INCREMENTAL_EXECUTION_MODE=mode,
        )
        assert value.market_data_incremental_execution_mode == mode


def test_incremental_execution_mode_validation_rejects_unknown_values() -> None:
    with pytest.raises(ValidationError, match="MARKET_DATA_INCREMENTAL_EXECUTION_MODE"):
        Settings(
            OPENAI_API_KEY="test-openai-key",
            SECRET_KEY="test-secret-key",
            MARKET_DATA_INCREMENTAL_EXECUTION_MODE="cloud_collector",
        )


def test_beat_registers_incremental_sync_only_for_local_collector(monkeypatch) -> None:
    monkeypatch.setattr(
        celery_module.settings,
        "market_data_incremental_sync_enabled",
        True,
    )
    monkeypatch.setattr(
        celery_module.settings,
        "market_data_incremental_execution_mode",
        "local_collector",
    )
    monkeypatch.setattr(
        celery_module.settings,
        "market_data_incremental_sync_cron_hour",
        4,
    )
    monkeypatch.setattr(
        celery_module.settings,
        "market_data_incremental_sync_cron_minute",
        30,
    )

    schedule = celery_module._build_beat_schedule(enabled=True)
    assert "market_data.run_incremental_sync" in schedule
    entry = schedule["market_data.run_incremental_sync"]
    assert entry["task"] == "market_data.run_incremental_sync"

    monkeypatch.setattr(
        celery_module.settings,
        "market_data_incremental_execution_mode",
        "remote_importer",
    )
    schedule_remote = celery_module._build_beat_schedule(enabled=True)
    assert "market_data.run_incremental_sync" not in schedule_remote


def test_worker_runtime_gate_skips_incremental_sync_when_not_local(monkeypatch) -> None:
    async def _should_not_run_local_sync() -> object:
        raise AssertionError("run_local_incremental_sync should not be called")

    monkeypatch.setattr(
        market_data_tasks.settings,
        "market_data_incremental_execution_mode",
        "remote_importer",
    )
    monkeypatch.setattr(
        market_data_tasks.settings,
        "market_data_incremental_sync_enabled",
        True,
    )
    monkeypatch.setattr(
        market_data_tasks,
        "run_local_incremental_sync",
        _should_not_run_local_sync,
    )

    result = market_data_tasks.run_incremental_sync_task()
    assert result["status"] == "skipped_not_local_collector"
    assert result["execution_mode"] == "remote_importer"


def test_worker_runtime_gate_returns_disabled_for_local_mode_when_switch_off(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        market_data_tasks.settings,
        "market_data_incremental_execution_mode",
        "local_collector",
    )
    monkeypatch.setattr(
        market_data_tasks.settings,
        "market_data_incremental_sync_enabled",
        False,
    )

    result = market_data_tasks.run_incremental_sync_task()
    assert result == {
        "status": "disabled",
        "execution_mode": "local_collector",
    }


def test_incremental_sync_result_to_dict_contract() -> None:
    result = LocalIncrementalSyncResult(
        status="ok",
        run_id="20260315T043000Z-abcd1234",
        symbols_seen=10,
        symbols_synced=6,
        files_uploaded=12,
        rows_uploaded=24_000,
        skipped_closed=2,
        skipped_uptodate=2,
        import_job_id="f96e0bad-9bb5-4e31-afd5-c6b6ab4f1418",
    )

    payload = result.to_dict()
    assert payload["status"] == "ok"
    assert payload["symbols_seen"] == 10
    assert payload["import_job_id"] is not None
    assert payload["run_id"] == "20260315T043000Z-abcd1234"
