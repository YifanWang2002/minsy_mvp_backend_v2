from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from packages.domain.market_data.incremental import local_sync_service


def test_split_frame_by_month_partitions_rows_and_keeps_order() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": [
                datetime(2025, 11, 30, 23, 59, tzinfo=UTC),
                datetime(2025, 12, 1, 0, 0, tzinfo=UTC),
                datetime(2025, 12, 1, 0, 1, tzinfo=UTC),
            ],
            "open": [1.0, 2.0, 3.0],
            "high": [1.1, 2.1, 3.1],
            "low": [0.9, 1.9, 2.9],
            "close": [1.0, 2.0, 3.0],
            "volume": [10.0, 20.0, 30.0],
        }
    )

    chunks = local_sync_service._split_frame_by_month(frame)
    assert [month for month, _ in chunks] == ["2025-11", "2025-12"]
    assert len(chunks[0][1]) == 1
    assert len(chunks[1][1]) == 2
    assert chunks[1][1]["timestamp"].iloc[0] == frame["timestamp"].iloc[1]


def test_apply_failed_backfill_start_uses_previous_failure_and_lower_bound() -> None:
    failure_key = "crypto|BTCUSD|eth"
    active_failures = {
        failure_key: {
            "start": "2026-03-10T00:00:00+00:00",
        }
    }
    start = datetime(2026, 3, 12, 0, 0, tzinfo=UTC)

    resumed = local_sync_service._apply_failed_backfill_start(
        active_failures=active_failures,
        failure_key=failure_key,
        start=start,
        lower_bound=None,
    )
    assert resumed == datetime(2026, 3, 10, 0, 0, tzinfo=UTC)

    bounded = local_sync_service._apply_failed_backfill_start(
        active_failures=active_failures,
        failure_key=failure_key,
        start=start,
        lower_bound=datetime(2026, 3, 11, 0, 0, tzinfo=UTC),
    )
    assert bounded == datetime(2026, 3, 11, 0, 0, tzinfo=UTC)


def test_record_and_clear_symbol_failure_updates_state_and_events(monkeypatch) -> None:
    captured_states: list[dict[str, dict[str, object]]] = []
    captured_events: list[dict[str, object]] = []

    def _capture_state(state: dict[str, dict[str, object]]) -> None:
        copied = {str(k): dict(v) for k, v in state.items()}
        captured_states.append(copied)

    def _capture_event(event: dict[str, object]) -> None:
        captured_events.append(dict(event))

    monkeypatch.setattr(local_sync_service, "_save_active_failures", _capture_state)
    monkeypatch.setattr(local_sync_service, "_append_failure_event", _capture_event)

    failure_key = "forex|EURUSD|eth"
    state: dict[str, dict[str, object]] = {}
    local_sync_service._record_symbol_failure(
        active_failures=state,
        failure_key=failure_key,
        market="forex",
        symbol="EURUSD",
        session="eth",
        provider="ibkr",
        start=datetime(2026, 3, 13, 0, 0, tzinfo=UTC),
        end=datetime(2026, 3, 13, 1, 0, tzinfo=UTC),
        exc=RuntimeError("network down"),
    )
    assert failure_key in state
    assert captured_states, "failure state should be persisted"
    assert captured_events and captured_events[-1]["event"] == "fetch_failed"

    local_sync_service._clear_symbol_failure(
        active_failures=state,
        failure_key=failure_key,
        market="forex",
        symbol="EURUSD",
        session="eth",
    )
    assert failure_key not in state
    assert captured_events[-1]["event"] == "fetch_recovered"


@pytest.mark.asyncio
async def test_fetch_1m_with_retries_retries_twice_then_success(monkeypatch) -> None:
    calls = {"count": 0}

    async def _flaky_fetch(**_kwargs):
        calls["count"] += 1
        if calls["count"] < 3:
            raise OSError("temporary network error")
        return []

    async def _no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(local_sync_service, "_fetch_alpaca_1m", _flaky_fetch)
    monkeypatch.setattr(local_sync_service.asyncio, "sleep", _no_sleep)

    rows = await local_sync_service._fetch_1m_with_retries(
        provider="alpaca",
        symbol="BTCUSD",
        market="crypto",
        start=datetime(2026, 3, 13, 0, 0, tzinfo=UTC),
        end=datetime(2026, 3, 13, 0, 10, tzinfo=UTC),
        retries=2,
    )
    assert rows == []
    assert calls["count"] == 3


@pytest.mark.asyncio
async def test_fetch_1m_with_retries_stops_after_configured_retries(monkeypatch) -> None:
    calls = {"count": 0}

    async def _always_fail(**_kwargs):
        calls["count"] += 1
        raise OSError("still down")

    async def _no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(local_sync_service, "_fetch_ibkr_1m", _always_fail)
    monkeypatch.setattr(local_sync_service.asyncio, "sleep", _no_sleep)

    with pytest.raises(OSError, match="still down"):
        await local_sync_service._fetch_1m_with_retries(
            provider="ibkr",
            symbol="EURUSD",
            market="forex",
            start=datetime(2026, 3, 13, 0, 0, tzinfo=UTC),
            end=datetime(2026, 3, 13, 0, 10, tzinfo=UTC),
            retries=2,
        )
    assert calls["count"] == 3


def test_replay_staged_incremental_to_local_scans_expected_layout(
    monkeypatch,
    tmp_path: Path,
) -> None:
    stage_file = (
        tmp_path
        / "2026-03-16"
        / "20260316T063138Z-b87c1171"
        / "crypto"
        / "BTCUSD"
        / "eth"
        / "1m"
        / "2026-03.parquet"
    )
    stage_file.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "timestamp": [datetime(2026, 3, 16, 6, 0, tzinfo=UTC)],
            "open": [1.0],
            "high": [1.1],
            "low": [0.9],
            "close": [1.0],
            "volume": [10.0],
        }
    )
    frame.to_parquet(stage_file, index=False)

    class _FakeLoader:
        pass

    calls: list[tuple[str, str, str, str, int]] = []

    def _fake_append_ohlcv_rows(**kwargs):
        rows = kwargs["rows"]
        calls.append(
            (
                kwargs["market"],
                kwargs["symbol"],
                kwargs["timeframe"],
                kwargs["session"],
                len(rows),
            )
        )

        class _Result:
            rows_written = len(rows)
            files_touched = 1

        return _Result()

    monkeypatch.setattr(local_sync_service, "DataLoader", _FakeLoader)
    monkeypatch.setattr(local_sync_service, "append_ohlcv_rows", _fake_append_ohlcv_rows)

    summary = local_sync_service.replay_staged_incremental_to_local(stage_root=tmp_path)
    assert summary.files_seen == 1
    assert summary.files_applied == 1
    assert summary.rows_written == 1
    assert summary.local_files_touched == 1
    assert summary.symbols_touched == 1
    assert calls == [("crypto", "BTCUSD", "1m", "eth", 1)]
