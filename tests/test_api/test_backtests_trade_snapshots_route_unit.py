"""Unit tests for backtest trade-snapshot route contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest
from pydantic import ValidationError

from apps.api.routes import backtests as backtests_routes
from apps.api.schemas.requests import BacktestTradeSnapshotRequest
from packages.domain.backtest.service import (
    BacktestJobNotFoundError,
    BacktestJobNotReadyError,
    BacktestTradeSnapshotInputError,
    BacktestTradesTruncatedForAllModeError,
)


def test_trade_snapshot_request_requires_null_count_for_all_mode() -> None:
    with pytest.raises(ValidationError):
        BacktestTradeSnapshotRequest(
            selection_mode="all",
            selection_count=1,
        )


def test_trade_snapshot_request_requires_count_for_non_all_modes() -> None:
    with pytest.raises(ValidationError):
        BacktestTradeSnapshotRequest(
            selection_mode="latest",
            selection_count=None,
        )


def test_trade_snapshot_request_requires_render_images_for_temp_save() -> None:
    with pytest.raises(ValidationError):
        BacktestTradeSnapshotRequest(
            selection_mode="latest",
            selection_count=1,
            render_images=False,
            save_images_to_temp=True,
        )


@pytest.mark.asyncio
async def test_trade_snapshot_route_success(monkeypatch) -> None:
    job_id = uuid4()
    strategy_id = uuid4()
    user = SimpleNamespace(id=uuid4())
    payload = BacktestTradeSnapshotRequest(
        selection_mode="latest",
        selection_count=3,
        lookback_bars=20,
        lookforward_bars=15,
        render_images=True,
        save_images_to_temp=True,
    )
    captured: dict[str, object] = {}

    async def _fake_service(*args, **kwargs):  # noqa: ANN002, ANN003
        del args
        captured.update(kwargs)
        return {
            "job_id": str(job_id),
            "strategy_id": str(strategy_id),
            "status": "done",
            "strategy_payload_source": "strategy_revision",
            "selection": {"mode": "latest", "selected_count": 3},
            "window": {"lookback_bars": 20, "lookforward_bars": 15},
            "snapshots": [],
            "warnings": [],
            "completed_at": datetime(2026, 3, 1, tzinfo=UTC).isoformat(),
        }

    monkeypatch.setattr(
        backtests_routes,
        "build_backtest_trade_snapshots",
        _fake_service,
    )

    result = await backtests_routes.get_backtest_trade_snapshots(
        job_id=job_id,
        payload=payload,
        user=user,
        db=object(),
    )

    assert result["job_id"] == str(job_id)
    assert result["strategy_id"] == str(strategy_id)
    assert result["status"] == "done"
    assert result["selection"]["mode"] == "latest"
    assert captured["save_images_to_temp"] is True


@pytest.mark.asyncio
async def test_trade_snapshot_route_not_found_maps_404(monkeypatch) -> None:
    async def _fake_service(*args, **kwargs):  # noqa: ANN002, ANN003
        del args, kwargs
        raise BacktestJobNotFoundError("Backtest job not found: missing")

    monkeypatch.setattr(
        backtests_routes,
        "build_backtest_trade_snapshots",
        _fake_service,
    )

    with pytest.raises(backtests_routes.HTTPException) as exc_info:
        await backtests_routes.get_backtest_trade_snapshots(
            job_id=uuid4(),
            payload=BacktestTradeSnapshotRequest(
                selection_mode="latest",
                selection_count=1,
            ),
            user=SimpleNamespace(id=uuid4()),
            db=object(),
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail["code"] == "BACKTEST_JOB_NOT_FOUND"


@pytest.mark.asyncio
async def test_trade_snapshot_route_not_ready_maps_409(monkeypatch) -> None:
    async def _fake_service(*args, **kwargs):  # noqa: ANN002, ANN003
        del args, kwargs
        raise BacktestJobNotReadyError(status="running")

    monkeypatch.setattr(
        backtests_routes,
        "build_backtest_trade_snapshots",
        _fake_service,
    )

    with pytest.raises(backtests_routes.HTTPException) as exc_info:
        await backtests_routes.get_backtest_trade_snapshots(
            job_id=uuid4(),
            payload=BacktestTradeSnapshotRequest(
                selection_mode="latest",
                selection_count=1,
            ),
            user=SimpleNamespace(id=uuid4()),
            db=object(),
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail["code"] == "BACKTEST_JOB_NOT_READY"
    assert exc_info.value.detail["status"] == "running"


@pytest.mark.asyncio
async def test_trade_snapshot_route_truncation_maps_409(monkeypatch) -> None:
    async def _fake_service(*args, **kwargs):  # noqa: ANN002, ANN003
        del args, kwargs
        raise BacktestTradesTruncatedForAllModeError(trades_total=100, trades_kept=20)

    monkeypatch.setattr(
        backtests_routes,
        "build_backtest_trade_snapshots",
        _fake_service,
    )

    with pytest.raises(backtests_routes.HTTPException) as exc_info:
        await backtests_routes.get_backtest_trade_snapshots(
            job_id=uuid4(),
            payload=BacktestTradeSnapshotRequest(
                selection_mode="all",
                selection_count=None,
            ),
            user=SimpleNamespace(id=uuid4()),
            db=object(),
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail["code"] == "BACKTEST_TRADES_TRUNCATED_FOR_ALL_MODE"
    assert exc_info.value.detail["trades_total"] == 100
    assert exc_info.value.detail["trades_kept"] == 20


@pytest.mark.asyncio
async def test_trade_snapshot_route_invalid_input_maps_422(monkeypatch) -> None:
    async def _fake_service(*args, **kwargs):  # noqa: ANN002, ANN003
        del args, kwargs
        raise BacktestTradeSnapshotInputError("invalid payload")

    monkeypatch.setattr(
        backtests_routes,
        "build_backtest_trade_snapshots",
        _fake_service,
    )

    with pytest.raises(backtests_routes.HTTPException) as exc_info:
        await backtests_routes.get_backtest_trade_snapshots(
            job_id=uuid4(),
            payload=BacktestTradeSnapshotRequest(
                selection_mode="latest",
                selection_count=1,
            ),
            user=SimpleNamespace(id=uuid4()),
            db=object(),
        )

    assert exc_info.value.status_code == 422
    assert exc_info.value.detail["code"] == "BACKTEST_TRADE_SNAPSHOT_INVALID_INPUT"
