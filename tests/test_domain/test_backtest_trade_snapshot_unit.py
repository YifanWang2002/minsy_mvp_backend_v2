"""Unit tests for backtest trade snapshot builders."""

from __future__ import annotations

import base64
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pandas as pd
import pytest

from packages.domain.backtest import service as backtest_service
from packages.domain.backtest import trade_snapshot as snapshot_module


class _FakeLoader:
    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def load(self, **_kwargs) -> pd.DataFrame:
        return self._frame.copy()


class _FakeDb:
    def __init__(self, scalar_results: list[object]) -> None:
        self._scalar_results = list(scalar_results)

    async def scalar(self, _stmt):
        if not self._scalar_results:
            return None
        return self._scalar_results.pop(0)


def _build_frame(rows: int = 40) -> pd.DataFrame:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    index = [start + timedelta(minutes=i) for i in range(rows)]
    base = [100.0 + i * 0.5 for i in range(rows)]
    frame = pd.DataFrame(
        {
            "open": base,
            "high": [value + 1.0 for value in base],
            "low": [value - 1.0 for value in base],
            "close": [value + 0.2 for value in base],
            "volume": [1000.0 + i for i in range(rows)],
        },
        index=index,
    )
    return frame


def _build_result_payload(frame: pd.DataFrame) -> dict[str, object]:
    trades: list[dict[str, object]] = []
    for idx in range(6):
        entry = frame.index[5 + idx * 4]
        exit_ = frame.index[7 + idx * 4]
        trades.append(
            {
                "side": "long",
                "entry_time": entry.isoformat(),
                "exit_time": exit_.isoformat(),
                "entry_price": float(frame.loc[entry, "close"]),
                "exit_price": float(frame.loc[exit_, "close"]),
                "quantity": 1.0,
                "bars_held": 2,
                "exit_reason": "signal_exit",
                "pnl": 1.23,
                "pnl_pct": 1.0,
                "commission": 0.01,
            }
        )

    return {
        "market": "crypto",
        "symbol": "BTCUSD",
        "timeframe": "1m",
        "trades": trades,
        "truncation": {
            "truncated": False,
            "trades_total": len(trades),
            "trades_kept": len(trades),
        },
    }


def _build_strategy_stub() -> SimpleNamespace:
    return SimpleNamespace(
        factors={"ema_fast": object()},
        universe=SimpleNamespace(market="crypto", tickers=("BTCUSD",), timeframe="1m"),
    )


def test_trade_snapshot_selection_modes_and_random_seed(monkeypatch) -> None:
    frame = _build_frame()
    result_payload = _build_result_payload(frame)
    strategy = _build_strategy_stub()

    def _fake_prepare_backtest_frame(data: pd.DataFrame, *, strategy):  # noqa: ANN001
        del strategy
        enriched = data.copy()
        enriched["ema_fast"] = enriched["close"].rolling(3, min_periods=1).mean()
        return enriched

    monkeypatch.setattr(
        snapshot_module,
        "prepare_backtest_frame",
        _fake_prepare_backtest_frame,
    )
    loader = _FakeLoader(frame)

    latest = snapshot_module.build_trade_snapshots_from_result(
        result_payload=result_payload,
        strategy=strategy,
        market="crypto",
        symbol="BTCUSD",
        timeframe="1m",
        selection_mode="latest",
        selection_count=2,
        lookback_bars=2,
        lookforward_bars=2,
        render_images=False,
        random_seed=None,
        loader=loader,
    )
    earliest = snapshot_module.build_trade_snapshots_from_result(
        result_payload=result_payload,
        strategy=strategy,
        market="crypto",
        symbol="BTCUSD",
        timeframe="1m",
        selection_mode="earliest",
        selection_count=2,
        lookback_bars=2,
        lookforward_bars=2,
        render_images=False,
        random_seed=None,
        loader=loader,
    )
    random_one = snapshot_module.build_trade_snapshots_from_result(
        result_payload=result_payload,
        strategy=strategy,
        market="crypto",
        symbol="BTCUSD",
        timeframe="1m",
        selection_mode="random",
        selection_count=3,
        lookback_bars=2,
        lookforward_bars=2,
        render_images=False,
        random_seed=777,
        loader=loader,
    )
    random_two = snapshot_module.build_trade_snapshots_from_result(
        result_payload=result_payload,
        strategy=strategy,
        market="crypto",
        symbol="BTCUSD",
        timeframe="1m",
        selection_mode="random",
        selection_count=3,
        lookback_bars=2,
        lookforward_bars=2,
        render_images=False,
        random_seed=777,
        loader=loader,
    )

    assert [item["trade_index"] for item in latest["snapshots"]] == [4, 5]
    assert [item["trade_index"] for item in earliest["snapshots"]] == [0, 1]
    assert [item["trade_index"] for item in random_one["snapshots"]] == [
        item["trade_index"] for item in random_two["snapshots"]
    ]
    assert random_one["selection"]["random_seed"] == 777


def test_trade_snapshot_window_and_indicator_contract(monkeypatch) -> None:
    frame = _build_frame()
    result_payload = _build_result_payload(frame)
    strategy = _build_strategy_stub()

    def _fake_prepare_backtest_frame(data: pd.DataFrame, *, strategy):  # noqa: ANN001
        del strategy
        enriched = data.copy()
        enriched["ema_fast"] = enriched["close"].rolling(3, min_periods=1).mean()
        return enriched

    monkeypatch.setattr(
        snapshot_module,
        "prepare_backtest_frame",
        _fake_prepare_backtest_frame,
    )

    payload = snapshot_module.build_trade_snapshots_from_result(
        result_payload=result_payload,
        strategy=strategy,
        market="crypto",
        symbol="BTCUSD",
        timeframe="1m",
        selection_mode="earliest",
        selection_count=1,
        lookback_bars=2,
        lookforward_bars=1,
        render_images=False,
        random_seed=None,
        loader=_FakeLoader(frame),
    )

    snapshot = payload["snapshots"][0]
    sliced = snapshot["slice"]
    assert sliced["entry_bar_offset"] == 2
    assert sliced["exit_bar_offset"] == 4
    assert sliced["bar_count"] == 6
    assert len(sliced["candles"]) == 6
    assert "ema_fast" in sliced["indicators"]
    assert len(sliced["indicators"]["ema_fast"]) == 6


def test_trade_snapshot_render_images_returns_base64_and_temp_path(
    monkeypatch,
    tmp_path: Path,
) -> None:
    frame = _build_frame()
    result_payload = _build_result_payload(frame)
    strategy = _build_strategy_stub()

    def _fake_prepare_backtest_frame(data: pd.DataFrame, *, strategy):  # noqa: ANN001
        del strategy
        enriched = data.copy()
        enriched["ema_fast"] = enriched["close"].rolling(3, min_periods=1).mean()
        return enriched

    monkeypatch.setattr(
        snapshot_module,
        "prepare_backtest_frame",
        _fake_prepare_backtest_frame,
    )
    monkeypatch.setattr(
        snapshot_module,
        "_BACKTEST_TRADE_SNAPSHOT_TEMP_ROOT",
        tmp_path,
    )
    job_id = uuid4()

    payload = snapshot_module.build_trade_snapshots_from_result(
        result_payload=result_payload,
        strategy=strategy,
        market="crypto",
        symbol="BTCUSD",
        timeframe="1m",
        selection_mode="latest",
        selection_count=1,
        lookback_bars=6,
        lookforward_bars=3,
        render_images=True,
        save_images_to_temp=True,
        job_id=job_id,
        random_seed=None,
        loader=_FakeLoader(frame),
    )

    snapshot = payload["snapshots"][0]
    image_base64 = snapshot["image_png_base64"]
    assert isinstance(image_base64, str)
    decoded = base64.b64decode(image_base64.encode("ascii"))
    assert len(decoded) > 100
    image_temp_path = snapshot["image_temp_path"]
    assert isinstance(image_temp_path, str)
    saved_file = Path(image_temp_path)
    assert saved_file.is_file()
    assert str(job_id) in image_temp_path


@pytest.mark.asyncio
async def test_service_all_mode_rejects_truncated_results() -> None:
    user_id = uuid4()
    job = SimpleNamespace(
        id=uuid4(),
        strategy_id=uuid4(),
        user_id=user_id,
        status="completed",
        results={
            "trades": [
                {
                    "entry_time": datetime(2025, 1, 1, tzinfo=UTC).isoformat(),
                    "exit_time": datetime(2025, 1, 2, tzinfo=UTC).isoformat(),
                }
            ],
            "truncation": {"trades_total": 12, "trades_kept": 1},
        },
        config={"strategy_version": 1},
        completed_at=datetime(2025, 1, 2, tzinfo=UTC),
    )
    db = _FakeDb([job])

    with pytest.raises(backtest_service.BacktestTradesTruncatedForAllModeError):
        await backtest_service.build_backtest_trade_snapshots(
            db,
            job_id=job.id,
            user_id=user_id,
            selection_mode="all",
            selection_count=None,
            lookback_bars=10,
            lookforward_bars=10,
            render_images=False,
            random_seed=None,
        )
