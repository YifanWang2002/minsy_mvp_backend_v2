from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from src.engine.data import DataLoader
from src.engine.data.local_coverage import (
    LocalCoverageInputError,
    detect_missing_ranges,
)


def _write_rows(path: Path, timestamps: list[datetime]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [100.0 + idx for idx in range(len(timestamps))],
            "high": [101.0 + idx for idx in range(len(timestamps))],
            "low": [99.0 + idx for idx in range(len(timestamps))],
            "close": [100.5 + idx for idx in range(len(timestamps))],
            "volume": [10.0 for _ in timestamps],
        }
    )
    frame.to_parquet(path, index=False)


def test_detect_missing_ranges_single_gap(tmp_path: Path) -> None:
    loader = DataLoader(data_dir=tmp_path)
    timestamps = [
        datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
        datetime(2024, 1, 1, 0, 1, tzinfo=UTC),
        datetime(2024, 1, 1, 0, 3, tzinfo=UTC),
    ]
    _write_rows(tmp_path / "crypto" / "BTCUSD_1min_eth_2024.parquet", timestamps)

    report = detect_missing_ranges(
        loader=loader,
        market="crypto",
        symbol="BTCUSD",
        timeframe="1m",
        start=datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
        end=datetime(2024, 1, 1, 0, 3, tzinfo=UTC),
    )

    assert report.expected_bars == 4
    assert report.missing_bars == 1
    assert report.present_bars == 3
    assert pytest.approx(report.local_coverage_pct, rel=1e-6) == 75.0
    assert len(report.missing_ranges) == 1
    assert report.missing_ranges[0].start == datetime(2024, 1, 1, 0, 2, tzinfo=UTC)
    assert report.missing_ranges[0].end == datetime(2024, 1, 1, 0, 2, tzinfo=UTC)


def test_detect_missing_ranges_when_local_absent_marks_full_window_missing(tmp_path: Path) -> None:
    loader = DataLoader(data_dir=tmp_path)
    (tmp_path / "crypto").mkdir(parents=True, exist_ok=True)

    report = detect_missing_ranges(
        loader=loader,
        market="crypto",
        symbol="ETHUSD",
        timeframe="1m",
        start=datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
        end=datetime(2024, 1, 1, 0, 2, tzinfo=UTC),
    )

    assert report.expected_bars == 3
    assert report.present_bars == 0
    assert report.missing_bars == 3
    assert len(report.missing_ranges) == 1
    assert report.local_coverage_pct == 0.0


def test_detect_missing_ranges_rejects_unsupported_timeframe(tmp_path: Path) -> None:
    loader = DataLoader(data_dir=tmp_path)
    with pytest.raises(LocalCoverageInputError):
        detect_missing_ranges(
            loader=loader,
            market="crypto",
            symbol="BTCUSD",
            timeframe="10m",
            start=datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
            end=datetime(2024, 1, 1, 0, 5, tzinfo=UTC),
        )
