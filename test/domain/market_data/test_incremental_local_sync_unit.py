from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from packages.domain.market_data.incremental.local_sync_service import (
    _split_frame_by_month,
)


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

    chunks = _split_frame_by_month(frame)
    assert [month for month, _ in chunks] == ["2025-11", "2025-12"]
    assert len(chunks[0][1]) == 1
    assert len(chunks[1][1]) == 2
    assert chunks[1][1]["timestamp"].iloc[0] == frame["timestamp"].iloc[1]
