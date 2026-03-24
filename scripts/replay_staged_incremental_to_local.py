#!/usr/bin/env python3
"""Replay staged incremental parquet files into local data parquet store."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    from packages.domain.market_data.incremental.local_sync_service import (
        replay_staged_incremental_to_local,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage-root",
        default="runtime/incremental",
        help="Root directory containing staged incremental runs.",
    )
    args = parser.parse_args()

    result = replay_staged_incremental_to_local(stage_root=Path(args.stage_root))
    print(json.dumps(result.to_dict(), ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
