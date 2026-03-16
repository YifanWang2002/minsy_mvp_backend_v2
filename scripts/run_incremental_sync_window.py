#!/usr/bin/env python3
"""Run local incremental sync with optional explicit time window."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime

from packages.domain.market_data.incremental.local_sync_service import (
    run_local_incremental_sync,
)
from packages.domain.market_data.incremental.provider_router import (
    normalize_incremental_market,
)


def _parse_dt(value: str | None) -> datetime | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    normalized = text.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--window-start",
        default="",
        help="ISO timestamp, e.g. 2025-11-01T00:00:00Z",
    )
    parser.add_argument(
        "--window-end",
        default="",
        help="ISO timestamp, e.g. 2026-03-12T23:59:00Z",
    )
    parser.add_argument(
        "--ignore-session-gate",
        action="store_true",
        help="Ignore market session gate and fetch historical windows directly.",
    )
    parser.add_argument(
        "--markets",
        default="",
        help="Comma-separated market list (crypto,us_stocks,forex,futures). Empty means all.",
    )
    args = parser.parse_args()
    include_markets: set[str] | None = None
    if str(args.markets).strip():
        include_markets = {
            normalize_incremental_market(item)
            for item in str(args.markets).split(",")
            if item.strip()
        }

    result = asyncio.run(
        run_local_incremental_sync(
            window_start=_parse_dt(args.window_start),
            window_end=_parse_dt(args.window_end),
            respect_session_gate=not bool(args.ignore_session_gate),
            include_markets=include_markets,
        )
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
