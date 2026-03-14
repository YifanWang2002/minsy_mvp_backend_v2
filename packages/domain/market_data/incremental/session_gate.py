"""Session gating helpers for incremental fetch windows."""

from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from packages.domain.market_data.incremental.provider_router import (
    normalize_incremental_market,
)

_NY_TZ = ZoneInfo("America/New_York")


def market_is_open_for_incremental(
    *,
    market: str,
    now: datetime | None = None,
) -> bool:
    """Return whether incremental fetching should run for this market now."""

    market_key = normalize_incremental_market(market)
    now_utc = (now or datetime.now(UTC)).astimezone(UTC)
    local_ny = now_utc.astimezone(_NY_TZ)
    weekday = local_ny.weekday()  # Monday=0, Sunday=6

    if market_key == "crypto":
        return True

    if market_key == "us_stocks":
        # Weekend hard gate. Fine-grained RTH window is enforced by provider data itself.
        return weekday < 5

    if market_key in {"forex", "futures"}:
        # Weekend close window in New York time:
        # - closed from Friday 17:00 until Sunday 17:00.
        if weekday == 4 and local_ny.hour >= 17:
            return False
        if weekday == 5:
            return False
        if weekday == 6 and local_ny.hour < 17:
            return False
        return True

    return False
