"""Predefined stress windows for crisis replay tests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class ScenarioWindow:
    """One stress scenario time window."""

    window_id: str
    label: str
    market: str
    start: datetime
    end: datetime


def list_windows(*, market: str) -> list[ScenarioWindow]:
    """List predefined windows for a market (plus global defaults)."""

    key = _normalize_market(market)
    candidates = list(_GLOBAL_WINDOWS)
    candidates.extend(_MARKET_WINDOWS.get(key, ()))
    return sorted(candidates, key=lambda item: item.start)


def resolve_window_set(
    *,
    market: str,
    window_set: str,
    custom_windows: list[dict[str, Any]] | None = None,
) -> list[ScenarioWindow]:
    """Resolve selected window set into concrete windows."""

    normalized = str(window_set).strip().lower() or "default"
    if normalized == "default":
        return list_windows(market=market)

    if normalized != "custom":
        raise ValueError("window_set must be one of: default, custom")

    output: list[ScenarioWindow] = []
    for index, raw in enumerate(custom_windows or []):
        if not isinstance(raw, dict):
            continue
        window_id = str(raw.get("window_id") or f"custom_{index + 1}").strip() or f"custom_{index + 1}"
        label = str(raw.get("label") or window_id).strip() or window_id
        start = _parse_datetime(raw.get("start"), field_name="start")
        end = _parse_datetime(raw.get("end"), field_name="end")
        if end < start:
            raise ValueError(f"custom window '{window_id}' has end before start")
        output.append(
            ScenarioWindow(
                window_id=window_id,
                label=label,
                market=_normalize_market(market),
                start=start,
                end=end,
            )
        )

    if not output:
        raise ValueError("custom window_set requires at least one valid window")
    return output


def serialize_windows(windows: list[ScenarioWindow]) -> list[dict[str, Any]]:
    """Serialize windows for MCP payloads."""

    return [
        {
            "window_id": item.window_id,
            "label": item.label,
            "market": item.market,
            "start": item.start.isoformat(),
            "end": item.end.isoformat(),
        }
        for item in windows
    ]


def _normalize_market(value: str) -> str:
    raw = str(value).strip().lower()
    mapping = {
        "stock": "us_stocks",
        "stocks": "us_stocks",
        "us_stocks": "us_stocks",
        "crypto": "crypto",
        "forex": "forex",
        "futures": "futures",
    }
    return mapping.get(raw, raw or "us_stocks")


def _parse_datetime(value: Any, *, field_name: str) -> datetime:
    text = str(value).strip()
    if not text:
        raise ValueError(f"custom window {field_name} cannot be empty")
    normalized = text.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


_GLOBAL_WINDOWS: tuple[ScenarioWindow, ...] = (
    ScenarioWindow(
        window_id="covid_crash_2020",
        label="COVID Crash 2020",
        market="global",
        start=datetime(2020, 2, 20, tzinfo=UTC),
        end=datetime(2020, 4, 30, tzinfo=UTC),
    ),
    ScenarioWindow(
        window_id="inflation_shock_2022",
        label="Inflation Shock 2022",
        market="global",
        start=datetime(2022, 1, 1, tzinfo=UTC),
        end=datetime(2022, 12, 31, tzinfo=UTC),
    ),
)

_MARKET_WINDOWS: dict[str, tuple[ScenarioWindow, ...]] = {
    "us_stocks": (
        ScenarioWindow(
            window_id="gfc_2008",
            label="Global Financial Crisis",
            market="us_stocks",
            start=datetime(2008, 9, 1, tzinfo=UTC),
            end=datetime(2009, 6, 30, tzinfo=UTC),
        ),
        ScenarioWindow(
            window_id="flash_crash_2010",
            label="Flash Crash 2010",
            market="us_stocks",
            start=datetime(2010, 5, 1, tzinfo=UTC),
            end=datetime(2010, 5, 31, tzinfo=UTC),
        ),
    ),
    "crypto": (
        ScenarioWindow(
            window_id="crypto_winter_2018",
            label="Crypto Winter 2018",
            market="crypto",
            start=datetime(2018, 1, 1, tzinfo=UTC),
            end=datetime(2018, 12, 31, tzinfo=UTC),
        ),
        ScenarioWindow(
            window_id="luna_3ac_2022",
            label="LUNA/3AC Deleveraging",
            market="crypto",
            start=datetime(2022, 5, 1, tzinfo=UTC),
            end=datetime(2022, 8, 31, tzinfo=UTC),
        ),
    ),
}
