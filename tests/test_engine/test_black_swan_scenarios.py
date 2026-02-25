from __future__ import annotations

from src.engine.stress.scenario_windows import list_windows, resolve_window_set


def test_list_windows_contains_global_and_market_specific() -> None:
    windows = list_windows(market="us_stocks")
    window_ids = {item.window_id for item in windows}
    assert "covid_crash_2020" in window_ids
    assert "gfc_2008" in window_ids


def test_resolve_custom_window_set() -> None:
    custom = resolve_window_set(
        market="crypto",
        window_set="custom",
        custom_windows=[
            {
                "window_id": "x1",
                "label": "Custom X1",
                "start": "2024-01-01T00:00:00Z",
                "end": "2024-01-31T00:00:00Z",
            }
        ],
    )
    assert len(custom) == 1
    assert custom[0].window_id == "x1"
