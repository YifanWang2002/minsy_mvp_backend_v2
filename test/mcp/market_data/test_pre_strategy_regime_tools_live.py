from __future__ import annotations

import json

import pytest

from apps.mcp.domains.market_data import tools as market_tools


@pytest.mark.external
async def test_live_pre_strategy_regime_snapshot_and_chart() -> None:
    snapshot_result = await market_tools.pre_strategy_get_regime_snapshot(
        market="crypto",
        symbol="BTCUSD",
        opportunity_frequency_bucket="daily",
        holding_period_bucket="intraday",
        lookback_bars=220,
    )
    snapshot_payload = json.loads(snapshot_result)

    assert snapshot_payload["ok"] is True, snapshot_result
    snapshot_id = str(snapshot_payload.get("snapshot_id") or "").strip()
    assert snapshot_id

    chart_result = market_tools.pre_strategy_render_candlestick(
        snapshot_id=snapshot_id,
        timeframe="primary",
        bars=120,
    )
    assert isinstance(chart_result, list), chart_result
    assert len(chart_result) >= 1
