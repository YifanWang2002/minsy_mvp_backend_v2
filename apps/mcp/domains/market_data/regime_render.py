"""Candlestick rendering helpers for pre-strategy regime tooling."""

from __future__ import annotations

from io import BytesIO
from typing import Any

import mplfinance as mpf
import pandas as pd
from mcp.server.fastmcp.utilities.types import Image


def render_candlestick_image(
    *,
    candles: pd.DataFrame,
    title: str,
    max_bars: int = 240,
) -> Image:
    """Render one candlestick image and return MCP Image helper."""

    if candles.empty:
        raise ValueError("candles cannot be empty")

    frame = candles.copy().sort_index()
    if len(frame) > max_bars:
        frame = frame.tail(max_bars)
    frame = frame[["open", "high", "low", "close", "volume"]]
    frame.index = pd.to_datetime(frame.index, utc=True)

    fig, _ = mpf.plot(
        frame,
        type="candle",
        style="charles",
        volume=True,
        title=title,
        returnfig=True,
        warn_too_much_data=max(max_bars, 200),
        figratio=(16, 9),
        figscale=1.1,
    )
    try:
        buffer = BytesIO()
        fig.savefig(
            buffer,
            format="png",
            dpi=120,
            bbox_inches="tight",
            facecolor="white",
        )
        return Image(data=buffer.getvalue(), format="png")
    finally:
        # mplfinance uses matplotlib under the hood; close explicitly.
        fig.clf()


def build_chart_alt_text(
    *,
    timeframe: str,
    summary: str,
    family_scores: dict[str, Any],
) -> str:
    trend = float(family_scores.get("trend_continuation", 0.0) or 0.0)
    reversion = float(family_scores.get("mean_reversion", 0.0) or 0.0)
    vol = float(family_scores.get("volatility_regime", 0.0) or 0.0)
    return (
        f"Chart timeframe={timeframe}. "
        f"Scores trend={trend:.2f}, reversion={reversion:.2f}, volatility={vol:.2f}. "
        f"{summary}"
    )

