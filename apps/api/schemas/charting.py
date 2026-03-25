"""Schemas for TradingView Charting Library datafeed endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChartingExchangeResponse(BaseModel):
    """One exchange filter entry for symbol search."""

    value: str
    name: str
    desc: str


class ChartingSymbolTypeResponse(BaseModel):
    """One symbol-type filter entry for symbol search."""

    name: str
    value: str


class ChartingDatafeedConfigResponse(BaseModel):
    """TradingView datafeed configuration payload."""

    supports_search: bool = True
    supports_group_request: bool = False
    supported_resolutions: list[str] = Field(default_factory=list)
    supports_marks: bool = False
    supports_timescale_marks: bool = False
    supports_time: bool = True
    exchanges: list[ChartingExchangeResponse] = Field(default_factory=list)
    symbols_types: list[ChartingSymbolTypeResponse] = Field(default_factory=list)


class ChartingSearchSymbolResponse(BaseModel):
    """One symbol search match."""

    symbol: str
    full_name: str | None = None
    description: str
    exchange: str
    ticker: str | None = None
    type: str


class ChartingResolveSymbolResponse(BaseModel):
    """Resolved TradingView symbol metadata."""

    name: str
    ticker: str
    full_name: str | None = None
    description: str
    long_description: str | None = None
    type: str
    session: str
    session_display: str | None = None
    exchange: str
    listed_exchange: str
    timezone: str
    format: str = "price"
    pricescale: int
    minmov: int = 1
    minmove2: int | None = None
    has_intraday: bool = True
    supported_resolutions: list[str] = Field(default_factory=list)
    intraday_multipliers: list[str] = Field(default_factory=list)
    has_daily: bool = True
    daily_multipliers: list[str] = Field(default_factory=lambda: ["1"])
    has_weekly_and_monthly: bool = False
    weekly_multipliers: list[str] = Field(default_factory=list)
    monthly_multipliers: list[str] = Field(default_factory=list)
    has_empty_bars: bool = False
    visible_plots_set: str = "ohlcv"
    volume_precision: int = 2
    data_status: str = "streaming"
    delay: int = 0
    minsy_market: str


class ChartingBarResponse(BaseModel):
    """One TradingView bar payload."""

    time: int
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


class ChartingHistoryMetadataResponse(BaseModel):
    """TradingView history metadata."""

    noData: bool = False
    nextTime: int | None = None


class ChartingHistoryResponse(BaseModel):
    """TradingView history result payload."""

    bars: list[ChartingBarResponse] = Field(default_factory=list)
    meta: ChartingHistoryMetadataResponse = Field(
        default_factory=ChartingHistoryMetadataResponse
    )
