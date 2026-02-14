"""
MCP Server: yfinance market data query tool.

Exposes tools via Model Context Protocol (MCP) to query stock/ETF/index
data from Yahoo Finance using the yfinance library.

Usage:
    python -m agents.mcp.yfinance_server          # stdio (default)
    python -m agents.mcp.yfinance_server --sse     # SSE on port 8100
"""

from __future__ import annotations

import json
import sys
from urllib import parse, request

import yfinance as yf
from mcp.server.fastmcp import FastMCP

# ── server instance ──────────────────────────────────────────────────
mcp = FastMCP("YFinance Market Data")


# ── helpers ──────────────────────────────────────────────────────────
VALID_INTERVALS = [
    "1m", "2m", "5m", "15m", "30m", "60m", "90m",
    "1h", "1d", "5d", "1wk", "1mo", "3mo",
]

VALID_PERIODS = [
    "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max",
]


def _ticker_info_summary(ticker: yf.Ticker) -> dict:
    """Return a concise summary from ticker.info."""
    info = ticker.info or {}
    keys = [
        "shortName", "longName", "symbol", "exchange", "market",
        "quoteType", "currency", "marketCap", "enterpriseValue",
        "trailingPE", "forwardPE", "dividendYield", "beta",
        "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
        "regularMarketPrice", "regularMarketOpen",
        "regularMarketDayHigh", "regularMarketDayLow",
        "regularMarketVolume", "averageVolume",
        "sector", "industry", "country",
    ]
    return {k: info[k] for k in keys if k in info}


# ── tools ────────────────────────────────────────────────────────────

@mcp.tool()
def get_quote(symbol: str) -> str:
    """
    Get the real-time (latest) quote / summary info for a symbol.

    Args:
        symbol: Ticker symbol, e.g. "AAPL", "0700.HK", "USDJPY=X", "BTC-USD"

    Returns:
        JSON string with key quote fields.
    """
    ticker = yf.Ticker(symbol)
    summary = _ticker_info_summary(ticker)
    if not summary:
        return json.dumps({"error": f"No data found for symbol '{symbol}'"})
    return json.dumps(summary, ensure_ascii=False, default=str)


@mcp.tool()
def get_history(
    symbol: str,
    period: str = "1mo",
    interval: str = "1d",
    start: str | None = None,
    end: str | None = None,
) -> str:
    """
    Get historical OHLCV price data for a symbol.

    Args:
        symbol:   Ticker symbol, e.g. "AAPL", "600519.SS", "^GSPC"
        period:   Data period (1d|5d|1mo|3mo|6mo|1y|2y|5y|10y|ytd|max).
                  Ignored when start/end are provided.
        interval: Bar interval (1m|2m|5m|15m|30m|60m|90m|1h|1d|5d|1wk|1mo|3mo).
        start:    Start date string, e.g. "2024-01-01". Optional.
        end:      End date string, e.g. "2024-12-31". Optional.

    Returns:
        JSON string of OHLCV rows (date, open, high, low, close, volume).
    """
    if interval not in VALID_INTERVALS:
        return json.dumps({"error": f"Invalid interval '{interval}'. Choose from {VALID_INTERVALS}"})
    if not start and period not in VALID_PERIODS:
        return json.dumps({"error": f"Invalid period '{period}'. Choose from {VALID_PERIODS}"})

    ticker = yf.Ticker(symbol)

    kwargs: dict = {"interval": interval}
    if start:
        kwargs["start"] = start
        if end:
            kwargs["end"] = end
    else:
        kwargs["period"] = period

    df = ticker.history(**kwargs)
    if df.empty:
        return json.dumps({"error": f"No history data for '{symbol}' with given parameters."})

    # Keep it concise – limit to latest 60 rows for large results
    if len(df) > 60:
        df = df.tail(60)
        truncated = True
    else:
        truncated = False

    df.index = df.index.strftime("%Y-%m-%d %H:%M")
    records = df[["Open", "High", "Low", "Close", "Volume"]].reset_index()
    records.columns = ["datetime", "open", "high", "low", "close", "volume"]

    result = {
        "symbol": symbol,
        "interval": interval,
        "rows": len(records),
        "truncated": truncated,
        "data": records.to_dict(orient="records"),
    }
    return json.dumps(result, ensure_ascii=False, default=str)


@mcp.tool()
def get_financials(symbol: str, statement: str = "income") -> str:
    """
    Get annual financial statement data.

    Args:
        symbol:    Ticker symbol, e.g. "AAPL"
        statement: One of "income", "balance", "cashflow".

    Returns:
        JSON string of the financial statement (latest 4 years).
    """
    ticker = yf.Ticker(symbol)
    if statement == "income":
        df = ticker.financials
    elif statement == "balance":
        df = ticker.balance_sheet
    elif statement == "cashflow":
        df = ticker.cashflow
    else:
        return json.dumps({"error": f"Unknown statement type '{statement}'. Use income/balance/cashflow."})

    if df is None or df.empty:
        return json.dumps({"error": f"No {statement} data for '{symbol}'."})

    df.columns = [c.strftime("%Y-%m-%d") if hasattr(c, "strftime") else str(c) for c in df.columns]
    result = {
        "symbol": symbol,
        "statement": statement,
        "data": df.to_dict(),
    }
    return json.dumps(result, ensure_ascii=False, default=str)


@mcp.tool()
def search_symbols(query: str, market: str = "") -> str:
    """
    Search for ticker symbols by keyword / company name.

    Args:
        query:  Search keyword, e.g. "Tesla", "semiconductor"
        market: Optional market filter: "US", "HK", "CN" (China A-share), "JP", etc.
                Leave empty for global search.

    Returns:
        JSON list of matching symbols with name and exchange info.
    """
    # yfinance doesn't have native search; we use the Yahoo Finance search API
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={parse.quote(query)}&quotesCount=15&newsCount=0"
    req = request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

    try:
        with request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        return json.dumps({"error": f"Search failed: {e}"})

    quotes = data.get("quotes", [])

    # Filter by market if specified
    market_map = {
        "US": ["NMS", "NYQ", "NGM", "PCX", "BTS", "NAS"],
        "HK": ["HKG"],
        "CN": ["SHH", "SHZ"],
        "JP": ["JPX"],
        "UK": ["LSE"],
    }
    if market.upper() in market_map:
        allowed = market_map[market.upper()]
        quotes = [q for q in quotes if q.get("exchange") in allowed]

    results = []
    for q in quotes:
        results.append({
            "symbol": q.get("symbol"),
            "name": q.get("shortname") or q.get("longname"),
            "type": q.get("quoteType"),
            "exchange": q.get("exchange"),
            "exchDisp": q.get("exchDisp"),
        })

    return json.dumps(results, ensure_ascii=False, default=str)


# ── entrypoint ───────────────────────────────────────────────────────
if __name__ == "__main__":
    transport = "stdio"
    if "--sse" in sys.argv:
        transport = "sse"
    mcp.run(transport=transport)
