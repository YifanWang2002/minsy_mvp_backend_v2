"""YFinance tools registered into the modular MCP server."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request

import yfinance as yf
from mcp.server.fastmcp import FastMCP

TOOL_NAMES: tuple[str, ...] = (
    "get_quote",
    "get_history",
    "get_financials",
    "search_symbols",
)

VALID_INTERVALS = [
    "1m",
    "2m",
    "5m",
    "15m",
    "30m",
    "60m",
    "90m",
    "1h",
    "1d",
    "5d",
    "1wk",
    "1mo",
    "3mo",
]

VALID_PERIODS = [
    "1d",
    "5d",
    "1mo",
    "3mo",
    "6mo",
    "1y",
    "2y",
    "5y",
    "10y",
    "ytd",
    "max",
]


def _ticker_info_summary(ticker: yf.Ticker) -> dict:
    info = ticker.info or {}
    keys = [
        "shortName",
        "longName",
        "symbol",
        "exchange",
        "market",
        "quoteType",
        "currency",
        "marketCap",
        "enterpriseValue",
        "trailingPE",
        "forwardPE",
        "dividendYield",
        "beta",
        "fiftyTwoWeekHigh",
        "fiftyTwoWeekLow",
        "regularMarketPrice",
        "regularMarketOpen",
        "regularMarketDayHigh",
        "regularMarketDayLow",
        "regularMarketVolume",
        "averageVolume",
        "sector",
        "industry",
        "country",
    ]
    return {k: info[k] for k in keys if k in info}


def register_yfinance_tools(mcp: FastMCP) -> None:
    """Register yfinance-style market tools."""

    @mcp.tool()
    def get_quote(symbol: str) -> str:
        """
        Get the real-time (latest) quote / summary info for a symbol.

        Args:
            symbol: Ticker symbol, e.g. "AAPL", "0700.HK", "USDJPY=X", "BTC-USD"
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
            period:   Data period, ignored when start/end are provided.
            interval: Bar interval.
            start:    Start date string, e.g. "2024-01-01". Optional.
            end:      End date string, e.g. "2024-12-31". Optional.
        """
        if interval not in VALID_INTERVALS:
            return json.dumps(
                {"error": f"Invalid interval '{interval}'. Choose from {VALID_INTERVALS}"}
            )
        if not start and period not in VALID_PERIODS:
            return json.dumps(
                {"error": f"Invalid period '{period}'. Choose from {VALID_PERIODS}"}
            )

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
            return json.dumps(
                {"error": f"No history data for '{symbol}' with given parameters."}
            )

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
        """
        ticker = yf.Ticker(symbol)
        if statement == "income":
            df = ticker.financials
        elif statement == "balance":
            df = ticker.balance_sheet
        elif statement == "cashflow":
            df = ticker.cashflow
        else:
            return json.dumps(
                {
                    "error": (
                        f"Unknown statement type '{statement}'. "
                        "Use income/balance/cashflow."
                    )
                }
            )

        if df is None or df.empty:
            return json.dumps({"error": f"No {statement} data for '{symbol}'."})

        df.columns = [
            c.strftime("%Y-%m-%d") if hasattr(c, "strftime") else str(c) for c in df.columns
        ]
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
            market: Optional market filter: "US", "HK", "CN", "JP", etc.
        """
        url = (
            "https://query2.finance.yahoo.com/v1/finance/search"
            f"?q={urllib.parse.quote(query)}&quotesCount=15&newsCount=0"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
        except Exception as exc:  # noqa: BLE001
            return json.dumps({"error": f"Search failed: {exc}"})

        quotes = data.get("quotes", [])
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
        for quote in quotes:
            results.append(
                {
                    "symbol": quote.get("symbol"),
                    "name": quote.get("shortname") or quote.get("longname"),
                    "type": quote.get("quoteType"),
                    "exchange": quote.get("exchange"),
                    "exchDisp": quote.get("exchDisp"),
                }
            )

        return json.dumps(results, ensure_ascii=False, default=str)
