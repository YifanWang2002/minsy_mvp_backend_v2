---
skill: tradingview_chart
description: >
  Embed a TradingView advanced chart via <AGENT_UI_JSON> blocks.
  Used when the agent wants to show a live financial chart for a specific symbol.
format_tag: AGENT_UI_JSON
---

# TradingView Chart Output Format

When showing a live financial chart to the user, emit exactly one wrapped JSON block:

```
<AGENT_UI_JSON>{"type":"tradingview_chart","symbol":"<SYMBOL>","interval":"<timeframe>"}</AGENT_UI_JSON>
```

# Parameters

## symbol (required)

The TradingView symbol identifier. Format varies by asset class:

- US equities: Use ticker only, NO exchange prefix. Examples: `SPY`, `QQQ`, `AAPL`, `NVDA`, `TSLA`
- Crypto: Use `BINANCE:` prefix with USDT pair. Examples: `BINANCE:BTCUSDT`, `BINANCE:ETHUSDT`, `BINANCE:SOLUSDT`
- Forex: Use `FX:` prefix. Examples: `FX:EURUSD`, `FX:GBPUSD`, `FX:USDJPY`
- Commodities (gold, oil, etc.): Use forex-style symbols. Examples: `XAUUSD` (gold), `XAGUSD` (silver), `USOIL` (crude oil)

IMPORTANT:
- For US stocks/ETFs (SPY, QQQ, AAPL, etc.), use the ticker directly WITHOUT any exchange prefix.
- Do NOT use `NYSE:SPY` or `NASDAQ:QQQ` - just use `SPY` or `QQQ`.
- Futures contracts (ES, NQ, GC, CL, etc.) are NOT supported in the embedded widget. Use ETF proxies instead:
  - ES (S&P 500 futures) -> use `SPY`
  - NQ (Nasdaq futures) -> use `QQQ`
  - GC (Gold futures) -> use `XAUUSD`
  - CL (Crude Oil futures) -> use `USOIL`

## interval (optional, default: "D")

The chart timeframe / interval.

Valid values:
- Minutes: `"1"`, `"5"`, `"15"`, `"30"`, `"60"`, `"240"`
- Daily: `"D"`
- Weekly: `"W"`
- Monthly: `"M"`

# Rules

- `type` must always be `"tradingview_chart"`.
- `symbol` must follow the format rules above exactly.
- `interval` is optional; omit to default to daily (`"D"`).
- Do **not** wrap the JSON in markdown code fences.
- The chart theme (light/dark) is automatically determined by the user's UI setting — do **not** include a `theme` field.
- You may emit multiple `tradingview_chart` blocks in a single turn (e.g. to compare two symbols).
- Always include a brief text description before the chart explaining what it shows.
- CRITICAL: When user mentions a tradable instrument, you MUST show the chart. Never skip the chart display.

# Examples

Show AAPL daily chart:
```
以下是 AAPL 的实时行情图表：
<AGENT_UI_JSON>{"type":"tradingview_chart","symbol":"AAPL","interval":"D"}</AGENT_UI_JSON>
```

Show BTC 4-hour chart:
```
以下是 BTC/USDT 的 4 小时线图表：
<AGENT_UI_JSON>{"type":"tradingview_chart","symbol":"BINANCE:BTCUSDT","interval":"240"}</AGENT_UI_JSON>
```

Compare SPY and QQQ (note: no exchange prefix for US equities):
```
以下分别是 SPY 和 QQQ 的日线行情：
<AGENT_UI_JSON>{"type":"tradingview_chart","symbol":"SPY","interval":"D"}</AGENT_UI_JSON>
<AGENT_UI_JSON>{"type":"tradingview_chart","symbol":"QQQ","interval":"D"}</AGENT_UI_JSON>
```

Show gold price (using forex-style symbol, not futures):
```
以下是黄金的实时行情：
<AGENT_UI_JSON>{"type":"tradingview_chart","symbol":"XAUUSD","interval":"D"}</AGENT_UI_JSON>
```
