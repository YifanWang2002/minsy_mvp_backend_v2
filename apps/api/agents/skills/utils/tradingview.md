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
<AGENT_UI_JSON>{"type":"tradingview_chart","symbol":"<EXCHANGE:TICKER>","interval":"<timeframe>"}</AGENT_UI_JSON>
```

# Parameters

## symbol (required)

The TradingView symbol identifier in `EXCHANGE:TICKER` format.

Common examples:
- US equities: `NASDAQ:AAPL`, `NASDAQ:NVDA`, `NYSE:SPY`, `NASDAQ:QQQ`
- Crypto: `BINANCE:BTCUSDT`, `BINANCE:ETHUSDT`, `BINANCE:SOLUSDT`
- Forex: `FX:EURUSD`, `FX:GBPUSD`, `FX:USDJPY`
- Indices: `TVC:SPX`, `TVC:DJI`, `TVC:NDQ`

## interval (optional, default: "D")

The chart timeframe / interval.

Valid values:
- Minutes: `"1"`, `"5"`, `"15"`, `"30"`, `"60"`, `"240"`
- Daily: `"D"`
- Weekly: `"W"`
- Monthly: `"M"`

# Rules

- `type` must always be `"tradingview_chart"`.
- `symbol` must be a valid TradingView symbol string (exchange prefix recommended).
- `interval` is optional; omit to default to daily (`"D"`).
- Do **not** wrap the JSON in markdown code fences.
- The chart theme (light/dark) is automatically determined by the user's UI setting — do **not** include a `theme` field.
- You may emit multiple `tradingview_chart` blocks in a single turn (e.g. to compare two symbols).
- Always include a brief text description before the chart explaining what it shows.

# Examples

Show AAPL daily chart:
```
以下是 AAPL 的实时行情图表：
<AGENT_UI_JSON>{"type":"tradingview_chart","symbol":"NASDAQ:AAPL","interval":"D"}</AGENT_UI_JSON>
```

Show BTC 4-hour chart:
```
以下是 BTC/USDT 的 4 小时线图表：
<AGENT_UI_JSON>{"type":"tradingview_chart","symbol":"BINANCE:BTCUSDT","interval":"240"}</AGENT_UI_JSON>
```

Compare two symbols in one turn:
```
以下分别是 SPY 和 QQQ 的日线行情：
<AGENT_UI_JSON>{"type":"tradingview_chart","symbol":"NYSE:SPY","interval":"D"}</AGENT_UI_JSON>
<AGENT_UI_JSON>{"type":"tradingview_chart","symbol":"NASDAQ:QQQ","interval":"D"}</AGENT_UI_JSON>
```
