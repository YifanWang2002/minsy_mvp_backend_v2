---
name: quant-strategy-dsl-generator
description: >
  Generate valid Quant Strategy DSL JSON (v1.0.0) for backtesting and live trading.
  Use when user asks to design, create, generate, modify, or iterate on a quantitative
  trading strategy. Produces a single JSON document conforming to strategy_dsl.schema.json.
  Covers: factor definitions, entry/exit condition trees, stop loss/take profit, position sizing.
---

# Quant Strategy DSL Generator

## What You Produce

A single JSON strategy document conforming to DSL v1.0.0. The output must be parseable by the backtest engine and reproducible by AI on re-generation.

**You do NOT generate**: `strategy_id`, `author_user_id`, `strategy_version`, `updated_at` — these are injected by the backend after the user confirms and submits.

## Gathering Information

Ask at most 2–3 questions when key info is missing. Prefer multiple-choice defaults. Skip questions for anything already provided.

1. **Market + tickers + timeframe** — e.g. crypto/BTCUSDT/4h
2. **Long/short intent** — long-only, short-only, or both? If both, define each side explicitly. Never auto-mirror.
3. **Risk management** — SL type (points / pct / ATR multiple), TP or bracket RR. Default: ATR×2 SL + RR 2.0 bracket.

## Top-Level JSON Structure

```json
{
  "dsl_version": "1.0.0",
  "strategy":  { "name": "...", "description": "..." },
  "universe":  { "market": "...", "tickers": ["..."] },
  "timeframe": "4h",
  "factors":   { ... },
  "trade":     { "long": {...}, "short": {...} }
}
```

## Factor Naming Rules (Critical)

Factor IDs are deterministic. Never invent aliases. Same type + same params = same ID everywhere.

**Convention**: `{type}_{numeric_params_canonical_order}[_{source_if_not_close}]`

| type | params | ID |
|------|--------|----|
| ema | `{period:20}` | `ema_20` |
| ema | `{period:20, source:"typical"}` | `ema_20_typical` |
| rsi | `{period:14}` | `rsi_14` |
| macd | `{fast:12, slow:26, signal:9}` | `macd_12_26_9` |
| bbands | `{period:20, std_dev:2}` | `bbands_20_2` |
| atr | `{period:14}` | `atr_14` |
| stoch | `{k_period:14, k_smooth:3, d_period:3}` | `stoch_14_3_3` |

**Multi-output dot notation**: `macd_12_26_9.macd_line`, `macd_12_26_9.signal`, `macd_12_26_9.histogram`, `bbands_20_2.upper`, `stoch_14_3_3.k`

**MCP tools**:
- Call `get_indicator_catalog(category="momentum")` (or without category) to inspect available categories and registry contracts.
- Call `get_indicator_detail(indicator="ema")` or `get_indicator_detail(indicator_list=[...])` for full skill detail.

## Reference System

All series accessed via `ref` strings:

- `price.close`, `price.high`, `price.typical`, etc.
- `volume`
- Factor ID: `ema_20`, `rsi_14`
- Factor output: `macd_12_26_9.histogram`

Operands can include `offset` for historical bars: `{"ref": "price.close", "offset": -1}` = previous bar close. Offset must be ≤ 0.

## Condition Tree

### Boolean combinators (branch nodes)

```json
{ "all": [ cond, cond, ... ] }   // AND
{ "any": [ cond, cond, ... ] }   // OR
{ "not": cond }                   // NOT (single child, not array)
```

### Comparison

```json
{ "cmp": { "left": {"ref":"rsi_14"}, "op": "gt", "right": 70 } }
```

Ops: `gt`, `gte`, `lt`, `lte`, `eq`, `neq`. Both sides can be ref or literal number.

### Crossover/Crossunder

```json
{ "cross": { "a": {"ref":"ema_9"}, "op": "cross_above", "b": {"ref":"ema_21"} } }
```

Ops: `cross_above`, `cross_below`. Fires on the single bar where crossing occurs.

### Reserved (DO NOT USE in v1)

`temporal` nodes (`within_bars`, `sequence`, `bars_since`) — schema accepts them but engine will reject.

## Exit Rules (Array, First-Hit-Wins)

Exits are an **array**. Whichever triggers first closes the position.

```json
"exits": [
  {
    "type": "signal_exit",
    "name": "exit_on_ema_cross",
    "condition": { "cross": { "a": {"ref":"ema_9"}, "op": "cross_below", "b": {"ref":"ema_21"} } }
  },
  {
    "type": "stop_loss",
    "name": "sl_atr_2x",
    "stop": { "kind": "atr_multiple", "atr_ref": "atr_14", "multiple": 2.0 }
  },
  {
    "type": "bracket_rr",
    "name": "tp_rr2",
    "stop": { "kind": "atr_multiple", "atr_ref": "atr_14", "multiple": 2.0 },
    "risk_reward": 2.0
  }
]
```

### Exit types

| type | Required | Behavior |
|------|----------|----------|
| `signal_exit` | `condition` | Fires when condition tree is true |
| `stop_loss` | `stop` | Fixed SL from entry price |
| `take_profit` | `take` | Fixed TP from entry price |
| `bracket_rr` | `risk_reward` + ONE of `stop`/`take` | Auto-derives the other side |

### Stop spec formats

```json
{ "kind": "points",       "value": 50 }
{ "kind": "pct",          "value": 0.025 }       // 2.5% as decimal
{ "kind": "atr_multiple", "atr_ref": "atr_14", "multiple": 2.0 }
```

## Long vs Short — Always Explicit

Define each side independently. When user says "and the opposite for shorts":

- `cross_above` → `cross_below`
- `gt` → `lt` (for thresholds)
- RSI > 70 filter → RSI < 30 filter
- But always think about whether the inversion is semantically correct for each condition.

User says "long only" → omit `short` from `trade` entirely.

## Position Sizing (Optional)

```json
"position_sizing": { "mode": "pct_equity", "pct": 0.25 }
```

Modes: `fixed_qty` (+ `qty`), `fixed_cash` (+ `cash`), `pct_equity` (+ `pct` as decimal fraction).

## Workflow After Generation

1. You output the JSON strategy.
2. User reviews and confirms → frontend sends to backend.
3. Backend returns `strategy_id`.
4. User provides `strategy_id` → you call MCP tools:
   - `strategy_validate_dsl(dsl_json=...)` → validate edits before persistence
   - `strategy_get_dsl(session_id=..., strategy_id=...)` → fetch latest payload + version
   - `strategy_patch_dsl(session_id=..., strategy_id=..., patch_json=..., expected_version=...)` → persist minimal changes
   - `strategy_list_versions(session_id=..., strategy_id=..., limit=...)` → inspect revision history
   - `strategy_diff_versions(session_id=..., strategy_id=..., from_version=..., to_version=...)` → compare changes
   - `strategy_get_version_dsl(session_id=..., strategy_id=..., version=...)` → fetch historical snapshot
   - `strategy_rollback_dsl(session_id=..., strategy_id=..., target_version=..., expected_version=...)` → rollback by creating a new latest version
   - `strategy_upsert_dsl(session_id=..., strategy_id=..., dsl_json=...)` → fallback full-payload persistence
   - `backtest_create_job(strategy_id=..., ...)` and `backtest_get_job(job_id=...)` → run and review stress test
5. Iterate based on results.

Use only the currently documented MCP tools in this file.

## Iteration Heuristics

| Problem | Likely Fix |
|---------|-----------|
| Low win rate | Tighten entry filters, add confirmation conditions |
| Good WR, low return | Exits too tight — widen TP or switch to signal exit |
| High drawdown | Tighten SL, add trend filter (e.g. price > ema_200) |
| Too few trades | Relax filters or widen parameter ranges |
| Too many low-quality trades | Add confluence conditions |

## Common Strategy Templates

### MA Crossover + Filter

Entry: ema_fast cross_above ema_slow AND rsi lt 70
Exit: signal (ema cross back) + ATR SL + bracket RR

### Mean Reversion (Bollinger)

Entry: price.close cross_below bbands.lower AND rsi lt 30
Exit: signal (price cross_above bbands.middle) + pct SL

### Momentum Breakout

Entry: price.close gt highest_close_20 (via cmp with offset) AND macd.histogram gt 0
Exit: ATR SL + bracket RR 2.5

## Validation Checklist

Before outputting, verify:

- [ ] `dsl_version` is `"1.0.0"`
- [ ] No backend-managed fields (no strategy_id, version, updated_at, author_user_id)
- [ ] Every factor ref in conditions exists as a key in `factors`
- [ ] Multi-output refs use correct dot notation
- [ ] `not` wraps a single condition (not an array)
- [ ] All offsets ≤ 0
- [ ] `atr_ref` in stops points to a defined ATR factor
- [ ] `bracket_rr` has exactly one of `stop`/`take`
- [ ] At least one of `long`/`short` in `trade`
- [ ] Factor IDs match the naming convention for their type + params
- [ ] `pct` values in stop_spec and position_sizing are decimal fractions (0.025 not 2.5)

## Factor Catalogue Categories

Call `get_indicator_catalog(category="...")` to explore:

| Category | Examples |
|----------|---------|
| overlap | ema, sma, wma, dema, tema, kama, bbands, ichimoku |
| momentum | rsi, macd, stoch, stochrsi, cci, williams_r, roc, mfi, ppo |
| volatility | atr, natr, true_range, keltner, donchian |
| volume | obv, ad, adosc, vwap, cmf |
| utils | zscore, linear regression, statistical helpers |

## Error Reference

| Error | Fix |
|-------|-----|
| `UNKNOWN_FACTOR_REF` | Add missing factor to `factors` or fix the ref string |
| `INVALID_OUTPUT_NAME` | Check multi-output table (e.g. `.macd_line` not `.line`) |
| `TEMPORAL_NOT_SUPPORTED` | Remove temporal nodes — not available in v1 runtime |
| `INVALID_NOT_STRUCTURE` | `not` must wrap a single condition object, not an array |
| `MISSING_ATR_REF` | `atr_multiple` stop needs `atr_ref` pointing to ATR factor |
| `BRACKET_RR_CONFLICT` | `bracket_rr` needs exactly one of `stop`/`take` |
| `FUTURE_LOOK` | Offset must be ≤ 0 |
| `NO_TRADE_SIDE` | At least one of `long`/`short` required |
| `ADDITIONAL_PROPERTY` | Unknown field in strict object — check for typos |
