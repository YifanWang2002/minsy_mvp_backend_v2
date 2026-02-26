# Quantitative Strategy DSL — Design Specification v1.0.0

## 1. Overview & Design Goals

This JSON-based DSL describes **what** to trade, not **how** to execute. It is designed to be:

- **AI-generatable**: Strict schema + deterministic naming → stable, validatable LLM output.
- **Machine-parseable**: Direct mapping to backtest/execution engine internals.
- **Forward-compatible**: Reserved extension points + `x-` prefix fields → non-destructive evolution.

The DSL describes the strategy logic snapshot only. Everything else is managed externally:

| Managed by AI (in DSL JSON) | Managed by backend (NOT in DSL) |
|---|---|
| `strategy.name`, `strategy.description` | `strategy_id`, `strategy_version`, `updated_at`, `author_user_id` |
| `universe`, `timeframe`, `factors` | Backtest date range, initial capital, commission, slippage |
| `trade.long`, `trade.short` | Version history, AI reasoning trace, backtest results |

**Workflow**: AI generates DSL JSON → user reviews/tweaks in frontend → frontend POSTs to backend → backend injects metadata, assigns `strategy_id`, stores → returns `strategy_id` to client → AI uses `strategy_id` in subsequent MCP tool calls (backtest, update, etc.).

---

## 2. Top-Level Structure

```json
{
  "dsl_version": "1.0.0",
  "strategy":    { "name": "...", "description": "..." },
  "universe":    { "market": "crypto", "tickers": ["BTCUSDT"] },
  "timeframe":   "4h",
  "factors":     { "ema_20": {...}, "rsi_14": {...} },
  "trade":       { "long": {...}, "short": {...} }
}
```

All top-level fields are required except that `trade` needs at least one of `long`/`short`.

### 2.1 Strict Field Control

Every object uses `"additionalProperties": false` to prevent field drift (the most common AI generation failure mode). Extension fields are allowed via `"patternProperties": { "^x-": {} }` — any key starting with `x-` passes validation. This gives both strictness and extensibility.

---

## 3. Factor System

### 3.1 Why "factors" not "indicators"

The `factors` key is named to accommodate future expansion beyond technical indicators: price action patterns, ML model outputs, cross-sectional rankings, etc. All are computational units that produce named output series.

### 3.2 Factor Definition

```json
"factors": {
  "ema_20": {
    "type": "ema",
    "params": { "period": 20, "source": "close" }
  },
  "macd_12_26_9": {
    "type": "macd",
    "params": { "fast": 12, "slow": 26, "signal": 9, "source": "close" },
    "outputs": ["macd_line", "signal", "histogram"]
  }
}
```

The key of each entry IS the deterministic factor ID. `type` + `params` define computation; `outputs` optionally declares multi-output names (engine may infer).

### 3.3 Deterministic ID Naming Convention

```
{type}_{numeric_params_in_canonical_order}[_{non_default_source}]
```

Rules:
1. Lowercase + underscores only: `^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$`
2. Numeric params in canonical order (defined per factor type in the catalogue)
3. Default `source` is `close` — omit from ID when default
4. Non-default source appended: `ema_20_typical`

| type | params | ID |
|------|--------|----|
| ema | `{period:20}` | `ema_20` |
| ema | `{period:20, source:"typical"}` | `ema_20_typical` |
| rsi | `{period:14}` | `rsi_14` |
| macd | `{fast:12, slow:26, signal:9}` | `macd_12_26_9` |
| bbands | `{period:20, std_dev:2}` | `bbands_20_2` |
| atr | `{period:14}` | `atr_14` |
| stoch | `{k_period:14, k_smooth:3, d_period:3}` | `stoch_14_3_3` |

**Guarantee**: Same type + same params → same ID → computed once across all tenants.

### 3.4 Multi-Output Factors

Referenced via dot notation in conditions:

| Factor | Outputs |
|--------|---------|
| `macd_*` | `.macd_line`, `.signal`, `.histogram` |
| `bbands_*` | `.upper`, `.middle`, `.lower` |
| `stoch_*` | `.k`, `.d` |
| `ichimoku_*` | `.tenkan`, `.kijun`, `.senkou_a`, `.senkou_b`, `.chikou` |

Single-output factors are referenced directly: `ema_20`, `rsi_14`, `atr_14`.

---

## 4. Reference System (`ref`)

All data series are accessed through a unified `ref` string:

| Pattern | Example | Meaning |
|---------|---------|---------|
| `price.{field}` | `price.close`, `price.typical` | Raw OHLCV price fields |
| `volume` | `volume` | Volume series |
| `{factor_id}` | `ema_20`, `rsi_14` | Single-output factor |
| `{factor_id}.{output}` | `macd_12_26_9.histogram` | Multi-output factor sub-line |

Regex: `^(price\.(open|high|low|close|hl2|hlc3|ohlc4|typical)|volume|[a-z][a-z0-9]*(?:_[a-z0-9]+)*(?:\.[a-z][a-z0-9_]*)?)$`

### 4.1 Operands

An `operand` in conditions is either:

```json
42.5                                      // literal number
{"ref": "ema_20"}                         // current bar value
{"ref": "price.close", "offset": -1}      // previous bar close
{"ref": "macd_12_26_9.signal", "offset": -2}  // 2 bars ago
```

`offset` must be ≤ 0 (no future look). Default is 0.

---

## 5. Condition System

Conditions form a **recursive boolean tree**. Entry and exit signals are both expressed as condition trees.

### 5.1 Branch Nodes (Boolean Combinators)

```json
{ "all": [ cond1, cond2, ... ] }    // AND: all must be true
{ "any": [ cond1, cond2, ... ] }    // OR: at least one true
{ "not": cond }                      // NOT: single child, inverted
```

Nest freely to any depth. `not` takes a single condition, not an array.

### 5.2 Leaf Nodes

#### `cmp` — Comparison
```json
{ "cmp": { "left": {"ref":"rsi_14"}, "op": "gt", "right": 70 } }
```
Operators: `gt`, `gte`, `lt`, `lte`, `eq`, `neq`

Both sides can be any operand (factor ref, price ref, literal). Examples:
- Factor vs constant: `rsi_14 gt 70`
- Factor vs factor: `ema_9 gt ema_21`
- Factor vs price: `ema_50 lt price.close`
- Price vs price: `price.close gt price.open` (bullish bar)
- With offset: `price.close gt price.close[-1]`

#### `cross` — Crossover/Crossunder Events
```json
{ "cross": { "a": {"ref":"ema_9"}, "op": "cross_above", "b": {"ref":"ema_21"} } }
```
Operators: `cross_above`, `cross_below`

Fires on the **single bar** where the crossing occurs:
- `cross_above`: `a[0] > b[0] AND a[-1] <= b[-1]`
- `cross_below`: `a[0] < b[0] AND a[-1] >= b[-1]`

#### `ref` — Boolean Factor Reference
```json
{ "ref": "is_bullish_engulfing_1" }
```
Directly references a boolean factor output. For future use with pattern detection factors.

#### `temporal` — RESERVED (v1.1+)
Three sub-types reserved for future implementation:
- `within_bars`: condition was true within last N bars
- `sequence`: condition A, then condition B within N bars
- `bars_since`: bars since condition was last true compared to threshold

Engines may reject `temporal` nodes in v1 runtime. Schema allows them for forward compatibility.

---

## 6. Trade Rules (Long & Short)

Long and short are **independently defined**. This is deliberate:
- Many factors aren't symmetrically invertible (bullish engulfing → bearish engulfing, not "bullish engulfing failure")
- Some strategies are inherently asymmetric
- AI generates each side explicitly, ensuring correctness

```json
"trade": {
  "long":  { "entry": {...}, "exits": [...], "position_sizing": {...} },
  "short": { "entry": {...}, "exits": [...], "position_sizing": {...} }
}
```

Omit a key entirely to disable that side (at least one required).

---

## 7. Exit System

### 7.1 Exits as Array (First-Hit-Wins)

Exits are an **array** of `exit_rule` objects. Multiple exits coexist; whichever triggers first closes the position. This array design supports future expansion (trailing stop, partial close, scale-out) without structural changes.

```json
"exits": [
  { "type": "signal_exit", "name": "...", "condition": {...} },
  { "type": "stop_loss",   "name": "...", "stop": {...} },
  { "type": "bracket_rr",  "name": "...", "stop": {...}, "risk_reward": 2.0 }
]
```

### 7.2 Exit Types

| type | Required fields | Behavior |
|------|----------------|----------|
| `signal_exit` | `condition` | Fires when condition tree evaluates true |
| `stop_loss` | `stop` | Fixed stop loss level from entry |
| `take_profit` | `take` | Fixed take profit level from entry |
| `bracket_rr` | `risk_reward` + exactly one of `stop`/`take` | Auto-derives the missing side from risk-reward ratio |

### 7.3 Stop Specification

```json
{ "kind": "points",       "value": 50 }           // 50 price units
{ "kind": "pct",          "value": 0.025 }         // 2.5% (decimal fraction)
{ "kind": "atr_multiple", "atr_ref": "atr_14", "multiple": 2.0 }
```

### 7.4 Exit Priority (Same-Bar Tie-Break)

When multiple exits trigger on the same bar, recommended engine priority:
1. `stop_loss` (checked on intrabar high/low)
2. `take_profit` / `bracket_rr` (checked on intrabar high/low)
3. `signal_exit` (evaluated at bar close)

Or deterministically by array order. The engine must document its tie-break rule.

---

## 8. Position Sizing (Optional)

```json
"position_sizing": {
  "mode": "pct_equity",
  "pct": 0.25
}
```

| mode | Required field | Meaning |
|------|---------------|---------|
| `fixed_qty` | `qty` | Fixed number of units/lots |
| `fixed_cash` | `cash` | Fixed dollar amount per trade |
| `pct_equity` | `pct` | Fraction of current equity (0.25 = 25%) |

---

## 9. Extensibility Architecture

### 9.1 `x-` Extension Fields

Every object allows `x-` prefixed fields for platform-specific metadata, debugging info, or experimental features — without breaking schema validation.

### 9.2 Version Negotiation

`dsl_version` uses semver. Engine reads it and:
- If supported → parse normally
- If minor version higher → parse with backward compat (ignore unknown leaf types gracefully)
- If major version higher → reject with clear error

### 9.3 Future Extension Roadmap

| Version | Capability | Extension Point |
|---------|-----------|----------------|
| v1.1.0 | Price action events + temporal sequences | New factor types + `temporal` condition nodes |
| v1.2.0 | Cross-sectional (ranking, z-score) | New factor types + `_cross_sectional` condition |
| v1.3.0 | ML signal integration | New factor types + `_ml_signal` condition |
| v1.4.0 | Trailing stop, partial exit | New `exit_rule.type` values |

All extensions plug into existing structures (new factor types, new condition leaf types, new exit types). The core tree architecture is stable.

---

## 10. Validation Rules

### 10.1 Schema Validation (JSON Schema Draft 2020-12)

Enforces: field existence, types, enums, recursive structure, `additionalProperties: false`.

### 10.2 Semantic Validation (Engine-Side)

1. Every factor key in `factors` must match the deterministic ID convention for its `type` + `params`.
2. Every `ref` in conditions must resolve to: a `price.*` field, `volume`, or a key in `factors` (optionally with `.output`).
3. Multi-output refs (e.g., `macd_12_26_9.histogram`) must reference a valid output name for that factor type.
4. `not` conditions must wrap a single condition (not an array — enforced by schema).
5. `atr_ref` in `stop_spec` must reference a valid ATR-type factor.
6. `bracket_rr` must have exactly one of `stop`/`take` (not both, not neither — enforced by schema).
7. `temporal` nodes may be rejected at runtime in v1 (engine flag).
8. All `offset` values in operands must be ≤ 0.
9. At least one of `trade.long` / `trade.short` must exist (enforced by schema `anyOf`).
