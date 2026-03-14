# 因子系统 README（中文）

> 路径说明：旧路径常写作 `src/engine/feature`。当前主线代码已迁移到  
> `backend/packages/domain/strategy/feature`（本文件所在位置）。

## 1. 我们现在有哪些因子？怎么分类？

### 1.1 顶层因子命名空间（`FactorKind`）

定义在 `specs.py`：

- `indicator`
- `price_event`
- `ml_signal`
- `custom`

当前真实注册情况（运行时）：

- `indicator`: 161
- `price_event`: 0
- `ml_signal`: 0
- `custom`: 0

也就是说，当前策略 DSL/回测链路里“可直接使用”的是 **指标型因子（indicator）**。

### 1.2 指标因子分类（`IndicatorCategory`）

- `overlap`（25）
- `momentum`（22）
- `volatility`（11）
- `volume`（13）
- `candle`（65）
- `utils`（25）

合计：161。

### 1.3 当前全部因子（按分类）

#### overlap (25)
`bbands`, `dema`, `donchian`, `ema`, `hl2`, `hlc3`, `hma`, `ht_trendline`, `ichimoku`, `kama`, `kc`, `linearreg`, `mama`, `midpoint`, `midprice`, `ohlc4`, `sar`, `sma`, `supertrend`, `t3`, `tema`, `trima`, `vwap`, `wcp`, `wma`

#### momentum (22)
`adx`, `adxr`, `apo`, `aroon`, `bop`, `cci`, `chop`, `cmo`, `dx`, `macd`, `minus_dm`, `mom`, `plus_dm`, `ppo`, `roc`, `rsi`, `stoch`, `stochf`, `stochrsi`, `trix`, `ultosc`, `willr`

#### volatility (11)
`atr`, `atrts`, `chandelier`, `massi`, `natr`, `pdist`, `rvi`, `stdev`, `trange`, `ui`, `variance`

#### volume (13)
`ad`, `adosc`, `cmf`, `efi`, `eom`, `kvo`, `mfi`, `nvi`, `obv`, `pvi`, `pvo`, `pvt`, `vwma`

#### candle (65)
`cdl_2crows`, `cdl_3blackcrows`, `cdl_3inside`, `cdl_3linestrike`, `cdl_3outside`, `cdl_3starsinsouth`, `cdl_3whitesoldiers`, `cdl_abandonedbaby`, `cdl_advanceblock`, `cdl_all`, `cdl_belthold`, `cdl_breakaway`, `cdl_closingmarubozu`, `cdl_concealbabyswall`, `cdl_counterattack`, `cdl_darkcloudcover`, `cdl_doji`, `cdl_dojistar`, `cdl_dragonflydoji`, `cdl_engulfing`, `cdl_eveningdojistar`, `cdl_eveningstar`, `cdl_gapsidesidewhite`, `cdl_gravestonedoji`, `cdl_hammer`, `cdl_hangingman`, `cdl_harami`, `cdl_haramicross`, `cdl_highwave`, `cdl_hikkake`, `cdl_hikkakemod`, `cdl_homingpigeon`, `cdl_identical3crows`, `cdl_inneck`, `cdl_inside`, `cdl_invertedhammer`, `cdl_kicking`, `cdl_kickingbylength`, `cdl_ladderbottom`, `cdl_longleggeddoji`, `cdl_longline`, `cdl_marubozu`, `cdl_matchinglow`, `cdl_mathold`, `cdl_morningdojistar`, `cdl_morningstar`, `cdl_onneck`, `cdl_piercing`, `cdl_rickshawman`, `cdl_risefall3methods`, `cdl_separatinglines`, `cdl_shootingstar`, `cdl_shortline`, `cdl_spinningtop`, `cdl_stalledpattern`, `cdl_sticksandwich`, `cdl_takuri`, `cdl_tasukigap`, `cdl_thrusting`, `cdl_tristar`, `cdl_unique3river`, `cdl_upsidegap2crows`, `cdl_xsidegap3methods`, `cdl_z`, `ha`

#### utils (25)
`avgprice`, `beta`, `correl`, `drawdown`, `entropy`, `ht_dcperiod`, `ht_dcphase`, `ht_phasor`, `ht_sine`, `ht_trendmode`, `kurtosis`, `linearreg_angle`, `linearreg_intercept`, `linearreg_slope`, `log_return`, `max`, `median`, `medprice`, `min`, `percent_return`, `quantile`, `skew`, `sum`, `tsf`, `zscore`

---

## 2. 因子在系统里的工作流

1. **注册层**  
   `IndicatorRegistry` 作为适配器，底层写入 `FeatureRegistry`（按 `(kind, name)` 存）。

2. **策略 DSL 校验层**  
   `semantic.py` 会拿 DSL 中 `factors[*].type` 去 `IndicatorRegistry.get(type)` 做语义校验；  
   所以 DSL 可用因子类型本质上来自注册表。

3. **回测执行层**  
   `backtest/factors.py` 用 `IndicatorWrapper.calculate(...)` 计算因子并回填到 DataFrame。

4. **条件求值层**  
   `backtest/condition.py` 用 `ref`（如 `ema_20`、`macd_12_26_9.histogram`）取列并做 `cmp/cross/all/any/not`。

---

## 3. 现在 AI 能通过 MCP 拿到哪些“因子信息”？

在 `apps/mcp/domains/strategy/tools.py` 里，和因子最相关的是：

1. `get_indicator_catalog(category="")`
- 返回指标目录、参数、输出、required_columns、skill 摘要/路径。
- 支持分类：`overlap`, `momentum`, `volatility`, `volume`, `utils`。
- 注意：**`candle` 被刻意从 catalog 输出中排除**（`_EXCLUDED_CATALOG_CATEGORIES`）。

2. `get_indicator_detail(indicator=... / indicator_list=[...])`
- 返回单个/多个指标的详细信息：
  - `registry`（metadata：params/outputs/required_columns 等）
  - `skill_path`
  - `content`（对应 markdown skill 全文）

3. `strategy_list_tunable_params(strategy_id=...)`
- 从策略当前 `factors` 里提取可调参数（`json_path/current_value/min/max/suggested_step`）。

4. `strategy_validate_dsl(dsl_json=...)`
- 会给出因子相关错误（如 `UNSUPPORTED_FACTOR_TYPE`、`UNKNOWN_FACTOR_REF`、`INVALID_OUTPUT_NAME` 等）。

### 3.1 当前“指标 skill 文件”覆盖

目录：`apps/mcp/domains/strategy/skills/indicators/`

已有 8 个：
- `adx`, `atr`, `bbands`, `chop`, `ema`, `macd`, `rsi`, `stoch`

说明：不是每个已注册指标都有独立 skill 文件；没有 skill 文件时，AI 仍可通过 registry 元数据使用该指标。

### 3.2 Agent 侧技能约束（strategy phase）

`apps/api/agents/skills/strategy/stages/schema_only.md` 明确要求：
- 先调用 `get_indicator_catalog`
- 必要时再调用 `get_indicator_detail`

---

## 4. 如果要加入自选因子，怎么做？

### 4.1 推荐路径：先按“指标型因子”接入（当前最稳）

1. 新增指标实现并注册
- 位置：`feature/indicators/categories/*.py`（或 `BaseIndicator + register_custom`）
- 注册时填写 `IndicatorMetadata`（`name/category/params/outputs/required_columns`）

2. 若要在 DSL 中保持 deterministic factor id
- 补充 `semantic.py` 里的 `_FACTOR_PARAM_ORDER`（如需要）

3. 若参数命名与 DSL 不同
- 同步补充 `semantic.py::_to_indicator_params`
- 同步补充 `backtest/factors.py::_to_indicator_params`

4. 若是多输出因子，且你想给 DSL 友好别名
- `semantic.py`：`_MULTI_OUTPUT_DEFAULTS` / `_LEGACY_OUTPUT_ALIASES`
- `backtest/factors.py`：`_DEFAULT_MULTI_OUTPUTS` / `_OUTPUT_COLUMN_ALIASES`
- `mcp strategy tools`：`_INDICATOR_DSL_OUTPUT_ALIASES`

5. 可选：补 indicator skill
- 新建 `apps/mcp/domains/strategy/skills/indicators/<name>.md`

### 4.2 非 indicator 的 price_event / ml_signal（进阶）

虽然 `FactorKind` 已预留，但当前 DSL 语义校验与回测执行都仍以 `IndicatorRegistry` 为核心。  
要做成完整链路，需要至少改：

1. 语义校验：`semantic.py` 不再只按 `IndicatorRegistry.get(type)` 校验  
2. 回测计算：`backtest/factors.py` 增加非指标因子计算入口  
3. MCP：补对应 catalog/detail 工具或扩展现有返回结构

---

## 5. 你提到的两个需求，如何实现？

### 5.1 基于时间的 filter（时段/星期几/月内第 N 个星期几某时刻）

### 当前状态

- DSL schema 里有 `temporal` 节点，但 v1 运行时 **不支持**。
- `semantic.py` 会报 `TEMPORAL_NOT_SUPPORTED`。
- `backtest/condition.py` 遇到 `temporal` 会直接抛异常。

### 实操建议（短期可上线）：做“时间门控因子”

把时间规则做成布尔因子（0/1），然后在条件里直接 `{"ref":"your_time_gate_factor"}`。

可做的因子例子：
- `session_0930_1130`：当天 09:30-11:30 为 1，否则 0
- `weekday_1`：周一为 1
- `nth_weekday_gate`：每月第 N 个星期 X 的特定时段为 1

优点：不改 DSL 结构，改动最小，马上可用。

### 中长期（DSL 原生支持）

新增一个日历/时段条件节点（比如 `calendar`），并同步改：
- `strategy_dsl_schema.json`（condition 新分支）
- `semantic.py`（语义校验）
- `backtest/condition.py`（compile + series evaluate）

### 5.2 价格行为因子（市场周期 classifier / 局部 K 线事件 / 形态）

### 现有基础

- 已有大量蜡烛形态（`candle` 65 个）和部分状态类指标（如 `chop`、`supertrend` 等）。

### 建议接入方式

先按指标型因子落地（最少改动）：

1. 定义新因子（例如）
- `market_cycle_*`：输出 `regime`（-1/0/1）或概率分量
- `micro_event_*`：输出局部事件（如 `sfp`, `breakout`, `inside_outside`）
- `pattern_score_*`：输出形态强度分数

2. 在条件里使用
- 单输出：`{"cmp":{"left":{"ref":"market_cycle_60"},"op":"gt","right":0}}`
- 多输出：`{"ref":"micro_event_20.breakout"}` 或与 `cmp/cross` 组合

3. 如果希望 AI 更稳定调用
- 给该因子补 `skill` 文件 + 在 catalog/detail 里可见

---

## 6. 自检命令（保持 README 与代码一致）

```bash
cd backend
uv run python - <<'PY'
from packages.domain.strategy.feature.indicators import IndicatorRegistry
from packages.domain.strategy.feature.indicators.base import IndicatorCategory
print("TOTAL", len(IndicatorRegistry.list_all()))
for c in IndicatorCategory:
    print(c.value, len(IndicatorRegistry.list_by_category(c)))
PY
```

---

## 7. 小结

- 当前生产链路是“**指标型因子优先**”。
- `time filter` 与更复杂 `price action` 都可以先以“新指标因子”形态快速接入。
- 等你要做 DSL 原生时序语义（temporal/calendar）时，再统一扩 schema + semantic + condition evaluator。
