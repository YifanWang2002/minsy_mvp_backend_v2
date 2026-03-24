# Factor System Regression Report (2026-03-17)

## Scope

验证因子体系改造后以下链路的可用性与一致性：

1. 因子注册/暴露（MCP catalog + runtime policy）
2. pre-strategy regime 计算（统一因子引擎）
3. backtest 因子计算
4. live 信号运行时因子计算
5. trade snapshot 因子序列与 pane 归类
6. P0~P3 改造约束与回归检查

## Executed Test Matrix

### A) Refactor Gate Checks

- `scripts/refactor_factor_checks.py --mode smoke`
- `scripts/refactor_factor_checks.py --mode p0`
- `scripts/refactor_factor_checks.py --mode p1`
- `scripts/refactor_factor_checks.py --mode p2`
- `scripts/refactor_factor_checks.py --mode p3`

结果：全部 `all checks passed`。
日志：
- `refactor_smoke.log`
- `refactor_p0.log`
- `refactor_p1.log`
- `refactor_p2.log`
- `refactor_p3.log`

### B) Pytest Chain Coverage Matrix

执行：

- `test/domain/strategy/test_strategy_pipeline_live.py` → `3 passed`
- `test/domain/trading_runtime/test_signal_runtime.py` → `5 passed`
- `test/domain/market_data/test_regime_feature_snapshot.py` → `2 passed`
- `test/domain/market_data/test_regime_family_scoring.py` → `2 passed`
- `test/domain/market_data/test_regime_timeframe_mapper.py` → `2 passed`
- `test/mcp/market_data/test_market_data_tools.py::test_090_pre_strategy_get_regime_snapshot_builds_snapshot_and_cache` → `1 passed`
- `test/api/agents/test_strategy_runtime_policy.py` → `4 passed`
- `tests/test_domain/test_backtest_trade_snapshot_unit.py` → `7 passed`
- `test/mcp/strategy/test_strategy_tools_factor_catalog_unit.py` → `4 passed`
- `test/mcp/router/test_mcp_tools_call_live.py -k indicator_catalog` → `2 passed`

合计：`32 passed`（分多条命令执行，见 `pytest_matrix.log`）。

### C) Multi-scenario Consistency (Real DSL, Multi-factor, Multi-timeframe)

脚本：`multi_scenario_consistency.py`

- 策略变体：`base`, `trend_plus`, `mr_plus`, `regime_plus`
- 时间线：`15m`, `1h`, `1d`
- 行情轮廓：`mean_reversion`, `trend`, `volatility_shock`
- 总案例数：`36`
- 校验项：
  - DSL 语义验证通过
  - `prepare_backtest_frame` 与 `LiveSignalRuntime._build_enriched_frame` 因子列一致
  - 最近 80 根上的最大绝对误差 `MAX_GLOBAL_ABS_DIFF=0.000e+00`
  - regime 家族推荐值落在允许集合内（该批真实 DSL/数据生成组合下推荐均为 `mean_reversion`）

结果：`RESULT=PASS`
日志：`multi_scenario_consistency.log`

### D) Regime Family Discriminative Check (Synthetic Snapshot)

脚本输出：`regime_family_discriminative.log`

- `trend_case` -> `trend_continuation`
- `mean_reversion_case` -> `mean_reversion`
- `volatility_case` -> `volatility_regime`

该检查用于验证 family scoring 本身具备区分能力，避免把 C 节“真实 DSL 批次恰好都落在均值回归区”误解为分类器失效。

## Coverage Snapshot

Coverage 命令结果见 `coverage_matrix.log`。

关键模块：

- `packages/domain/market_data/regime/feature_snapshot.py` → `91%`
- `packages/domain/backtest/factors.py` → `73%`
- `packages/domain/backtest/trade_snapshot.py` → `73%`
- `packages/domain/trading/runtime/signal_runtime.py` → `74%`
- `packages/domain/strategy/feature/contracts.py` → `52%`
- `apps/mcp/domains/strategy/tools.py` → `23%`
- `apps/mcp/domains/market_data/tools.py` → `31%`
- Test run total (selected scope) → `34%`

## Consistency Conclusions

1. **跨链路一致性**：在 4 个真实 DSL 变体 × 3 个时间线 × 3 类行情轮廓下，backtest 与 live 的因子列在最近 80 根上误差为 0。
2. **pre-strategy 连续性**：regime snapshot + family scoring 路径通过，且缓存路径测试通过。
3. **MCP 因子暴露稳定**：catalog 工具可用、注册表字段齐全、工具注册不包含 `get_indicator_detail`。
4. **P0~P3 契约回归**：专项 gate checks 全通过，覆盖单一来源、合同层、文档同步、性能缓存、弃用元数据等。

## Residual Risk (明确保留)

1. coverage 不是全仓库全模块；当前重点覆盖“因子体系相关核心路径”。
2. `apps/mcp/domains/strategy/tools.py` 与 `apps/mcp/domains/market_data/tools.py` 仍有大量分支未覆盖（错误分支、参数异常分支、历史兼容分支）。
3. real-time trading 的 `runtime_service.py` 等大模块未在本轮纳入覆盖。

## Recommended Next Test Increment

1. 增加 `strategy/tools.py` 的失败路径与边界参数表驱动测试。
2. 增加 `market_data/tools.py` 的异常路径与缺数分支覆盖。
3. 将 `multi_scenario_consistency.py` 纳入 CI（或转换为 pytest 参数化用例）。
