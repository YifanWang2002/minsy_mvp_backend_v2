# Price Action Factor Research Framework

基于Al Brooks价格行为学理论的量化因子研究框架。

## 概述

这个框架用于研究和验证价格行为（Price Action）因子在交易策略中的有效性。通过系统化的A/B测试，我们可以识别哪些因子能够提升策略的胜率、盈亏比和整体表现，同时保持合理的交易频率。

## 核心理念

基于Al Brooks的价格行为学，我们关注：

1. **趋势强度** - 通过连续K线、高低点结构等判断趋势质量
2. **K线质量** - 实体大小、收盘位置、相对ATR的幅度
3. **入场时机** - 在趋势中的最佳入场点（回调、突破等）
4. **风险控制** - 通过因子过滤减少低质量交易

## 已实现的因子

### 1. Internal Bar Strength (IBS)
```python
IBS = (Close - Low) / (High - Low)
```
- 衡量收盘价在K线中的位置
- 接近1.0表示收在高位（看涨）
- 接近0.0表示收在低位（看跌）
- **用途**: 做多时要求IBS > 0.6，确保在强势位置入场

### 2. Bar Range vs ATR
```python
Ratio = (High - Low) / ATR(14)
```
- 衡量当前K线相对于近期波动的大小
- > 1.5表示强势动能K线
- < 0.5表示弱势/盘整K线
- **用途**: 要求比值 > 1.2，只在有动能时入场

### 3. Body to Range Ratio
```python
Ratio = |Close - Open| / (High - Low)
```
- 衡量实体占整根K线的比例
- > 0.7表示强方向性K线
- < 0.3表示犹豫/十字星
- **用途**: 要求比值 > 0.6，确保有明确方向

### 4. Consecutive Bars
- 统计连续同向K线数量
- 正值 = 连续阳线
- 负值 = 连续阴线
- **用途**: 做多要求 >= 2根连续阳线，确认动能

### 5. Trend Structure Score
- 基于高低点结构评估趋势
- +1.0 = 强上升趋势（higher highs & higher lows）
- -1.0 = 强下降趋势（lower highs & lower lows）
- **用途**: 做多要求 > 0.3，确保趋势对齐

### 6. Two-Leg Pullback
- 识别Al Brooks的经典回调模式
- 上升趋势中：两次回调后的反弹
- **用途**: 可选择只在回调后入场

### 7. Breakout Bar Quality
- 综合评估突破K线的质量
- 考虑：实体大小、相对ATR、收盘位置
- 0.0-1.0评分
- **用途**: 要求 > 0.7，只做高质量突破

### 8. Trend Strength Composite
- 综合多个因子的趋势强度评分
- 包含：趋势结构、连续K线、动能、实体强度
- -1.0到+1.0
- **用途**: 做多要求 > 0.4，确保整体趋势强劲

## 使用方法

### 1. 准备数据

确保你有至少2年的历史数据：

```bash
# 数据应该在 data/ 目录下，格式为：
# data/crypto/BTCUSD_5min_eth_2023.parquet
# data/crypto/BTCUSD_5min_eth_2024.parquet
```

### 2. 运行单因子实验

测试每个因子的独立效果：

```python
from research.price_action.experiments import run_single_factor_experiments

results = run_single_factor_experiments()
```

这将运行7个实验，每个测试一个因子，并与baseline对比。

### 3. 运行组合因子实验

测试最佳因子的组合：

```python
from research.price_action.experiments import run_combined_factor_experiments

results = run_combined_factor_experiments()
```

### 4. 自定义实验

```python
from research.price_action.backtest import (
    BacktestExperiment,
    FactorFilterConfig,
    PriceActionBacktester,
)

# 创建自定义因子配置
custom_filter = FactorFilterConfig(
    ibs_long_min=0.65,           # 收盘位置要求
    range_atr_min=1.3,           # 动能要求
    body_ratio_min=0.6,          # 实体强度要求
    consecutive_long_min=2,      # 连续阳线要求
    trend_structure_long_min=0.3, # 趋势结构要求
)

# 创建实验
experiment = BacktestExperiment(
    name="Custom Filter",
    description="My custom factor combination",
    strategy_dsl=create_baseline_strategy(),
    factor_filter=custom_filter,
    market="crypto",
    symbol="BTCUSD",
    timeframe="5m",
    start_date="2023-01-01",
    end_date="2025-01-01",
)

# 运行回测
backtester = PriceActionBacktester(data_dir="data")
result = backtester.run_experiment(experiment)
```

### 5. 运行完整研究流程

```bash
cd backend
python -m research.price_action.experiments
```

这将：
1. 运行所有单因子实验
2. 运行所有组合因子实验
3. 生成对比报告
4. 保存结果到 `research/results/`

## 结果分析

实验结果包含以下关键指标：

### 交易频率
- **Total Trades**: 总交易次数
- **Change**: 相对baseline的变化
- 目标：不要过度减少交易机会（避免降到个位数）

### 胜率
- **Win Rate**: 获胜交易占比
- **Change**: 相对baseline的提升
- 目标：通过因子过滤提升胜率

### 收益
- **Total Return %**: 总收益率
- **Expectancy**: 每笔交易的平均盈亏
- 目标：提升单笔和整体表现

### 风险
- **Max Drawdown %**: 最大回撤
- 目标：通过过滤降低回撤

## 迭代优化流程

1. **Phase 1: 单因子测试**
   - 识别哪些因子最有效
   - 观察对交易频率的影响
   - 记录最佳阈值

2. **Phase 2: 因子组合**
   - 组合表现最好的2-3个因子
   - 测试保守vs激进的配置
   - 平衡胜率和交易频率

3. **Phase 3: 参数优化**
   - 微调因子阈值
   - 测试不同市场/时间周期
   - 验证稳定性

4. **Phase 4: 集成到DSL**
   - 将验证有效的因子集成到策略DSL
   - 创建可复用的因子库
   - 编写使用文档

## 架构设计

```
research/
├── __init__.py
└── price_action/
    ├── __init__.py
    ├── factors.py          # 因子计算逻辑
    ├── backtest.py         # 回测框架
    ├── experiments.py      # 实验定义和执行
    └── results/            # 实验结果（JSON）
        ├── single_factor_results.json
        └── combined_factor_results.json
```

### 复用现有模块

- `packages.domain.backtest.engine` - 事件驱动回测引擎
- `packages.domain.backtest.analytics` - 性能分析
- `packages.domain.market_data.data.data_loader` - 数据加载
- `packages.domain.strategy.parser` - 策略DSL解析

## 扩展指南

### 添加新因子

1. 在 `factors.py` 中添加静态方法：

```python
@staticmethod
def my_new_factor(df: pd.DataFrame) -> pd.Series:
    """Calculate my new factor."""
    # Your calculation here
    return result
```

2. 在 `calculate_all_factors` 中添加：

```python
result["pa_my_factor"] = PriceActionFactors.my_new_factor(df)
```

3. 在 `FactorFilterConfig` 中添加配置：

```python
@dataclass(frozen=True, slots=True)
class FactorFilterConfig:
    # ... existing fields ...
    my_factor_min: float | None = None
```

4. 在 `_apply_factor_filter` 中添加过滤逻辑：

```python
if filter_config.my_factor_min is not None:
    mask &= data["pa_my_factor"] >= filter_config.my_factor_min
```

### 测试不同策略

修改 `create_baseline_strategy()` 来测试不同的基础策略（RSI、MACD、布林带等）。

### 测试不同市场/周期

修改实验配置中的 `market`, `symbol`, `timeframe` 参数。

## 注意事项

1. **数据质量**: 确保使用高质量的历史数据
2. **过拟合**: 避免过度优化参数，保持因子逻辑简单
3. **样本外测试**: 在不同时间段验证因子有效性
4. **交易成本**: 考虑滑点和手续费的影响
5. **市场环境**: 因子在不同市场环境下可能表现不同

## 下一步

1. 运行完整的实验套件
2. 分析结果，识别最佳因子组合
3. 在不同市场和时间周期验证
4. 将有效因子集成到生产策略中
5. 持续监控和优化

## 参考资料

- Al Brooks: "Trading Price Action Trends"
- Al Brooks: "Trading Price Action Trading Ranges"
- Al Brooks: "Trading Price Action Reversals"
