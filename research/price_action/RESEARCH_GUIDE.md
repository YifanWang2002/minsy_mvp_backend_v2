# Price Action Research Guide - Step-by-Step Instructions

## Overview

This guide provides systematic instructions for conducting price action factor research. The framework tests whether Al Brooks-style price action factors can enhance multiple trend-following strategies, with rigorous in-sample/out-of-sample validation.

## Research Methodology

### Core Principles

1. **Multiple Baselines**: Test 4 different trend-following strategies to ensure findings generalize
2. **In-Sample/Out-of-Sample**: Use 2023 for factor discovery, 2024 for validation
3. **Systematic Testing**: Test individual factors first, then combinations
4. **Trade-off Analysis**: Balance win rate improvement vs trade frequency reduction
5. **Robustness**: Only trust factors that work across multiple strategies and time periods

### Baseline Strategies

All strategies follow similar principles (trend-following, momentum-based):

1. **EMA Crossover**: 20/50 EMA crossover with ATR-based stops
2. **MACD Trend**: MACD crossover with 200 EMA trend filter
3. **Donchian Breakout**: 20-period high breakout with 100 EMA filter
4. **ADX Strong Trend**: ADX > 25 with directional movement confirmation

### Price Action Factors

8 quantifiable factors based on Al Brooks methodology:

1. **IBS (Internal Bar Strength)**: Close position within bar range
2. **Range/ATR**: Bar size relative to recent volatility
3. **Body Ratio**: Body size relative to total range
4. **Consecutive Bars**: Count of consecutive same-direction bars
5. **Trend Structure**: Higher highs/lower lows pattern scoring
6. **Two-Leg Pullback**: Classic pullback pattern detection
7. **Breakout Quality**: Composite quality score for breakout bars
8. **Trend Strength**: Composite trend strength combining multiple factors

## Step-by-Step Research Process

### Phase 1: Setup and Validation (5 minutes)

```bash
cd backend

# 1. Verify data availability
ls -lh data/crypto/BTCUSD_5min_eth_*.parquet

# 2. Run framework validation test
uv run python research/price_action/test_framework.py

# Expected output: All factors calculate successfully
```

### Phase 2: Run Complete Research (30-60 minutes)

```bash
# Run full research with all strategies and out-of-sample validation
uv run python research/price_action/research_runner.py
```

This will:
- Test 4 baseline strategies
- Apply 7 factor variants to each strategy
- Run in-sample (2023) and out-of-sample (2024) for each
- Total: 4 strategies × 8 variants × 2 periods = 64 backtests
- Generate comprehensive comparison reports

### Phase 3: Analyze Results (15 minutes)

```bash
# View complete results
cat research/results/complete_research_results.json | jq '.strategies | keys'

# Extract key metrics for a specific strategy
cat research/results/complete_research_results.json | jq '.strategies.ema_crossover'
```

**Key Questions to Answer:**

1. **Which factors improve performance consistently?**
   - Look for factors that improve return AND win rate across multiple strategies
   - Check if improvement holds out-of-sample

2. **What's the trade-off?**
   - How much do trade counts decrease?
   - Is the improvement per trade worth the reduced frequency?

3. **Which strategies benefit most?**
   - Some strategies may benefit more from certain factors
   - Identify strategy-factor synergies

4. **Do combinations work better?**
   - Conservative vs moderate multi-filters
   - Diminishing returns from too many filters?

### Phase 4: Iterate and Refine (30 minutes)

Based on Phase 3 findings, create custom experiments:

```python
# Example: Test refined thresholds
from research.price_action.backtest import (
    BacktestExperiment,
    FactorFilterConfig,
    PriceActionBacktester,
)
from research.price_action.strategies import create_ema_crossover_strategy

backtester = PriceActionBacktester(data_dir="data")

# If IBS and Range/ATR showed promise, test different thresholds
custom_filter = FactorFilterConfig(
    ibs_long_min=0.55,  # Slightly looser than 0.6
    range_atr_min=1.1,  # Slightly looser than 1.2
)

experiment = BacktestExperiment(
    name="Custom Refined",
    description="Refined IBS + Range/ATR thresholds",
    strategy_dsl=create_ema_crossover_strategy(),
    factor_filter=custom_filter,
    market="crypto",
    symbol="BTCUSD",
    timeframe="5m",
    start_date="2023-01-01",
    end_date="2024-12-31",
    initial_capital=100000.0,
)

result = backtester.run_experiment(experiment)
print(f"Total Return: {result['summary']['total_return_pct']:.2f}%")
print(f"Win Rate: {result['summary']['win_rate']:.2f}%")
print(f"Total Trades: {result['summary']['total_trades']}")
```

### Phase 5: Document Findings (15 minutes)

Create a findings document:

```bash
# Create findings summary
cat > research/results/findings_summary.md << 'EOF'
# Price Action Research Findings

## Date: [Current Date]

## Executive Summary
[Brief overview of key findings]

## Methodology
- Baseline strategies: 4 trend-following strategies
- Time periods: 2023 (in-sample), 2024 (out-of-sample)
- Factors tested: 8 individual + 2 combinations

## Key Findings

### 1. Most Effective Factors
[List factors that consistently improved performance]

### 2. Strategy-Specific Insights
[Which factors work best with which strategies]

### 3. Out-of-Sample Validation
[Which improvements held up in 2024]

### 4. Trade-offs
[Trade frequency vs performance improvements]

## Recommendations

### For Production Integration
[Which factors to integrate into live strategies]

### For Further Research
[Areas needing more investigation]

## Detailed Results
[Link to JSON files with complete data]
EOF
```

## Expected Outcomes

### Success Criteria

A successful research iteration should identify:

1. **At least 1-2 factors** that consistently improve win rate by 3-5% across multiple strategies
2. **Validated improvements** that hold out-of-sample (2024 performance similar to 2023)
3. **Acceptable trade-offs** where trade count doesn't drop below 50% of baseline
4. **Clear integration path** for incorporating factors into production strategies

### Warning Signs

Be cautious if you see:

1. **Overfitting**: Huge in-sample improvement that disappears out-of-sample
2. **Over-filtering**: Trade count drops to single digits per year
3. **Inconsistent results**: Factor works for one strategy but hurts others
4. **Unstable metrics**: Large variance in performance across time periods

## Iteration Guidelines

### When to Iterate

Iterate if:
- Initial factors show promise but need threshold tuning
- Want to test new factor combinations
- Need to test on different timeframes (1min, 15min, 1h)
- Want to test on different symbols (ETHUSD, etc.)

### How to Iterate

1. **Threshold Tuning**: Adjust factor thresholds (e.g., IBS 0.55 vs 0.6 vs 0.65)
2. **Combination Testing**: Try different factor combinations
3. **Timeframe Testing**: Test on 1min or 15min data
4. **Symbol Testing**: Validate on ETHUSD or other crypto pairs
5. **New Factors**: Implement additional Al Brooks concepts

### Iteration Template

```python
# research/price_action/custom_iteration.py
from research.price_action.backtest import (
    BacktestExperiment,
    FactorFilterConfig,
    PriceActionBacktester,
)
from research.price_action.strategies import get_all_baseline_strategies

def run_custom_iteration():
    """Run custom iteration based on initial findings."""
    backtester = PriceActionBacktester(data_dir="data")

    # Define your custom filter based on Phase 3 findings
    custom_filter = FactorFilterConfig(
        # Add your refined parameters here
        ibs_long_min=0.58,
        range_atr_min=1.15,
        trend_structure_long_min=0.25,
    )

    strategies = get_all_baseline_strategies()
    results = {}

    for name, dsl in strategies.items():
        print(f"Testing {name}...")

        # In-sample
        is_exp = BacktestExperiment(
            name=f"{name}_custom_is",
            description="Custom refined filter",
            strategy_dsl=dsl,
            factor_filter=custom_filter,
            market="crypto",
            symbol="BTCUSD",
            timeframe="5m",
            start_date="2023-01-01",
            end_date="2023-12-31",
            initial_capital=100000.0,
        )
        is_result = backtester.run_experiment(is_exp)

        # Out-of-sample
        oos_exp = BacktestExperiment(
            name=f"{name}_custom_oos",
            description="Custom refined filter",
            strategy_dsl=dsl,
            factor_filter=custom_filter,
            market="crypto",
            symbol="BTCUSD",
            timeframe="5m",
            start_date="2024-01-01",
            end_date="2024-12-31",
            initial_capital=100000.0,
        )
        oos_result = backtester.run_experiment(oos_exp)

        results[name] = {
            "insample": is_result,
            "outofsample": oos_result,
        }

        print(f"  IS: {is_result['summary']['total_return_pct']:.2f}% | OOS: {oos_result['summary']['total_return_pct']:.2f}%")

    return results

if __name__ == "__main__":
    results = run_custom_iteration()
```

## Integration into Production

Once you've identified validated factors:

### Step 1: Create Factor Library

```python
# packages/domain/strategy/factors/price_action.py
"""Validated price action factors for production use."""

from dataclasses import dataclass

@dataclass(frozen=True)
class PriceActionFilter:
    """Production-ready price action filter configuration."""

    ibs_min: float = 0.6
    range_atr_min: float = 1.2
    trend_structure_min: float = 0.3

    def to_dsl(self) -> dict:
        """Convert to strategy DSL format."""
        return {
            "type": "price_action_filter",
            "params": {
                "ibs_min": self.ibs_min,
                "range_atr_min": self.range_atr_min,
                "trend_structure_min": self.trend_structure_min,
            }
        }
```

### Step 2: Update Strategy DSL Schema

Add price action filter support to `packages/domain/strategy/assets/strategy_dsl_schema.json`

### Step 3: Integrate into Backtest Engine

Update `packages/domain/backtest/engine.py` to apply price action filters during signal evaluation

### Step 4: Document for Users

Create user-facing documentation explaining:
- What price action filters do
- When to use them
- How to configure them
- Expected impact on performance

## Troubleshooting

### Common Issues

**Issue**: "ModuleNotFoundError: No module named 'redis'"
**Solution**: Use `uv run python` instead of plain `python`

**Issue**: "FileNotFoundError: data/crypto/BTCUSD_5min_eth_2023.parquet"
**Solution**: Verify data files exist with `ls data/crypto/`

**Issue**: Backtest runs very slowly
**Solution**:
- Reduce date range for initial testing
- Use fewer strategies/variants
- Check if data is properly indexed

**Issue**: All factors show negative results
**Solution**:
- Check if baseline strategy itself is profitable
- Verify factor calculations are correct
- Consider that factors may not help this particular strategy

## Next Steps After Research

1. **Validate on More Data**: Test on 2022, 2025 data
2. **Test Other Symbols**: ETHUSD, SOLUSD, etc.
3. **Test Other Timeframes**: 1min, 15min, 1h
4. **Combine with Other Techniques**: Test alongside other filters (volatility, volume, etc.)
5. **Live Paper Trading**: Test validated factors in paper trading before production
6. **Monitor Performance**: Track live performance vs backtest expectations

## Research Log Template

Keep a research log to track iterations:

```markdown
# Research Log

## Iteration 1 - [Date]
- **Goal**: Initial factor testing across 4 strategies
- **Configuration**: Standard thresholds (IBS 0.6, Range/ATR 1.2, etc.)
- **Results**: [Summary]
- **Key Findings**: [Bullet points]
- **Next Steps**: [What to try next]

## Iteration 2 - [Date]
- **Goal**: Refine IBS and Range/ATR thresholds
- **Configuration**: IBS 0.55, Range/ATR 1.15
- **Results**: [Summary]
- **Key Findings**: [Bullet points]
- **Next Steps**: [What to try next]
```

## Conclusion

This research framework provides a systematic approach to discovering and validating price action factors. The key is to:

1. Test rigorously with multiple strategies and time periods
2. Validate out-of-sample before trusting results
3. Balance performance improvement with trade frequency
4. Iterate based on findings
5. Only integrate factors that show consistent, validated improvements

Good luck with your research!
