"""Backtest framework for price action factor research."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from packages.domain.backtest.analytics import build_backtest_overview
from packages.domain.backtest.engine import EventDrivenBacktestEngine
from packages.domain.backtest.types import BacktestConfig
from packages.domain.market_data.data.data_loader import DataLoader
from packages.domain.strategy.models import ParsedStrategyDsl
from packages.domain.strategy.parser import build_parsed_strategy
from research.price_action.factors import PriceActionFactors


@dataclass(frozen=True, slots=True)
class FactorFilterConfig:
    """Configuration for factor-based trade filtering."""

    # IBS thresholds (for long entries, want IBS > threshold)
    ibs_long_min: float | None = None
    ibs_short_max: float | None = None

    # Bar range vs ATR (want strong bars, ratio > threshold)
    range_atr_min: float | None = None

    # Body to range ratio (want conviction, ratio > threshold)
    body_ratio_min: float | None = None

    # Consecutive bars (want momentum, count >= threshold)
    consecutive_long_min: int | None = None
    consecutive_short_max: int | None = None

    # Trend structure score (want aligned trend, score > threshold)
    trend_structure_long_min: float | None = None
    trend_structure_short_max: float | None = None

    # Two-leg pullback (require pattern: 1=bullish, -1=bearish, None=ignore)
    require_two_leg_pullback: bool = False

    # Breakout quality (want high quality, score > threshold)
    breakout_quality_min: float | None = None

    # Composite trend strength (want strong trend, score > threshold)
    trend_strength_long_min: float | None = None
    trend_strength_short_max: float | None = None

    # NEW ADVANCED FACTORS
    # Tail ratio (upper vs lower wick)
    tail_ratio_long_max: float | None = None  # Want lower wick > upper (negative values)
    tail_ratio_short_min: float | None = None  # Want upper wick > lower (positive values)

    # Gap quality (sustained gaps)
    gap_quality_long_min: float | None = None  # Want positive sustained gaps
    gap_quality_short_max: float | None = None  # Want negative sustained gaps

    # False breakout detection
    false_breakout_long_max: float | None = None  # Want bullish false breakout (negative)
    false_breakout_short_min: float | None = None  # Want bearish false breakout (positive)

    # Volatility percentile
    volatility_pct_min: float | None = None  # Want high volatility
    volatility_pct_max: float | None = None  # Want low volatility

    # Volume-price divergence
    volume_divergence_long_min: float | None = None  # Want bullish divergence (positive)
    volume_divergence_short_max: float | None = None  # Want bearish divergence (negative)

    # Bar follow-through
    follow_through_min: float | None = None  # Want strong continuation

    # Multi-timeframe alignment
    mtf_alignment_long_min: float | None = None  # Want bullish alignment (positive)
    mtf_alignment_short_max: float | None = None  # Want bearish alignment (negative)


@dataclass(frozen=True, slots=True)
class BacktestExperiment:
    """Single backtest experiment configuration."""

    name: str
    description: str
    strategy_dsl: dict[str, Any]
    factor_filter: FactorFilterConfig | None
    market: str
    symbol: str
    timeframe: str
    start_date: str
    end_date: str
    initial_capital: float = 100000.0
    commission_rate: float = 0.0001
    slippage_bps: float = 1.0


class PriceActionBacktester:
    """Backtest engine with price action factor filtering."""

    def __init__(self, data_dir: str | None = None) -> None:
        self.data_loader = DataLoader(data_dir=data_dir)

    def run_experiment(self, experiment: BacktestExperiment) -> dict[str, Any]:
        """Run a single backtest experiment with optional factor filtering.

        Args:
            experiment: Experiment configuration

        Returns:
            Backtest result dictionary with performance metrics
        """
        # Load market data
        data = self.data_loader.load(
            market=experiment.market,
            symbol=experiment.symbol,
            timeframe=experiment.timeframe,
            start_date=experiment.start_date,
            end_date=experiment.end_date,
        )

        # Calculate price action factors
        data_with_factors = PriceActionFactors.calculate_all_factors(data)

        # Parse strategy DSL (without modification)
        strategy = build_parsed_strategy(experiment.strategy_dsl)

        # Run backtest
        config = BacktestConfig(
            initial_capital=experiment.initial_capital,
            commission_rate=experiment.commission_rate,
            slippage_bps=experiment.slippage_bps,
            record_bar_events=False,
        )

        engine = EventDrivenBacktestEngine(
            strategy=strategy,
            data=data_with_factors,
            config=config,
        )

        result = engine.run()

        # Apply factor filter post-processing if configured
        if experiment.factor_filter is not None:
            result = self._filter_trades_by_factors(
                result,
                data_with_factors,
                experiment.factor_filter,
                experiment.initial_capital,
            )

        # Convert to dict and add metadata
        result_dict = self._result_to_dict(result)
        result_dict["experiment_name"] = experiment.name
        result_dict["experiment_description"] = experiment.description
        result_dict["market"] = experiment.market
        result_dict["symbol"] = experiment.symbol
        result_dict["timeframe"] = experiment.timeframe
        result_dict["factor_filter"] = (
            self._filter_config_to_dict(experiment.factor_filter)
            if experiment.factor_filter
            else None
        )

        return result_dict

    def run_ab_test(
        self,
        baseline: BacktestExperiment,
        variants: list[BacktestExperiment],
    ) -> dict[str, Any]:
        """Run A/B test comparing baseline vs multiple variants.

        Args:
            baseline: Baseline experiment (usually no factor filtering)
            variants: List of variant experiments with different factor configs

        Returns:
            Dictionary with comparison results
        """
        print(f"Running baseline: {baseline.name}")
        baseline_result = self.run_experiment(baseline)

        variant_results = []
        for variant in variants:
            print(f"Running variant: {variant.name}")
            variant_result = self.run_experiment(variant)
            variant_results.append(variant_result)

        # Compare results
        comparison = self._compare_results(baseline_result, variant_results)

        return {
            "baseline": baseline_result,
            "variants": variant_results,
            "comparison": comparison,
        }

    def _filter_trades_by_factors(
        self,
        result: Any,
        data: pd.DataFrame,
        filter_config: FactorFilterConfig,
        initial_capital: float,
    ) -> Any:
        """Filter trades based on price action factors at entry time.

        This is a post-processing step that removes trades where the entry bar
        didn't pass the factor filter criteria.

        Args:
            result: Backtest result object
            data: DataFrame with factor columns
            filter_config: Factor filter configuration

        Returns:
            Modified backtest result with filtered trades
        """
        from packages.domain.backtest.types import BacktestResult, BacktestSummary, BacktestTrade, EquityPoint
        from datetime import datetime

        # Create filter mask
        filter_mask = self._create_filter_mask(data, filter_config)

        # Filter trades
        filtered_trades = []
        for trade in result.trades:
            # Check if entry time passes the filter
            entry_time = trade.entry_time
            if entry_time in filter_mask.index and filter_mask.loc[entry_time]:
                filtered_trades.append(trade)

        # Recalculate summary statistics
        if len(filtered_trades) == 0:
            # No trades passed the filter
            return BacktestResult(
                config=result.config,
                summary=BacktestSummary(
                    total_trades=0,
                    winning_trades=0,
                    losing_trades=0,
                    win_rate=0.0,
                    total_pnl=0.0,
                    total_return_pct=0.0,
                    final_equity=initial_capital,
                    max_drawdown_pct=0.0,
                ),
                trades=tuple(),
                equity_curve=tuple([
                    EquityPoint(
                        timestamp=result.started_at,
                        equity=initial_capital,
                    )
                ]),
                returns=tuple(),
                events=tuple(),
                performance={},
                started_at=result.started_at,
                finished_at=result.finished_at,
            )

        # Recalculate metrics
        winning_trades = sum(1 for t in filtered_trades if t.pnl > 0)
        losing_trades = sum(1 for t in filtered_trades if t.pnl < 0)
        total_pnl = sum(t.pnl for t in filtered_trades)

        # Rebuild equity curve
        equity_curve = [EquityPoint(timestamp=result.started_at, equity=initial_capital)]
        current_equity = initial_capital
        for trade in sorted(filtered_trades, key=lambda t: t.exit_time):
            current_equity += trade.pnl
            equity_curve.append(EquityPoint(timestamp=trade.exit_time, equity=current_equity))

        # Calculate returns
        returns = [t.pnl_pct / 100 for t in filtered_trades]

        # Calculate max drawdown
        peak = initial_capital
        max_dd = 0.0
        for point in equity_curve:
            if point.equity > peak:
                peak = point.equity
            dd = (peak - point.equity) / peak * 100 if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd

        return BacktestResult(
            config=result.config,
            summary=BacktestSummary(
                total_trades=len(filtered_trades),
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=(winning_trades / len(filtered_trades) * 100) if filtered_trades else 0.0,
                total_pnl=total_pnl,
                total_return_pct=(total_pnl / initial_capital * 100) if initial_capital > 0 else 0.0,
                final_equity=current_equity,
                max_drawdown_pct=max_dd,
            ),
            trades=tuple(filtered_trades),
            equity_curve=tuple(equity_curve),
            returns=tuple(returns),
            events=tuple(),  # Don't filter events, just keep empty
            performance=result.performance,
            started_at=result.started_at,
            finished_at=result.finished_at,
        )

    def _create_filter_mask(
        self,
        data: pd.DataFrame,
        filter_config: FactorFilterConfig,
    ) -> pd.Series:
        """Create boolean mask for factor filtering.

        Args:
            data: DataFrame with factor columns
            filter_config: Factor filter configuration

        Returns:
            Boolean series indicating which bars pass the filter
        """
        mask = pd.Series(True, index=data.index)

        # IBS filters
        if filter_config.ibs_long_min is not None:
            mask &= data["pa_ibs"] >= filter_config.ibs_long_min
        if filter_config.ibs_short_max is not None:
            mask &= data["pa_ibs"] <= filter_config.ibs_short_max

        # Range vs ATR filter
        if filter_config.range_atr_min is not None:
            mask &= data["pa_range_atr"] >= filter_config.range_atr_min

        # Body ratio filter
        if filter_config.body_ratio_min is not None:
            mask &= data["pa_body_ratio"] >= filter_config.body_ratio_min

        # Consecutive bars filters
        if filter_config.consecutive_long_min is not None:
            mask &= data["pa_consecutive"] >= filter_config.consecutive_long_min
        if filter_config.consecutive_short_max is not None:
            mask &= data["pa_consecutive"] <= filter_config.consecutive_short_max

        # Trend structure filters
        if filter_config.trend_structure_long_min is not None:
            mask &= data["pa_trend_structure"] >= filter_config.trend_structure_long_min
        if filter_config.trend_structure_short_max is not None:
            mask &= data["pa_trend_structure"] <= filter_config.trend_structure_short_max

        # Two-leg pullback filter
        if filter_config.require_two_leg_pullback:
            mask &= data["pa_two_leg"] != 0

        # Breakout quality filter
        if filter_config.breakout_quality_min is not None:
            mask &= data["pa_breakout_quality"] >= filter_config.breakout_quality_min

        # Trend strength filters
        if filter_config.trend_strength_long_min is not None:
            mask &= data["pa_trend_strength"] >= filter_config.trend_strength_long_min
        if filter_config.trend_strength_short_max is not None:
            mask &= data["pa_trend_strength"] <= filter_config.trend_strength_short_max

        # NEW ADVANCED FACTORS
        # Tail ratio filters
        if filter_config.tail_ratio_long_max is not None:
            mask &= data["pa_tail_ratio"] <= filter_config.tail_ratio_long_max
        if filter_config.tail_ratio_short_min is not None:
            mask &= data["pa_tail_ratio"] >= filter_config.tail_ratio_short_min

        # Gap quality filters
        if filter_config.gap_quality_long_min is not None:
            mask &= data["pa_gap_quality"] >= filter_config.gap_quality_long_min
        if filter_config.gap_quality_short_max is not None:
            mask &= data["pa_gap_quality"] <= filter_config.gap_quality_short_max

        # False breakout filters
        if filter_config.false_breakout_long_max is not None:
            mask &= data["pa_false_breakout"] <= filter_config.false_breakout_long_max
        if filter_config.false_breakout_short_min is not None:
            mask &= data["pa_false_breakout"] >= filter_config.false_breakout_short_min

        # Volatility percentile filters
        if filter_config.volatility_pct_min is not None:
            mask &= data["pa_volatility_pct"] >= filter_config.volatility_pct_min
        if filter_config.volatility_pct_max is not None:
            mask &= data["pa_volatility_pct"] <= filter_config.volatility_pct_max

        # Volume divergence filters
        if filter_config.volume_divergence_long_min is not None:
            mask &= data["pa_volume_divergence"] >= filter_config.volume_divergence_long_min
        if filter_config.volume_divergence_short_max is not None:
            mask &= data["pa_volume_divergence"] <= filter_config.volume_divergence_short_max

        # Follow-through filter
        if filter_config.follow_through_min is not None:
            mask &= data["pa_follow_through"] >= filter_config.follow_through_min

        # Multi-timeframe alignment filters
        if filter_config.mtf_alignment_long_min is not None:
            mask &= data["pa_mtf_alignment"] >= filter_config.mtf_alignment_long_min
        if filter_config.mtf_alignment_short_max is not None:
            mask &= data["pa_mtf_alignment"] <= filter_config.mtf_alignment_short_max

        return mask

    def _apply_factor_filter_to_data(
        self,
        data: pd.DataFrame,
        filter_config: FactorFilterConfig,
    ) -> pd.DataFrame:
        """Apply factor-based filtering by adding pa_filter column.

        Args:
            data: DataFrame with OHLC and factor columns
            filter_config: Factor filter configuration

        Returns:
            DataFrame with pa_filter column added
        """
        mask = self._create_filter_mask(data, filter_config)
        filtered_data = data.copy()
        filtered_data["pa_filter"] = mask.astype(float)
        return filtered_data

    def _apply_factor_filter(
        self,
        data: pd.DataFrame,
        filter_config: FactorFilterConfig,
    ) -> pd.DataFrame:
        """Legacy method - kept for backwards compatibility.

        Use _apply_factor_filter_to_data instead.
        """
        return self._apply_factor_filter_to_data(data, filter_config)

    def _inject_filter_into_strategy(self, strategy_dsl: dict) -> dict:
        """Inject pa_filter condition into strategy entry logic.

        The pa_filter column must already exist in the data DataFrame.
        This method modifies the strategy DSL to check pa_filter > 0.5
        in addition to the original entry conditions.

        Args:
            strategy_dsl: Original strategy DSL

        Returns:
            Modified strategy DSL with pa_filter added to entry conditions
        """
        import copy
        modified_dsl = copy.deepcopy(strategy_dsl)

        # Add pa_filter as a "passthrough" factor that just references the data column
        if "factors" not in modified_dsl:
            modified_dsl["factors"] = {}

        # Define pa_filter as an identity indicator (just passes through the column value)
        modified_dsl["factors"]["pa_filter"] = {
            "type": "identity",
            "params": {
                "source": "pa_filter"
            }
        }

        # Inject pa_filter into long entry condition
        if "trade" in modified_dsl and "long" in modified_dsl["trade"]:
            long_entry = modified_dsl["trade"]["long"]["entry"]
            original_condition = long_entry["condition"]

            # Wrap original condition with AND pa_filter > 0.5
            long_entry["condition"] = {
                "and": [
                    original_condition,
                    {
                        "compare": {
                            "a": {"ref": "pa_filter"},
                            "op": ">",
                            "b": {"value": 0.5},
                        }
                    },
                ]
            }

        # Inject pa_filter into short entry condition if exists
        if "trade" in modified_dsl and "short" in modified_dsl["trade"]:
            short_entry = modified_dsl["trade"]["short"]["entry"]
            original_condition = short_entry["condition"]

            short_entry["condition"] = {
                "and": [
                    original_condition,
                    {
                        "compare": {
                            "a": {"ref": "pa_filter"},
                            "op": ">",
                            "b": {"value": 0.5},
                        }
                    },
                ]
            }

        return modified_dsl

    def _result_to_dict(self, result: Any) -> dict[str, Any]:
        """Convert BacktestResult to dictionary."""
        # Calculate expectancy
        expectancy = 0.0
        if result.summary.total_trades > 0:
            expectancy = result.summary.total_pnl / result.summary.total_trades

        return {
            "summary": {
                "total_trades": result.summary.total_trades,
                "winning_trades": result.summary.winning_trades,
                "losing_trades": result.summary.losing_trades,
                "win_rate": result.summary.win_rate,
                "total_pnl": result.summary.total_pnl,
                "total_return": result.summary.total_return_pct,  # Rename for consistency
                "final_equity": result.summary.final_equity,
                "max_drawdown_pct": result.summary.max_drawdown_pct,
                "expectancy": expectancy,
            },
            "trades": [
                {
                    "side": trade.side.value,
                    "entry_time": trade.entry_time.isoformat(),
                    "exit_time": trade.exit_time.isoformat(),
                    "entry_price": trade.entry_price,
                    "exit_price": trade.exit_price,
                    "quantity": trade.quantity,
                    "bars_held": trade.bars_held,
                    "exit_reason": trade.exit_reason,
                    "pnl": trade.pnl,
                    "pnl_pct": trade.pnl_pct,
                    "commission": trade.commission,
                }
                for trade in result.trades
            ],
            "equity_curve": [
                {
                    "timestamp": point.timestamp.isoformat(),
                    "equity": point.equity,
                }
                for point in result.equity_curve
            ],
            "returns": list(result.returns),
            "performance": result.performance,
            "started_at": result.started_at.isoformat(),
            "finished_at": result.finished_at.isoformat(),
        }

    def _filter_config_to_dict(self, config: FactorFilterConfig) -> dict[str, Any]:
        """Convert FactorFilterConfig to dictionary."""
        return {
            "ibs_long_min": config.ibs_long_min,
            "ibs_short_max": config.ibs_short_max,
            "range_atr_min": config.range_atr_min,
            "body_ratio_min": config.body_ratio_min,
            "consecutive_long_min": config.consecutive_long_min,
            "consecutive_short_max": config.consecutive_short_max,
            "trend_structure_long_min": config.trend_structure_long_min,
            "trend_structure_short_max": config.trend_structure_short_max,
            "require_two_leg_pullback": config.require_two_leg_pullback,
            "breakout_quality_min": config.breakout_quality_min,
            "trend_strength_long_min": config.trend_strength_long_min,
            "trend_strength_short_max": config.trend_strength_short_max,
        }

    def _compare_results(
        self,
        baseline: dict[str, Any],
        variants: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Compare variant results against baseline.

        Args:
            baseline: Baseline backtest result
            variants: List of variant backtest results

        Returns:
            List of comparison dictionaries
        """
        comparisons = []

        baseline_summary = baseline["summary"]

        for variant in variants:
            variant_summary = variant["summary"]

            comparison = {
                "variant_name": variant["experiment_name"],
                "variant_description": variant["experiment_description"],
                "factor_filter": variant["factor_filter"],
                "metrics": {
                    "total_trades": {
                        "baseline": baseline_summary["total_trades"],
                        "variant": variant_summary["total_trades"],
                        "change": variant_summary["total_trades"] - baseline_summary["total_trades"],
                        "change_pct": (
                            (variant_summary["total_trades"] / baseline_summary["total_trades"] - 1) * 100
                            if baseline_summary["total_trades"] > 0
                            else 0
                        ),
                    },
                    "win_rate": {
                        "baseline": baseline_summary["win_rate"],
                        "variant": variant_summary["win_rate"],
                        "change": variant_summary["win_rate"] - baseline_summary["win_rate"],
                    },
                    "total_return_pct": {
                        "baseline": baseline_summary["total_return_pct"],
                        "variant": variant_summary["total_return_pct"],
                        "change": variant_summary["total_return_pct"] - baseline_summary["total_return_pct"],
                    },
                    "max_drawdown_pct": {
                        "baseline": baseline_summary["max_drawdown_pct"],
                        "variant": variant_summary["max_drawdown_pct"],
                        "change": variant_summary["max_drawdown_pct"] - baseline_summary["max_drawdown_pct"],
                    },
                },
            }

            # Calculate expectancy (avg PnL per trade)
            baseline_trades = baseline["trades"]
            variant_trades = variant["trades"]

            baseline_expectancy = (
                sum(t["pnl"] for t in baseline_trades) / len(baseline_trades)
                if baseline_trades
                else 0
            )
            variant_expectancy = (
                sum(t["pnl"] for t in variant_trades) / len(variant_trades)
                if variant_trades
                else 0
            )

            comparison["metrics"]["expectancy"] = {
                "baseline": baseline_expectancy,
                "variant": variant_expectancy,
                "change": variant_expectancy - baseline_expectancy,
                "change_pct": (
                    (variant_expectancy / baseline_expectancy - 1) * 100
                    if baseline_expectancy != 0
                    else 0
                ),
            }

            comparisons.append(comparison)

        return comparisons
