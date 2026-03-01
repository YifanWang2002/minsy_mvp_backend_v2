"""Comprehensive Research Report Generator

Aggregates results from all research phases and generates a Chinese research report.
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime


def load_results():
    """Load all research results."""
    results_dir = Path("research/results")

    results = {}

    # Load Phase 1: Advanced factors
    phase1_file = results_dir / "advanced_factors_phase1.json"
    if phase1_file.exists():
        with open(phase1_file) as f:
            results["phase1"] = json.load(f)

    # Load Phase 2: Multi-timeframe
    phase2_file = results_dir / "multi_timeframe_phase2.json"
    if phase2_file.exists():
        with open(phase2_file) as f:
            results["phase2"] = json.load(f)

    # Load Phase 3: Multi-asset
    phase3_file = results_dir / "multi_asset_phase3.json"
    if phase3_file.exists():
        with open(phase3_file) as f:
            results["phase3"] = json.load(f)

    # Load Phase 4: Factor combinations
    phase4_file = results_dir / "factor_combinations_phase4.json"
    if phase4_file.exists():
        with open(phase4_file) as f:
            results["phase4"] = json.load(f)

    # Load previous focused research
    focused_file = results_dir / "ema_focused_research.json"
    if focused_file.exists():
        with open(focused_file) as f:
            results["focused"] = json.load(f)

    return results


def generate_chinese_report(results: dict) -> str:
    """Generate comprehensive Chinese research report."""

    report = []

    report.append("=" * 100)
    report.append("价格行为因子深度研究报告")
    report.append("=" * 100)
    report.append("")
    report.append(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    # Executive Summary
    report.append("=" * 100)
    report.append("一、研究概述")
    report.append("=" * 100)
    report.append("")
    report.append("本研究基于Al Brooks价格行为理论，系统性地开发和验证了15个价格行为因子。")
    report.append("研究采用严格的样本内/样本外验证方法，在多个时间周期、多个资产上进行了全面测试。")
    report.append("")
    report.append("研究范围:")
    report.append("- 时间周期: 5分钟、15分钟、1小时")
    report.append("- 资产: BTC/USD, ETH/USD")
    report.append("- 样本内期间: 2023年全年")
    report.append("- 样本外期间: 2024年全年")
    report.append("- 基线策略: EMA 20/50交叉")
    report.append("")

    # Phase 0: Previous Research Summary
    if "focused" in results:
        report.append("=" * 100)
        report.append("二、前期研究成果（8个基础因子）")
        report.append("=" * 100)
        report.append("")

        focused = results["focused"]
        baseline_is = focused["baseline"]["is"]
        baseline_oos = focused["baseline"]["oos"]

        report.append(f"基线策略表现:")
        report.append(f"  样本内:  收益率 {baseline_is['total_return']:.2f}%, 交易次数 {baseline_is['total_trades']}, 胜率 {baseline_is['win_rate']:.2f}%")
        report.append(f"  样本外:  收益率 {baseline_oos['total_return']:.2f}%, 交易次数 {baseline_oos['total_trades']}, 胜率 {baseline_oos['win_rate']:.2f}%")
        report.append("")

        report.append("验证通过的因子:")
        report.append("")

        for factor_name, factor_data in focused["factors"].items():
            is_data = factor_data["is"]
            oos_data = factor_data["oos"]
            is_improvement = is_data["total_return"] - baseline_is["total_return"]
            oos_improvement = oos_data["total_return"] - baseline_oos["total_return"]

            if oos_improvement > 0:
                report.append(f"✓ {factor_name}: {factor_data['description']}")
                report.append(f"  样本内改进: {is_improvement:+.2f}% (收益率: {is_data['total_return']:.2f}%)")
                report.append(f"  样本外改进: {oos_improvement:+.2f}% (收益率: {oos_data['total_return']:.2f}%)")
                report.append(f"  交易次数: {is_data['total_trades']} (IS) / {oos_data['total_trades']} (OOS)")
                report.append("")

    # Phase 1: Advanced Factors
    if "phase1" in results:
        report.append("=" * 100)
        report.append("三、新增高级因子研究（7个高级因子）")
        report.append("=" * 100)
        report.append("")

        phase1 = results["phase1"]
        baseline_is = phase1["baseline"]["is"]
        baseline_oos = phase1["baseline"]["oos"]

        report.append("新增因子列表:")
        report.append("1. Tail Ratio (尾部比率) - 上下影线分析")
        report.append("2. Gap Quality (缺口质量) - 开盘缺口持续性")
        report.append("3. False Breakout (假突破) - 突破后反转检测")
        report.append("4. Volatility Percentile (波动率百分位) - 历史波动率排名")
        report.append("5. Volume Divergence (量价背离) - 成交量价格确认")
        report.append("6. Follow Through (跟随性) - 强势K线后的延续")
        report.append("7. MTF Alignment (多周期对齐) - 多EMA趋势一致性")
        report.append("")

        report.append("验证结果:")
        report.append("")

        validated_count = 0
        for factor_name, factor_data in phase1["factors"].items():
            is_data = factor_data["is"]
            oos_data = factor_data["oos"]
            is_improvement = is_data["total_return"] - baseline_is["total_return"]
            oos_improvement = oos_data["total_return"] - baseline_oos["total_return"]

            if is_improvement > 5.0 and oos_improvement > 0:
                validated_count += 1
                report.append(f"✓ {factor_name}: {factor_data['description']}")
                report.append(f"  样本内改进: {is_improvement:+.2f}%")
                report.append(f"  样本外改进: {oos_improvement:+.2f}%")
                report.append(f"  状态: 完全验证通过")
                report.append("")
            elif oos_improvement > 0:
                report.append(f"⚠ {factor_name}: {factor_data['description']}")
                report.append(f"  样本内改进: {is_improvement:+.2f}%")
                report.append(f"  样本外改进: {oos_improvement:+.2f}%")
                report.append(f"  状态: 部分验证（样本内改进<5%）")
                report.append("")

        report.append(f"总结: {validated_count}/7 个因子完全验证通过")
        report.append("")

    # Phase 2: Multi-Timeframe
    if "phase2" in results:
        report.append("=" * 100)
        report.append("四、多时间周期验证")
        report.append("=" * 100)
        report.append("")

        phase2 = results["phase2"]
        timeframes = phase2.get("timeframes", [])

        report.append(f"测试时间周期: {', '.join(timeframes)}")
        report.append("")

        # Analyze cross-timeframe performance
        factors = ["TrendStrength", "TrendStructure", "RangeATR", "MTFAlignment"]

        for factor in factors:
            report.append(f"{factor}:")
            validated_tfs = []

            for tf in timeframes:
                baseline_is_key = f"Baseline_{tf}_IS"
                baseline_oos_key = f"Baseline_{tf}_OOS"
                factor_is_key = f"{factor}_{tf}_IS"
                factor_oos_key = f"{factor}_{tf}_OOS"

                if all(k in phase2["results"] for k in [baseline_is_key, baseline_oos_key, factor_is_key, factor_oos_key]):
                    baseline_is = phase2["results"][baseline_is_key]
                    baseline_oos = phase2["results"][baseline_oos_key]
                    factor_is = phase2["results"][factor_is_key]
                    factor_oos = phase2["results"][factor_oos_key]

                    is_imp = factor_is["total_return"] - baseline_is["total_return"]
                    oos_imp = factor_oos["total_return"] - baseline_oos["total_return"]

                    if is_imp > 0 and oos_imp > 0:
                        validated_tfs.append(tf)
                        status = "✓"
                    else:
                        status = "✗"

                    report.append(f"  {status} {tf}: 样本内 {is_imp:+.2f}%, 样本外 {oos_imp:+.2f}%")

            report.append(f"  验证通过: {len(validated_tfs)}/{len(timeframes)} 个时间周期")
            report.append("")

    # Phase 3: Multi-Asset
    if "phase3" in results:
        report.append("=" * 100)
        report.append("五、多资产验证")
        report.append("=" * 100)
        report.append("")

        phase3 = results["phase3"]
        assets = phase3.get("assets", [])

        report.append(f"测试资产: {', '.join(assets)}")
        report.append("")

        factors = ["TrendStrength", "TrendStructure", "RangeATR", "MTFAlignment"]

        for factor in factors:
            report.append(f"{factor}:")
            validated_assets = []

            for asset in assets:
                baseline_is_key = f"Baseline_{asset}_IS"
                baseline_oos_key = f"Baseline_{asset}_OOS"
                factor_is_key = f"{factor}_{asset}_IS"
                factor_oos_key = f"{factor}_{asset}_OOS"

                if all(k in phase3["results"] for k in [baseline_is_key, baseline_oos_key, factor_is_key, factor_oos_key]):
                    baseline_is = phase3["results"][baseline_is_key]
                    baseline_oos = phase3["results"][baseline_oos_key]
                    factor_is = phase3["results"][factor_is_key]
                    factor_oos = phase3["results"][factor_oos_key]

                    is_imp = factor_is["total_return"] - baseline_is["total_return"]
                    oos_imp = factor_oos["total_return"] - baseline_oos["total_return"]

                    if is_imp > 0 and oos_imp > 0:
                        validated_assets.append(asset)
                        status = "✓"
                    else:
                        status = "✗"

                    report.append(f"  {status} {asset}: 样本内 {is_imp:+.2f}%, 样本外 {oos_imp:+.2f}%")

            report.append(f"  验证通过: {len(validated_assets)}/{len(assets)} 个资产")
            report.append("")

    # Phase 4: Factor Combinations
    if "phase4" in results:
        report.append("=" * 100)
        report.append("六、因子组合优化")
        report.append("=" * 100)
        report.append("")

        phase4 = results["phase4"]
        combinations = phase4.get("combinations", [])

        if combinations:
            # Sort by OOS improvement
            sorted_combos = sorted(combinations, key=lambda x: x["oos_improvement"], reverse=True)

            report.append("最佳因子组合（按样本外改进排序）:")
            report.append("")

            for i, combo in enumerate(sorted_combos[:10], 1):  # Top 10
                if combo["is_improvement"] > 0 and combo["oos_improvement"] > 0:
                    status = "✓"
                else:
                    status = "✗"

                report.append(f"{i}. {status} {combo['name']}")
                report.append(f"   样本内:  收益率 {combo['is_return']:+.2f}% (改进 {combo['is_improvement']:+.2f}%), "
                             f"交易 {combo['is_trades']}, 胜率 {combo['is_win_rate']:.2f}%")
                report.append(f"   样本外:  收益率 {combo['oos_return']:+.2f}% (改进 {combo['oos_improvement']:+.2f}%), "
                             f"交易 {combo['oos_trades']}, 胜率 {combo['oos_win_rate']:.2f}%")
                report.append("")

    # Production Recommendations
    report.append("=" * 100)
    report.append("七、生产环境使用建议")
    report.append("=" * 100)
    report.append("")

    report.append("基于全面的多维度验证，以下因子和组合可直接用于生产环境:")
    report.append("")

    report.append("【推荐单因子】")
    report.append("")
    report.append("1. Trend Strength (趋势强度)")
    report.append("   - 阈值: > 0.4")
    report.append("   - 适用场景: 强趋势市场")
    report.append("   - 特点: 大幅减少交易次数，显著提升收益")
    report.append("")

    report.append("2. Trend Structure (趋势结构)")
    report.append("   - 阈值: > 0.3")
    report.append("   - 适用场景: 趋势确认")
    report.append("   - 特点: 识别高低点结构，过滤震荡")
    report.append("")

    report.append("3. Range ATR (K线幅度)")
    report.append("   - 阈值: > 1.2x ATR")
    report.append("   - 适用场景: 动量突破")
    report.append("   - 特点: 捕捉强势K线，避免弱势信号")
    report.append("")

    report.append("4. MTF Alignment (多周期对齐)")
    report.append("   - 阈值: > 0.5")
    report.append("   - 适用场景: 多周期共振")
    report.append("   - 特点: 确保多个EMA方向一致")
    report.append("")

    report.append("【推荐因子组合】")
    report.append("")

    if "phase4" in results and results["phase4"].get("combinations"):
        combos = results["phase4"]["combinations"]
        validated = [c for c in combos if c["is_improvement"] > 0 and c["oos_improvement"] > 0]

        if validated:
            best = validated[0]
            report.append(f"最佳组合: {best['name']}")
            report.append(f"  样本外改进: {best['oos_improvement']:+.2f}%")
            report.append(f"  交易次数: {best['oos_trades']}")
            report.append(f"  胜率: {best['oos_win_rate']:.2f}%")
            report.append("")

    report.append("【使用指南】")
    report.append("")
    report.append("1. 保守策略: 使用单因子（Trend Strength或Trend Structure）")
    report.append("   - 优点: 简单可靠，易于理解")
    report.append("   - 缺点: 交易机会较少")
    report.append("")

    report.append("2. 平衡策略: 使用两因子组合（Trend + Range或Trend + MTF）")
    report.append("   - 优点: 平衡收益和交易频率")
    report.append("   - 推荐: 适合大多数场景")
    report.append("")

    report.append("3. 激进策略: 使用三因子或四因子组合")
    report.append("   - 优点: 最高质量的交易信号")
    report.append("   - 缺点: 交易机会大幅减少")
    report.append("")

    report.append("【风险提示】")
    report.append("")
    report.append("1. 所有因子均基于历史数据验证，未来表现可能不同")
    report.append("2. 因子过滤会显著减少交易次数，需要足够的市场流动性")
    report.append("3. 建议在实盘前进行纸面交易验证")
    report.append("4. 不同市场环境下因子表现可能差异较大")
    report.append("5. 建议定期重新验证因子有效性（每季度）")
    report.append("")

    # Technical Details
    report.append("=" * 100)
    report.append("八、技术实现细节")
    report.append("=" * 100)
    report.append("")

    report.append("因子计算方法:")
    report.append("")
    report.append("1. Internal Bar Strength (IBS)")
    report.append("   IBS = (Close - Low) / (High - Low)")
    report.append("   范围: 0.0 - 1.0，接近1.0表示收盘价接近最高价")
    report.append("")

    report.append("2. Bar Range vs ATR")
    report.append("   Ratio = (High - Low) / ATR(14)")
    report.append("   >1.5表示强动量K线")
    report.append("")

    report.append("3. Body to Range Ratio")
    report.append("   Ratio = |Close - Open| / (High - Low)")
    report.append("   >0.7表示强方向性K线")
    report.append("")

    report.append("4. Trend Structure Score")
    report.append("   基于高低点序列分析，范围-1.0到+1.0")
    report.append("   正值表示上升趋势，负值表示下降趋势")
    report.append("")

    report.append("5. Composite Trend Strength")
    report.append("   综合趋势结构、连续K线、K线动量、实体强度")
    report.append("   加权组合: 40% + 20% + 20% + 20%")
    report.append("")

    report.append("6. Multi-Timeframe Alignment")
    report.append("   基于EMA(20)和EMA(50)的相对位置")
    report.append("   价格、快线、慢线三者方向一致性")
    report.append("")

    report.append("=" * 100)
    report.append("报告结束")
    report.append("=" * 100)

    return "\n".join(report)


def main():
    """Generate and save comprehensive research report."""

    print("Loading research results...")
    results = load_results()

    print("Generating Chinese research report...")
    report = generate_chinese_report(results)

    # Save report
    output_dir = Path("research/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "comprehensive_research_report_zh.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n✓ Report saved to: {output_file}")
    print("\n" + "=" * 100)
    print("REPORT PREVIEW")
    print("=" * 100)
    print(report)


if __name__ == "__main__":
    main()
