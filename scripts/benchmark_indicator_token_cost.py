"""Benchmark token footprint for indicator tool payloads.

This script estimates pre/post payload size for strategy indicator discovery.
Pre-change is simulated as "catalog + detail + skill markdown body".
Post-change is current runtime payloads (skills disabled by default).
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

from apps.mcp.domains.strategy import tools as strategy_tools
from packages.domain.strategy.feature.indicators import IndicatorRegistry


def _estimate_tokens(text: str) -> int:
    # Lightweight approximation used consistently for before/after comparison.
    return max(1, int(round(len(text) / 4.0)))


def _percentile(values: list[int], pct: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * pct))
    return ordered[max(0, min(index, len(ordered) - 1))]


def _simulate_old_detail_payload(raw_detail: dict[str, Any]) -> str:
    cloned = json.loads(json.dumps(raw_detail, ensure_ascii=False))
    if not isinstance(cloned, dict):
        return json.dumps(raw_detail, ensure_ascii=False)
    indicators = cloned.get("indicators")
    if not isinstance(indicators, list):
        return json.dumps(cloned, ensure_ascii=False)
    skill_dir = Path("apps/mcp/domains/strategy/skills/indicators")
    for item in indicators:
        if not isinstance(item, dict):
            continue
        indicator = str(item.get("indicator", "")).strip().lower()
        if not indicator:
            continue
        skill_path = skill_dir / f"{indicator}.md"
        if not skill_path.exists():
            continue
        item["skill_path"] = str(skill_path)
        item["content"] = skill_path.read_text(encoding="utf-8")
    return json.dumps(cloned, ensure_ascii=False)


def _build_report(*, sample_size: int, sample_width: int, seed: int) -> dict[str, Any]:
    all_indicators = IndicatorRegistry.list_all()
    if not all_indicators:
        raise RuntimeError("No indicators registered")

    skill_dir = Path("apps/mcp/domains/strategy/skills/indicators")
    skill_indicators = sorted(
        path.stem.strip().lower()
        for path in skill_dir.glob("*.md")
        if path.stem.strip()
    )
    skill_indicators = [name for name in skill_indicators if name in set(all_indicators)]

    catalog_raw = strategy_tools.get_indicator_catalog()
    catalog_tokens = _estimate_tokens(catalog_raw)

    def _sample_metrics(pool: list[str], *, local_seed: int) -> dict[str, Any]:
        random_generator = random.Random(local_seed)
        old_totals: list[int] = []
        new_totals: list[int] = []
        old_detail_only: list[int] = []
        new_detail_only: list[int] = []
        for _ in range(sample_size):
            picks = random_generator.sample(pool, min(sample_width, len(pool)))
            detail_raw = strategy_tools.get_indicator_detail(indicator_list=picks)
            detail_payload = json.loads(detail_raw)
            old_detail_raw = _simulate_old_detail_payload(detail_payload)

            new_detail_tokens = _estimate_tokens(detail_raw)
            old_detail_tokens = _estimate_tokens(old_detail_raw)
            new_total_tokens = catalog_tokens + new_detail_tokens
            old_total_tokens = catalog_tokens + old_detail_tokens

            new_detail_only.append(new_detail_tokens)
            old_detail_only.append(old_detail_tokens)
            new_totals.append(new_total_tokens)
            old_totals.append(old_total_tokens)
        old_p50 = _percentile(old_totals, 0.50)
        old_p95 = _percentile(old_totals, 0.95)
        new_p50 = _percentile(new_totals, 0.50)
        new_p95 = _percentile(new_totals, 0.95)
        return {
            "old": {
                "p50_total_tokens": old_p50,
                "p95_total_tokens": old_p95,
                "avg_total_tokens": int(round(mean(old_totals))),
                "avg_detail_tokens": int(round(mean(old_detail_only))),
            },
            "new": {
                "p50_total_tokens": new_p50,
                "p95_total_tokens": new_p95,
                "avg_total_tokens": int(round(mean(new_totals))),
                "avg_detail_tokens": int(round(mean(new_detail_only))),
            },
            "savings": {
                "p50_tokens": old_p50 - new_p50,
                "p95_tokens": old_p95 - new_p95,
                "p50_ratio": round((old_p50 - new_p50) / old_p50, 4) if old_p50 else 0.0,
                "p95_ratio": round((old_p95 - new_p95) / old_p95, 4) if old_p95 else 0.0,
            },
        }

    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "sample_size": sample_size,
        "sample_width": sample_width,
        "seed": seed,
        "catalog_tokens": catalog_tokens,
        "skill_indicator_count": len(skill_indicators),
        "cohorts": {
            "all_indicators": _sample_metrics(all_indicators, local_seed=seed),
            "skill_indicators_only": _sample_metrics(
                skill_indicators if skill_indicators else all_indicators,
                local_seed=seed + 1,
            ),
        },
    }


def _markdown(report: dict[str, Any]) -> str:
    cohorts = report.get("cohorts", {})
    all_row = cohorts.get("all_indicators", {})
    skill_row = cohorts.get("skill_indicators_only", {})

    def _cohort_table(title: str, payload: dict[str, Any]) -> list[str]:
        old = payload.get("old", {})
        new = payload.get("new", {})
        savings = payload.get("savings", {})
        return [
            f"## {title}",
            "",
            "| metric | pre-change (simulated old) | current | delta |",
            "|---|---:|---:|---:|",
            f"| p50 | {old.get('p50_total_tokens', 0)} | {new.get('p50_total_tokens', 0)} | {savings.get('p50_tokens', 0)} |",
            f"| p95 | {old.get('p95_total_tokens', 0)} | {new.get('p95_total_tokens', 0)} | {savings.get('p95_tokens', 0)} |",
            f"| avg | {old.get('avg_total_tokens', 0)} | {new.get('avg_total_tokens', 0)} | {int(old.get('avg_total_tokens', 0)) - int(new.get('avg_total_tokens', 0))} |",
            "",
            f"- p50 savings ratio: `{savings.get('p50_ratio', 0.0)}`",
            f"- p95 savings ratio: `{savings.get('p95_ratio', 0.0)}`",
            "",
        ]

    lines = [
        "# Indicator Token Cost Baseline",
        "",
        f"- Generated at (UTC): `{report['generated_at_utc']}`",
        f"- Sample size: `{report['sample_size']}` tasks",
        f"- Indicators per task: `{report['sample_width']}`",
        f"- Catalog payload (approx tokens): `{report['catalog_tokens']}`",
        f"- Skill indicator count: `{report['skill_indicator_count']}`",
        "",
    ]
    lines.extend(_cohort_table("All Indicators (random cohort)", all_row))
    lines.extend(_cohort_table("Skill Indicators Only (focused cohort)", skill_row))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--sample-width", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260317)
    parser.add_argument(
        "--output",
        default="dev_docs/indicator_token_cost_baseline_2026-03-17.md",
    )
    args = parser.parse_args()

    report = _build_report(
        sample_size=max(1, int(args.sample_size)),
        sample_width=max(1, int(args.sample_width)),
        seed=int(args.seed),
    )
    output_path = Path(args.output)
    output_path.write_text(_markdown(report), encoding="utf-8")
    print(f"wrote token benchmark report: {output_path}")


if __name__ == "__main__":
    main()
