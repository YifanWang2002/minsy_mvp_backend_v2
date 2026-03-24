"""Benchmark pre-strategy feature-engine cache effectiveness."""

from __future__ import annotations

import argparse
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from packages.domain.market_data.regime import feature_snapshot as regime_snapshot


def _build_ohlcv(rows: int = 900) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC")
    x = np.linspace(0.0, 24.0, rows)
    base = 100.0 + np.sin(x) * 3.0 + np.linspace(0.0, 8.0, rows)
    noise = np.cos(x * 2.3) * 0.15
    close = base + noise
    open_ = close + np.sin(x * 1.7) * 0.12
    high = np.maximum(open_, close) + 0.5 + np.abs(np.sin(x * 0.7)) * 0.2
    low = np.minimum(open_, close) - 0.5 - np.abs(np.cos(x * 0.6)) * 0.2
    volume = 1000.0 + 120.0 * np.sin(x * 1.3) + np.linspace(0.0, 50.0, rows)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.maximum(volume, 1.0),
        },
        index=idx,
    )


def _run_once(frame: pd.DataFrame, lookback_bars: int) -> float:
    started = time.perf_counter()
    regime_snapshot.build_regime_feature_snapshot(
        frame,
        timeframe="1h",
        lookback_bars=lookback_bars,
        pivot_window=5,
    )
    return (time.perf_counter() - started) * 1000.0


def _build_report(*, cold_ms: float, hot_ms: list[float], stats: dict[str, object]) -> str:
    avg_hot = sum(hot_ms) / max(len(hot_ms), 1)
    p95_hot = sorted(hot_ms)[int(round((len(hot_ms) - 1) * 0.95))] if hot_ms else 0.0
    improvement = ((cold_ms - avg_hot) / cold_ms) if cold_ms > 0 else 0.0
    lines = [
        "# Pre-Strategy Cache Benchmark",
        "",
        f"- Generated at (UTC): `{datetime.now(UTC).isoformat()}`",
        f"- Cold run latency (ms): `{cold_ms:.3f}`",
        f"- Hot run avg latency (ms): `{avg_hot:.3f}`",
        f"- Hot run p95 latency (ms): `{p95_hot:.3f}`",
        f"- Relative improvement (cold -> hot avg): `{improvement:.4f}`",
        "",
        "## Cache Stats",
        "",
        f"- entries: `{stats.get('entries')}`",
        f"- hits: `{stats.get('hits')}`",
        f"- misses: `{stats.get('misses')}`",
        f"- ttl_seconds: `{stats.get('ttl_seconds')}`",
        f"- max_entries: `{stats.get('max_entries')}`",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=900)
    parser.add_argument("--lookback", type=int, default=500)
    parser.add_argument("--hot-runs", type=int, default=20)
    parser.add_argument(
        "--output",
        default="dev_docs/pre_strategy_cache_benchmark_2026-03-17.md",
    )
    args = parser.parse_args()

    frame = _build_ohlcv(rows=max(120, int(args.rows)))
    lookback = max(120, int(args.lookback))
    hot_runs = max(1, int(args.hot_runs))

    regime_snapshot.reset_feature_engine_cache()
    cold_ms = _run_once(frame, lookback)
    hot_ms = [_run_once(frame, lookback) for _ in range(hot_runs)]
    stats = regime_snapshot.get_feature_engine_cache_stats()

    report = _build_report(cold_ms=cold_ms, hot_ms=hot_ms, stats=stats)
    output_path = Path(args.output)
    output_path.write_text(report, encoding="utf-8")
    print(f"wrote cache benchmark report: {output_path}")


if __name__ == "__main__":
    main()

