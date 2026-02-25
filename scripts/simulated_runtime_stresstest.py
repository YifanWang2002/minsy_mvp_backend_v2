#!/usr/bin/env python3
# ruff: noqa: E402
"""Simulated multi-user paper-trading stress test with CPU/memory telemetry.

This script does not pull live market data or submit broker orders.
It exercises:
- user auth/register
- broker account creation per user
- strategy confirm + deployment lifecycle
- repeated process-now cycles with synthetic bars
- signal/order persistence paths
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import psutil
from dotenv import load_dotenv
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings  # noqa: E402
from src.engine.execution.adapters.base import OhlcvBar  # noqa: E402
from src.engine.execution.signal_store import signal_store  # noqa: E402
from src.engine.market_data.aggregator import BarAggregator  # noqa: E402
from src.engine.market_data.runtime import market_data_runtime  # noqa: E402
from src.engine.strategy import EXAMPLE_PATH, load_strategy_payload  # noqa: E402
from src.main import app  # noqa: E402


@dataclass(frozen=True, slots=True)
class DeploymentRef:
    deployment_id: UUID
    token: str
    symbol: str
    variant: str
    timeframe: str


class ResourceSampler:
    def __init__(self, interval_seconds: float = 1.0) -> None:
        self.interval_seconds = max(0.1, interval_seconds)
        self._rss_mb: list[float] = []
        self._cpu_pct: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        process = psutil.Process()
        process.cpu_percent(interval=None)

        def _run() -> None:
            while not self._stop.is_set():
                rss_mb = process.memory_info().rss / (1024 * 1024)
                cpu = process.cpu_percent(interval=None)
                self._rss_mb.append(rss_mb)
                self._cpu_pct.append(cpu)
                self._stop.wait(self.interval_seconds)

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, float]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        return {
            "rss_avg_mb": _avg(self._rss_mb),
            "rss_p95_mb": _percentile(self._rss_mb, 0.95),
            "rss_max_mb": max(self._rss_mb) if self._rss_mb else 0.0,
            "cpu_avg_pct": _avg(self._cpu_pct),
            "cpu_p95_pct": _percentile(self._cpu_pct, 0.95),
            "cpu_max_pct": max(self._cpu_pct) if self._cpu_pct else 0.0,
            "sample_count": float(len(self._rss_mb)),
        }


def _avg(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    idx = int(round((len(values) - 1) * max(0.0, min(1.0, q))))
    return sorted(values)[idx]


def _parse_csv(raw: str, *, upper: bool = False) -> list[str]:
    values: list[str] = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        token = token.upper() if upper else token.lower()
        if token not in values:
            values.append(token)
    return values


def _disabled_short_entry() -> dict[str, Any]:
    return {"cmp": {"left": {"ref": "price.close"}, "op": "lt", "right": 0}}


def _replace_factor_refs(node: Any, factor_id_map: dict[str, str]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "ref" and isinstance(value, str):
                replacement = value
                for old_id, new_id in factor_id_map.items():
                    if replacement == old_id:
                        replacement = new_id
                        break
                    prefix = f"{old_id}."
                    if replacement.startswith(prefix):
                        replacement = f"{new_id}.{replacement[len(prefix):]}"
                        break
                node[key] = replacement
                continue
            _replace_factor_refs(value, factor_id_map)
        return

    if isinstance(node, list):
        for item in node:
            _replace_factor_refs(item, factor_id_map)


def _rename_factor_ids(payload: dict[str, Any], factor_id_map: dict[str, str]) -> None:
    factors = payload.get("factors")
    if isinstance(factors, dict):
        remapped: dict[str, Any] = {}
        for factor_id, definition in factors.items():
            remapped[factor_id_map.get(str(factor_id), str(factor_id))] = definition
        payload["factors"] = remapped
    _replace_factor_refs(payload.get("trade"), factor_id_map)


def _build_strategy_payload(*, symbol: str, variant: str, timeframe: str) -> dict[str, Any]:
    payload = load_strategy_payload(EXAMPLE_PATH)
    payload["strategy"]["name"] = f"stress {symbol} {variant} {timeframe}"
    payload["strategy"]["description"] = "simulated runtime stress strategy"
    payload["universe"] = {"market": "crypto", "tickers": [symbol]}
    payload["timeframe"] = timeframe

    payload["factors"]["ema_9"]["params"]["period"] = 3
    payload["factors"]["ema_21"]["params"]["period"] = 6
    payload["factors"]["rsi_14"]["params"]["period"] = 4
    _rename_factor_ids(
        payload,
        {
            "ema_9": "ema_3",
            "ema_21": "ema_6",
            "rsi_14": "rsi_4",
        },
    )

    if variant == "trend":
        payload["trade"]["long"]["entry"]["condition"] = {
            "cmp": {"left": {"ref": "ema_3"}, "op": "gt", "right": {"ref": "ema_6"}}
        }
        payload["trade"]["long"]["exits"] = [
            {
                "type": "signal_exit",
                "name": "trend_exit",
                "condition": {"cmp": {"left": {"ref": "ema_3"}, "op": "lt", "right": {"ref": "ema_6"}}},
            }
        ]
    elif variant == "mean":
        payload["trade"]["long"]["entry"]["condition"] = {
            "cmp": {"left": {"ref": "rsi_4"}, "op": "lt", "right": 45}
        }
        payload["trade"]["long"]["exits"] = [
            {
                "type": "signal_exit",
                "name": "mean_exit",
                "condition": {"cmp": {"left": {"ref": "rsi_4"}, "op": "gt", "right": 55}},
            }
        ]
    elif variant == "breakout":
        payload["trade"]["long"]["entry"]["condition"] = {
            "all": [
                {"cmp": {"left": {"ref": "price.close"}, "op": "gt", "right": {"ref": "ema_3"}}},
                {"cmp": {"left": {"ref": "rsi_4"}, "op": "gt", "right": 50}},
            ]
        }
        payload["trade"]["long"]["exits"] = [
            {
                "type": "signal_exit",
                "name": "breakout_exit",
                "condition": {"cmp": {"left": {"ref": "price.close"}, "op": "lt", "right": {"ref": "ema_6"}}},
            }
        ]
    else:
        raise ValueError(f"unsupported variant: {variant}")

    payload["trade"]["short"]["entry"]["condition"] = _disabled_short_entry()
    payload["trade"]["short"]["exits"] = [
        {
            "type": "signal_exit",
            "name": "short_exit_disabled",
            "condition": _disabled_short_entry(),
        }
    ]
    return payload


def _resolve_base_price(symbol: str) -> Decimal:
    if symbol == "BTCUSD":
        return Decimal("68000")
    if symbol == "ETHUSD":
        return Decimal("2000")
    if symbol == "SOLUSD":
        return Decimal("90")
    if symbol == "LTCUSD":
        return Decimal("60")
    if symbol == "DOGEUSD":
        return Decimal("0.12")
    return Decimal("100")


def _resolve_order_qty(symbol: str) -> float:
    if symbol == "BTCUSD":
        return 0.0002
    if symbol == "ETHUSD":
        return 0.01
    if symbol == "DOGEUSD":
        return 100.0
    return 0.2


def _make_bar(
    *,
    symbol: str,
    ts: datetime,
    idx: int,
    prev_close: Decimal,
) -> Decimal:
    base = _resolve_base_price(symbol)
    wave = Decimal(str(math.sin(idx / 3.0) * 0.002))
    noise = Decimal(str((random.random() - 0.5) * 0.0015))
    drift = Decimal(str(math.sin(idx / 13.0) * 0.0005))
    pct_move = wave + noise + drift
    close = (prev_close * (Decimal("1") + pct_move)).quantize(Decimal("0.00000001"))
    if close <= 0:
        close = base
    open_ = prev_close
    high = max(open_, close) * Decimal("1.001")
    low = min(open_, close) * Decimal("0.999")
    volume = Decimal("1.0") + Decimal(str(abs(math.sin(idx / 5.0))))
    market_data_runtime.ingest_1m_bar(
        market="crypto",
        symbol=symbol,
        bar=OhlcvBar(
            timestamp=ts,
            open=open_,
            high=high,
            low=low,
            close=close,
            volume=volume,
        ),
    )
    return close


def _seed_bars(symbols: list[str], *, warmup_bars: int) -> tuple[datetime, dict[str, Decimal]]:
    start_ts = datetime.now(UTC).replace(second=0, microsecond=0) - timedelta(minutes=warmup_bars)
    closes = {symbol: _resolve_base_price(symbol) for symbol in symbols}
    ts = start_ts
    for idx in range(warmup_bars):
        for symbol in symbols:
            closes[symbol] = _make_bar(
                symbol=symbol,
                ts=ts,
                idx=idx,
                prev_close=closes[symbol],
            )
        ts = ts + timedelta(minutes=1)
    return ts, closes


def _register_user(client: TestClient) -> str:
    email = f"sim_stress_{uuid4().hex}@test.com"
    resp = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "pass1234", "name": "Sim Stress User"},
    )
    if resp.status_code != 201:
        raise RuntimeError(f"register failed: {resp.status_code} {resp.text}")
    return str(resp.json()["access_token"])


def _create_broker_account(client: TestClient, headers: dict[str, str], *, mode: str) -> str:
    resp = client.post(
        "/api/v1/broker-accounts",
        headers=headers,
        json={
            "provider": "alpaca",
            "mode": mode,
            "credentials": {"api_key": "demo", "api_secret": "demo"},
            "metadata": {"source": "simulated_runtime_stress"},
        },
    )
    if resp.status_code != 201:
        raise RuntimeError(f"broker account create failed: {resp.status_code} {resp.text}")
    return str(resp.json()["broker_account_id"])


def _create_deployments_for_user(
    client: TestClient,
    *,
    headers: dict[str, str],
    broker_account_id: str,
    symbols: list[str],
    variants: list[str],
    timeframes: list[str],
    mode: str,
) -> list[DeploymentRef]:
    refs: list[DeploymentRef] = []
    for symbol in symbols:
        order_qty = _resolve_order_qty(symbol)
        for timeframe in timeframes:
            for variant in variants:
                strategy_payload = _build_strategy_payload(
                    symbol=symbol,
                    variant=variant,
                    timeframe=timeframe,
                )
                thread = client.post("/api/v1/chat/new-thread", headers=headers, json={"metadata": {}})
                if thread.status_code != 201:
                    raise RuntimeError(f"new-thread failed: {thread.status_code} {thread.text}")
                session_id = thread.json()["session_id"]
                confirm = client.post(
                    "/api/v1/strategies/confirm",
                    headers=headers,
                    json={
                        "session_id": session_id,
                        "dsl_json": strategy_payload,
                        "auto_start_backtest": False,
                    },
                )
                if confirm.status_code != 200:
                    raise RuntimeError(f"strategy confirm failed: {confirm.status_code} {confirm.text}")
                strategy_id = confirm.json()["strategy_id"]
                create = client.post(
                    "/api/v1/deployments",
                    headers=headers,
                    json={
                        "strategy_id": strategy_id,
                        "broker_account_id": broker_account_id,
                        "mode": mode,
                        "capital_allocated": "10000",
                        "risk_limits": {
                            "order_qty": order_qty,
                            "max_position_notional": 5000,
                            "max_symbol_exposure_pct": 0.95,
                        },
                        "runtime_state": {"source": "simulated_runtime_stress"},
                    },
                )
                if create.status_code != 201:
                    raise RuntimeError(f"deployment create failed: {create.status_code} {create.text}")
                deployment_id = UUID(create.json()["deployment_id"])
                start = client.post(f"/api/v1/deployments/{deployment_id}/start", headers=headers)
                if start.status_code != 200:
                    raise RuntimeError(f"deployment start failed: {start.status_code} {start.text}")
                refs.append(
                    DeploymentRef(
                        deployment_id=deployment_id,
                        token=headers["Authorization"].split(" ", 1)[1],
                        symbol=symbol,
                        variant=variant,
                        timeframe=timeframe,
                    )
                )
    return refs


def _configure_runtime_timeframes(timeframes: list[str]) -> None:
    aggregate_timeframes = tuple(tf for tf in timeframes if tf != "1m")
    market_data_runtime._aggregator = BarAggregator(  # noqa: SLF001
        timeframes=aggregate_timeframes or ("5m",),
        timezone=settings.market_data_aggregate_timezone,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--users", type=int, default=10, help="Concurrent simulated users.")
    parser.add_argument(
        "--symbols",
        default="BTCUSD,ETHUSD,SOLUSD",
        help="Comma-separated symbols.",
    )
    parser.add_argument(
        "--variants",
        default="trend,mean,breakout",
        help="Comma-separated strategy variants.",
    )
    parser.add_argument(
        "--timeframes",
        default="1m,2m,5m",
        help="Comma-separated strategy timeframes.",
    )
    parser.add_argument("--warmup-bars", type=int, default=180, help="Warmup 1m bars per symbol.")
    parser.add_argument("--cycles", type=int, default=8, help="Signal-processing cycles.")
    parser.add_argument("--mode", choices=["paper", "live"], default="paper")
    parser.add_argument("--cpu-sample-seconds", type=float, default=1.0, help="CPU/RSS sample interval.")
    parser.add_argument(
        "--artifact-dir",
        default="artifacts/simulated_runtime_stress",
        help="Directory to write JSON summary.",
    )
    return parser.parse_args()


def main() -> None:
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        load_dotenv(env_file)

    args = _parse_args()
    if args.users <= 0:
        raise SystemExit("--users must be > 0")
    if args.cycles <= 0:
        raise SystemExit("--cycles must be > 0")

    symbols = _parse_csv(args.symbols, upper=True)
    variants = _parse_csv(args.variants, upper=False)
    timeframes = _parse_csv(args.timeframes, upper=False)
    if not symbols or not variants or not timeframes:
        raise SystemExit("symbols/variants/timeframes must be non-empty")

    settings.paper_trading_execute_orders = False
    settings.paper_trading_enqueue_on_start = False
    settings.paper_trading_enabled = True
    settings.paper_trading_kill_switch_global = False
    _configure_runtime_timeframes(timeframes)

    signal_store.clear()
    market_data_runtime.reset()

    ts, closes = _seed_bars(symbols, warmup_bars=args.warmup_bars)

    start_wall = time.perf_counter()
    setup_start = time.perf_counter()
    deployments: list[DeploymentRef] = []
    errors = 0
    process_latencies_ms: list[float] = []
    signal_counts: dict[str, int] = {}
    orders_emitted = 0

    sampler = ResourceSampler(interval_seconds=args.cpu_sample_seconds)
    sampler.start()
    try:
        with TestClient(app) as client:
            for _ in range(args.users):
                token = _register_user(client)
                headers = {"Authorization": f"Bearer {token}"}
                broker_account_id = _create_broker_account(client, headers, mode=args.mode)
                deployments.extend(
                    _create_deployments_for_user(
                        client,
                        headers=headers,
                        broker_account_id=broker_account_id,
                        symbols=symbols,
                        variants=variants,
                        timeframes=timeframes,
                        mode=args.mode,
                    )
                )

            setup_seconds = time.perf_counter() - setup_start
            ts_cursor = ts
            for cycle in range(1, args.cycles + 1):
                for symbol in symbols:
                    closes[symbol] = _make_bar(
                        symbol=symbol,
                        ts=ts_cursor,
                        idx=args.warmup_bars + cycle,
                        prev_close=closes[symbol],
                    )
                ts_cursor = ts_cursor + timedelta(minutes=1)

                for deployment in deployments:
                    headers = {"Authorization": f"Bearer {deployment.token}"}
                    t0 = time.perf_counter()
                    resp = client.post(
                        f"/api/v1/deployments/{deployment.deployment_id}/process-now",
                        headers=headers,
                    )
                    latency_ms = (time.perf_counter() - t0) * 1000.0
                    process_latencies_ms.append(latency_ms)
                    if resp.status_code != 200:
                        errors += 1
                        continue
                    body = resp.json()
                    signal = str(body.get("signal", ""))
                    signal_counts[signal] = signal_counts.get(signal, 0) + 1
                    if body.get("order_id"):
                        orders_emitted += 1

            # Stop deployments for clean state.
            for deployment in deployments:
                headers = {"Authorization": f"Bearer {deployment.token}"}
                client.post(f"/api/v1/deployments/{deployment.deployment_id}/stop", headers=headers)
    finally:
        resource = sampler.stop()

    total_seconds = time.perf_counter() - start_wall
    total_process_calls = len(process_latencies_ms)
    throughput_calls_per_sec = (total_process_calls / total_seconds) if total_seconds > 0 else 0.0
    orders_per_sec = (orders_emitted / total_seconds) if total_seconds > 0 else 0.0

    summary = {
        "ts": datetime.now(UTC).isoformat(),
        "users": args.users,
        "symbols": symbols,
        "variants": variants,
        "timeframes": timeframes,
        "deployments": len(deployments),
        "cycles": args.cycles,
        "warmup_bars": args.warmup_bars,
        "total_seconds": round(total_seconds, 3),
        "setup_seconds": round(setup_seconds, 3),
        "total_process_calls": total_process_calls,
        "throughput_calls_per_sec": round(throughput_calls_per_sec, 3),
        "orders_emitted": orders_emitted,
        "orders_per_sec": round(orders_per_sec, 3),
        "errors": errors,
        "latency_ms_p50": round(_percentile(process_latencies_ms, 0.50), 3),
        "latency_ms_p95": round(_percentile(process_latencies_ms, 0.95), 3),
        "latency_ms_p99": round(_percentile(process_latencies_ms, 0.99), 3),
        "latency_ms_avg": round(_avg(process_latencies_ms), 3),
        "signals": signal_counts,
        "resource": {
            "rss_avg_mb": round(resource["rss_avg_mb"], 3),
            "rss_p95_mb": round(resource["rss_p95_mb"], 3),
            "rss_max_mb": round(resource["rss_max_mb"], 3),
            "cpu_avg_pct": round(resource["cpu_avg_pct"], 3),
            "cpu_p95_pct": round(resource["cpu_p95_pct"], 3),
            "cpu_max_pct": round(resource["cpu_max_pct"], 3),
            "samples": int(resource["sample_count"]),
        },
        "notes": [
            "Simulated bars and local process-now calls only (no external market data, no broker order submission).",
            "Use as architecture baseline, not external-network SLA.",
        ],
    }

    out_root = Path(args.artifact_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    out_file = out_root / f"simulated_runtime_stress_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    out_file.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=True, indent=2))
    print(f"[SIM_STRESS][ARTIFACT] {out_file}")


if __name__ == "__main__":
    main()
