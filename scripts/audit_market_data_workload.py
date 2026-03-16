"""Audit market-data workload from one or multiple API environments.

Usage:
    uv run python scripts/audit_market_data_workload.py \
      --target local=http://127.0.0.1:8000 \
      --target remote=https://api.example.com \
      --token "$API_TOKEN"
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any

import httpx


@dataclass(slots=True)
class TargetSnapshot:
    target: str
    base_url: str
    window_minutes: int
    active_instruments: int
    unique_symbols: int
    by_market: dict[str, int]
    refresh_scheduler_metrics: dict[str, Any]
    http_429_total: int
    http_429_by_symbol: dict[str, int]
    http_429_by_endpoint: dict[str, int]


def _parse_target(value: str) -> tuple[str, str]:
    raw = str(value).strip()
    if "=" not in raw:
        raise argparse.ArgumentTypeError(
            "--target must use '<label>=<base_url>' format, e.g. local=http://127.0.0.1:8000"
        )
    label, base_url = raw.split("=", 1)
    label = label.strip()
    base_url = base_url.strip().rstrip("/")
    if not label or not base_url:
        raise argparse.ArgumentTypeError("target label and base_url must not be empty")
    return label, base_url


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit market-data workload snapshots")
    parser.add_argument(
        "--target",
        action="append",
        type=_parse_target,
        required=True,
        help="label and base URL, format: local=http://127.0.0.1:8000",
    )
    parser.add_argument(
        "--token",
        default="",
        help="optional bearer token for authenticated /api/v1/market-data/health",
    )
    parser.add_argument(
        "--window-minutes",
        type=int,
        default=60,
        help="market-data health lookback window in minutes (default: 60)",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=20.0,
        help="HTTP timeout in seconds (default: 20)",
    )
    parser.add_argument(
        "--output",
        choices=("json", "table"),
        default="table",
        help="output format (default: table)",
    )
    return parser


def _headers(token: str) -> dict[str, str]:
    normalized = token.strip()
    if not normalized:
        return {}
    return {"Authorization": f"Bearer {normalized}"}


async def _fetch_target_snapshot(
    *,
    label: str,
    base_url: str,
    token: str,
    window_minutes: int,
    timeout_seconds: float,
) -> TargetSnapshot:
    url = f"{base_url}/api/v1/market-data/health"
    headers = _headers(token)
    params = {"window_minutes": max(1, int(window_minutes))}
    timeout = httpx.Timeout(timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(url, params=params, headers=headers)
        response.raise_for_status()
        payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError(f"{label}: unexpected response payload type")

    active_subscriptions = payload.get("active_subscriptions", [])
    by_market_counter: Counter[str] = Counter()
    symbol_set: set[str] = set()
    if isinstance(active_subscriptions, list):
        for row in active_subscriptions:
            if not isinstance(row, dict):
                continue
            market = str(row.get("market", "unknown")).strip().lower() or "unknown"
            symbol = str(row.get("symbol", "")).strip().upper()
            by_market_counter[market] += 1
            if symbol:
                symbol_set.add(symbol)

    error_summary = payload.get("error_summary", [])
    total_429 = 0
    if isinstance(error_summary, list):
        for row in error_summary:
            if not isinstance(row, dict):
                continue
            status = row.get("http_status")
            if status == 429:
                total_429 += int(row.get("count", 0) or 0)

    by_symbol: defaultdict[str, int] = defaultdict(int)
    by_endpoint: defaultdict[str, int] = defaultdict(int)
    recent_errors = payload.get("recent_errors", [])
    if isinstance(recent_errors, list):
        for row in recent_errors:
            if not isinstance(row, dict):
                continue
            if row.get("http_status") != 429:
                continue
            symbol = str(row.get("symbol", "")).strip().upper() or "<UNKNOWN>"
            endpoint = str(row.get("endpoint", "")).strip() or "<UNKNOWN>"
            by_symbol[symbol] += 1
            by_endpoint[endpoint] += 1

    refresh_metrics = payload.get("refresh_scheduler_metrics")
    if not isinstance(refresh_metrics, dict):
        refresh_metrics = {}

    return TargetSnapshot(
        target=label,
        base_url=base_url,
        window_minutes=max(1, int(window_minutes)),
        active_instruments=sum(by_market_counter.values()),
        unique_symbols=len(symbol_set),
        by_market=dict(sorted(by_market_counter.items())),
        refresh_scheduler_metrics=refresh_metrics,
        http_429_total=total_429,
        http_429_by_symbol=dict(sorted(by_symbol.items(), key=lambda item: (-item[1], item[0]))),
        http_429_by_endpoint=dict(sorted(by_endpoint.items(), key=lambda item: (-item[1], item[0]))),
    )


def _render_table(rows: list[TargetSnapshot]) -> str:
    lines: list[str] = []
    for row in rows:
        lines.append(f"[{row.target}] {row.base_url}")
        lines.append(
            f"  active_instruments={row.active_instruments} unique_symbols={row.unique_symbols} "
            f"window_minutes={row.window_minutes}"
        )
        lines.append(f"  by_market={json.dumps(row.by_market, ensure_ascii=True)}")
        lines.append(
            "  refresh_metrics="
            + json.dumps(row.refresh_scheduler_metrics, ensure_ascii=True, sort_keys=True)
        )
        lines.append(f"  http_429_total={row.http_429_total}")
        lines.append(
            "  http_429_by_symbol="
            + json.dumps(row.http_429_by_symbol, ensure_ascii=True, sort_keys=True)
        )
        lines.append(
            "  http_429_by_endpoint="
            + json.dumps(row.http_429_by_endpoint, ensure_ascii=True, sort_keys=True)
        )
    return "\n".join(lines)


async def _run() -> int:
    args = _build_parser().parse_args()
    tasks = [
        _fetch_target_snapshot(
            label=label,
            base_url=base_url,
            token=args.token,
            window_minutes=args.window_minutes,
            timeout_seconds=args.timeout_seconds,
        )
        for label, base_url in args.target
    ]
    snapshots = list(await asyncio.gather(*tasks))
    snapshots.sort(key=lambda item: item.target)

    if args.output == "json":
        print(json.dumps([asdict(item) for item in snapshots], ensure_ascii=True, indent=2))
    else:
        print(_render_table(snapshots))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_run()))
