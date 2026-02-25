#!/usr/bin/env python3
"""Minimal paper-trading load probe for status endpoint latency."""

from __future__ import annotations

import argparse
import asyncio
import statistics
from time import monotonic, perf_counter

import httpx


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    rank = max(0, min(len(values) - 1, int(round((len(values) - 1) * q))))
    return sorted(values)[rank]


async def run_loadtest(
    *,
    base_url: str,
    hours: float,
    interval_seconds: float,
    timeout_seconds: float,
) -> None:
    duration_seconds = max(1.0, hours * 3600.0)
    deadline = monotonic() + duration_seconds
    latencies_ms: list[float] = []
    successes = 0
    failures = 0

    async with httpx.AsyncClient(timeout=max(1.0, timeout_seconds), trust_env=False) as client:
        while monotonic() < deadline:
            start = perf_counter()
            try:
                response = await client.get(f"{base_url.rstrip('/')}/api/v1/status")
                response.raise_for_status()
            except Exception:  # noqa: BLE001
                failures += 1
            else:
                successes += 1
                latencies_ms.append((perf_counter() - start) * 1000.0)
            await asyncio.sleep(max(0.0, interval_seconds))

    print("paper trading load probe complete")
    print(f"base_url={base_url}")
    print(f"duration_hours={hours}")
    print(f"successes={successes}")
    print(f"failures={failures}")
    if latencies_ms:
        print(f"latency_p50_ms={_percentile(latencies_ms, 0.50):.2f}")
        print(f"latency_p95_ms={_percentile(latencies_ms, 0.95):.2f}")
        print(f"latency_avg_ms={statistics.fmean(latencies_ms):.2f}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Backend base URL")
    parser.add_argument("--hours", type=float, default=4.0, help="Duration in hours")
    parser.add_argument("--interval-seconds", type=float, default=0.5, help="Polling interval")
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=12.0,
        help="HTTP timeout applied to status and optional SSE probe.",
    )
    parser.add_argument(
        "--stream-deployment-id",
        default="",
        help="Optional deployment id for SSE probe (requires auth token).",
    )
    parser.add_argument(
        "--auth-token",
        default="",
        help="Optional bearer token used when probing SSE endpoint.",
    )
    return parser.parse_args()


async def run_sse_probe(
    *,
    base_url: str,
    deployment_id: str,
    auth_token: str,
    timeout_seconds: float,
) -> bool:
    if not deployment_id or not auth_token:
        return False
    headers = {"Authorization": f"Bearer {auth_token}"}
    url = (
        f"{base_url.rstrip('/')}/api/v1/stream/deployments/{deployment_id}"
        "?max_events=6&poll_seconds=0.2"
    )
    async with httpx.AsyncClient(
        timeout=max(1.0, timeout_seconds),
        headers=headers,
        trust_env=False,
    ) as client:
        response = await client.get(url)
        if response.status_code != 200:
            return False
        return "event: heartbeat" in response.text and "event: pnl_update" in response.text


if __name__ == "__main__":
    args = _parse_args()
    asyncio.run(
        run_loadtest(
            base_url=args.base_url,
            hours=args.hours,
            interval_seconds=args.interval_seconds,
            timeout_seconds=args.timeout_seconds,
        )
    )
    if args.stream_deployment_id and args.auth_token:
        ok = asyncio.run(
            run_sse_probe(
                base_url=args.base_url,
                deployment_id=args.stream_deployment_id,
                auth_token=args.auth_token,
                timeout_seconds=args.timeout_seconds,
            )
        )
        print(f"sse_probe_ok={ok}")
