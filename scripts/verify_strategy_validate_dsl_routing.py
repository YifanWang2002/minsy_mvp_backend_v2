#!/usr/bin/env python3
"""Probe strategy_validate_dsl through the same public MCP URL the API advertises."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

# Ensure service-scoped env layering matches the API process by default.
os.environ.setdefault("MINSY_SERVICE", "api")

from packages.shared_settings.loader.service_loader import get_api_settings


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Call strategy_validate_dsl through the configured public MCP URL and "
            "optionally compare it against another URL."
        )
    )
    parser.add_argument("--session-id", required=True, help="Existing chat session UUID.")
    parser.add_argument(
        "--mcp-url",
        default="",
        help="Override MCP URL. Defaults to API settings.strategy_mcp_server_url.",
    )
    parser.add_argument(
        "--compare-url",
        default="",
        help="Optional second URL to compare against (for example mcp.minsyai.com).",
    )
    parser.add_argument(
        "--expect-compare-failure",
        action="store_true",
        help="Treat a compare probe failure as the expected outcome.",
    )
    return parser


def _build_probe_dsl() -> dict[str, Any]:
    return {
        "dsl_version": "1.0.0",
        "strategy": {
            "name": "Routing Probe",
            "description": "Minimal valid DSL used to verify MCP routing.",
        },
        "universe": {
            "market": "futures",
            "tickers": ["GC"],
        },
        "timeframe": "5m",
        "factors": {
            "ema_8": {
                "type": "ema",
                "params": {"length": 8, "source": "close"},
            }
        },
        "trade": {
            "long": {
                "entry": {
                    "order": {"type": "market"},
                    "condition": {
                        "cmp": {
                            "left": {"ref": "price.close"},
                            "op": "gt",
                            "right": {"ref": "ema_8"},
                        }
                    },
                },
                "exits": [
                    {
                        "type": "signal_exit",
                        "name": "exit_on_loss_of_trend",
                        "condition": {
                            "cmp": {
                                "left": {"ref": "price.close"},
                                "op": "lt",
                                "right": {"ref": "ema_8"},
                            }
                        },
                        "order": {"type": "market"},
                    }
                ],
                "position_sizing": {"mode": "pct_equity", "pct": 0.1},
            }
        },
    }


@dataclass(slots=True)
class ProbeResult:
    url: str
    ok: bool
    payload: dict[str, Any]


async def _call_strategy_validate(*, url: str, session_id: str) -> ProbeResult:
    async with streamable_http_client(url) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool(
                "strategy_validate_dsl",
                {
                    "dsl_json": json.dumps(_build_probe_dsl()),
                    "session_id": session_id,
                },
            )

    text_payload = ""
    for item in result.content:
        item_text = getattr(item, "text", None)
        if isinstance(item_text, str) and item_text.strip():
            text_payload = item_text.strip()
            break
    if not text_payload:
        raise RuntimeError(f"No text payload returned from MCP URL: {url}")

    payload = json.loads(text_payload)
    return ProbeResult(url=url, ok=bool(payload.get("ok")), payload=payload)


def _render_payload(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


async def _run() -> int:
    args = _build_parser().parse_args()
    settings = get_api_settings()
    primary_url = args.mcp_url.strip() or settings.strategy_mcp_server_url
    compare_url = args.compare_url.strip()

    print(f"runtime_env={settings.runtime_env}")
    print(f"resolved_strategy_mcp_url={settings.strategy_mcp_server_url}")

    primary = await _call_strategy_validate(url=primary_url, session_id=args.session_id)
    print(f"\n[primary] {primary.url}")
    print(_render_payload(primary.payload))

    compare: ProbeResult | None = None
    if compare_url:
        compare = await _call_strategy_validate(url=compare_url, session_id=args.session_id)
        print(f"\n[compare] {compare.url}")
        print(_render_payload(compare.payload))
    elif args.expect_compare_failure:
        print("--expect-compare-failure requires --compare-url", file=sys.stderr)
        return 1

    if not primary.ok:
        return 1
    if compare is not None:
        if args.expect_compare_failure:
            if compare.ok:
                print(
                    "compare URL unexpectedly succeeded while failure was expected",
                    file=sys.stderr,
                )
                return 1
        elif not compare.ok:
            return 1
    return 0


def main() -> int:
    try:
        return asyncio.run(_run())
    except Exception as exc:  # noqa: BLE001
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
