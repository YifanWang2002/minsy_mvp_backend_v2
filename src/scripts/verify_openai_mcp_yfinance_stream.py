#!/usr/bin/env python3
"""Verify a remote MCP server can be called by OpenAI Responses API via stream.

This script:
1) Sends streamed requests to force real market-data MCP tools.
2) Uses `tool_choice` to target each tool directly.
3) Parses stream events and final response output.
4) Prints per-attempt success/failure and a per-tool summary.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from openai import APIError, OpenAI

DEFAULT_SERVER_URL = "https://mcp.minsyai.com/mcp"
DEFAULT_SERVER_LABEL = "market_data"
DEFAULT_MODEL = "gpt-5.2"
DEFAULT_CASE_SET = "smoke"


@dataclass
class AttemptCase:
    tool_name: str
    prompt: str
    params: dict[str, Any]
    tag: str


@dataclass
class AttemptResult:
    tool_name: str
    tag: str
    prompt: str
    params: dict[str, Any]
    ok: bool = False
    reason: str = ""
    response_id: str | None = None
    event_counts: dict[str, int] = field(default_factory=dict)
    called_tools: list[str] = field(default_factory=list)
    mcp_item_statuses: list[str] = field(default_factory=list)
    mcp_item_errors: list[str] = field(default_factory=list)
    tool_payload_ok_values: list[bool] = field(default_factory=list)
    tool_payload_errors: list[str] = field(default_factory=list)
    api_error: str | None = None
    output_text: str = ""

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _build_smoke_cases() -> list[AttemptCase]:
    return [
        AttemptCase(
            tool_name="get_symbol_quote",
            tag="quote_stock_aapl",
            params={"market": "stock", "symbol": "AAPL"},
            prompt=(
                "Call MCP tool get_symbol_quote exactly once with market stock "
                "and symbol AAPL. Then summarize in one line."
            ),
        ),
        AttemptCase(
            tool_name="get_symbol_quote",
            tag="quote_crypto_btcusd",
            params={"market": "crypto", "symbol": "BTCUSD"},
            prompt=(
                "Call MCP tool get_symbol_quote exactly once with market crypto "
                "and symbol BTCUSD. Then summarize in one line."
            ),
        ),
        AttemptCase(
            tool_name="get_symbol_quote",
            tag="quote_forex_eurusd",
            params={"market": "forex", "symbol": "EURUSD"},
            prompt=(
                "Call MCP tool get_symbol_quote exactly once with market forex "
                "and symbol EURUSD. Then summarize in one line."
            ),
        ),
        AttemptCase(
            tool_name="get_symbol_quote",
            tag="quote_futures_es",
            params={"market": "futures", "symbol": "ES"},
            prompt=(
                "Call MCP tool get_symbol_quote exactly once with market futures "
                "and symbol ES. Then summarize in one line."
            ),
        ),
        AttemptCase(
            tool_name="get_symbol_candles",
            tag="candles_stock_aapl",
            params={"market": "stock", "symbol": "AAPL", "period": "1d", "interval": "1d"},
            prompt=(
                "Call MCP tool get_symbol_candles exactly once with market stock, "
                "symbol AAPL, period 1d, interval 1d. Then report number of rows."
            ),
        ),
        AttemptCase(
            tool_name="get_symbol_candles",
            tag="candles_crypto_btcusd",
            params={"market": "crypto", "symbol": "BTCUSD", "period": "1d", "interval": "1d"},
            prompt=(
                "Call MCP tool get_symbol_candles exactly once with market crypto, "
                "symbol BTCUSD, period 1d, interval 1d. Then report number of rows."
            ),
        ),
        AttemptCase(
            tool_name="get_symbol_metadata",
            tag="metadata_forex_eurusd",
            params={"market": "forex", "symbol": "EURUSD"},
            prompt=(
                "Call MCP tool get_symbol_metadata exactly once with market forex "
                "and symbol EURUSD. Then summarize key fields."
            ),
        ),
        AttemptCase(
            tool_name="get_symbol_metadata",
            tag="metadata_futures_es",
            params={"market": "futures", "symbol": "ES"},
            prompt=(
                "Call MCP tool get_symbol_metadata exactly once with market futures "
                "and symbol ES. Then summarize key fields."
            ),
        ),
    ]


def _build_full_matrix_cases() -> list[AttemptCase]:
    symbols_by_market: dict[str, list[str]] = {
        "stock": ["AAPL", "MSFT"],
        "crypto": ["BTCUSD", "ETHUSD"],
        "forex": ["EURUSD", "USDJPY"],
        "futures": ["ES", "NQ"],
    }

    cases: list[AttemptCase] = []
    for market, symbols in symbols_by_market.items():
        for symbol in symbols:
            symbol_tag = symbol.lower()

            cases.append(
                AttemptCase(
                    tool_name="get_symbol_quote",
                    tag=f"quote_{market}_{symbol_tag}",
                    params={"market": market, "symbol": symbol},
                    prompt=(
                        "Call MCP tool get_symbol_quote exactly once with market "
                        f"{market} and symbol {symbol}. Then summarize in one line."
                    ),
                )
            )

            cases.append(
                AttemptCase(
                    tool_name="get_symbol_candles",
                    tag=f"candles_{market}_{symbol_tag}",
                    params={
                        "market": market,
                        "symbol": symbol,
                        "period": "1d",
                        "interval": "1d",
                    },
                    prompt=(
                        "Call MCP tool get_symbol_candles exactly once with market "
                        f"{market}, symbol {symbol}, period 1d, interval 1d. "
                        "Then report number of rows."
                    ),
                )
            )

            cases.append(
                AttemptCase(
                    tool_name="get_symbol_metadata",
                    tag=f"metadata_{market}_{symbol_tag}",
                    params={"market": market, "symbol": symbol},
                    prompt=(
                        "Call MCP tool get_symbol_metadata exactly once with market "
                        f"{market} and symbol {symbol}. Then summarize key fields."
                    ),
                )
            )
    return cases


def build_cases(case_set: str) -> list[AttemptCase]:
    case_set_key = case_set.strip().lower()
    if case_set_key == "smoke":
        return _build_smoke_cases()
    if case_set_key == "full24":
        return _build_full_matrix_cases()
    raise ValueError(f"Unsupported case set: {case_set}")


def _dump_model(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json", exclude_none=True)
    if isinstance(obj, dict):
        return obj
    return {"value": str(obj)}


def run_attempt(
    client: OpenAI,
    *,
    model: str,
    server_label: str,
    server_url: str,
    require_approval: str,
    authorization: str | None,
    headers: dict[str, str],
    case: AttemptCase,
    timeout_seconds: float,
) -> AttemptResult:
    result = AttemptResult(
        tool_name=case.tool_name,
        tag=case.tag,
        prompt=case.prompt,
        params=case.params,
    )
    event_counter: Counter[str] = Counter()

    tool_def: dict[str, Any] = {
        "type": "mcp",
        "server_label": server_label,
        "server_url": server_url,
        "require_approval": require_approval,
    }
    if authorization:
        tool_def["authorization"] = authorization
    if headers:
        tool_def["headers"] = headers

    try:
        with client.responses.stream(
            model=model,
            input=case.prompt,
            tools=[tool_def],
            tool_choice={
                "type": "mcp",
                "server_label": server_label,
                "name": case.tool_name,
            },
            timeout=timeout_seconds,
        ) as stream:
            for event in stream:
                event_type = getattr(event, "type", "unknown")
                event_counter[event_type] += 1

                if event_type == "response.output_text.delta":
                    delta = getattr(event, "delta", "")
                    if delta:
                        result.output_text += delta
                    continue

                if event_type != "response.output_item.added":
                    continue

                event_payload = _dump_model(event)
                item = event_payload.get("item") or {}
                item_type = item.get("type")

                if item_type == "mcp_call":
                    name = item.get("name")
                    if isinstance(name, str):
                        result.called_tools.append(name)
                    status = item.get("status")
                    if isinstance(status, str):
                        result.mcp_item_statuses.append(status)
                    error_text = item.get("error")
                    if isinstance(error_text, str) and error_text:
                        result.mcp_item_errors.append(error_text)

                if item_type == "mcp_list_tools":
                    error_text = item.get("error")
                    if isinstance(error_text, str) and error_text:
                        result.mcp_item_errors.append(error_text)

            final_response = stream.get_final_response()
            result.response_id = final_response.id

            for output_item in final_response.output or []:
                output_payload = _dump_model(output_item)
                if output_payload.get("type") != "mcp_call":
                    continue
                name = output_payload.get("name")
                if isinstance(name, str):
                    result.called_tools.append(name)
                status = output_payload.get("status")
                if isinstance(status, str):
                    result.mcp_item_statuses.append(status)
                error_text = output_payload.get("error")
                if isinstance(error_text, str) and error_text:
                    result.mcp_item_errors.append(error_text)
                if name == case.tool_name:
                    output_raw = output_payload.get("output")
                    if isinstance(output_raw, str):
                        try:
                            parsed_output = json.loads(output_raw)
                        except json.JSONDecodeError:
                            parsed_output = None
                        if isinstance(parsed_output, dict):
                            payload_ok = parsed_output.get("ok")
                            if isinstance(payload_ok, bool):
                                result.tool_payload_ok_values.append(payload_ok)
                                if not payload_ok:
                                    payload_error = parsed_output.get("error")
                                    if isinstance(payload_error, str) and payload_error:
                                        result.tool_payload_errors.append(payload_error)

    except APIError as exc:
        result.api_error = f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # noqa: BLE001
        result.api_error = f"{type(exc).__name__}: {exc}"
    finally:
        result.event_counts = dict(event_counter)

    called_expected = case.tool_name in result.called_tools
    has_completed = "completed" in result.mcp_item_statuses
    has_fail_event = any(
        key.endswith(".failed") and count > 0 for key, count in result.event_counts.items()
    )
    has_api_error = bool(result.api_error)
    has_item_error = len(result.mcp_item_errors) > 0
    has_tool_payload_failure = any(not ok for ok in result.tool_payload_ok_values)

    if (
        called_expected
        and has_completed
        and not has_fail_event
        and not has_api_error
        and not has_item_error
        and not has_tool_payload_failure
    ):
        result.ok = True
        result.reason = "Tool call completed successfully."
    else:
        result.ok = False
        reasons: list[str] = []
        if not called_expected:
            reasons.append("expected tool not observed in mcp_call items")
        if not has_completed:
            reasons.append("no completed mcp_call status observed")
        if has_fail_event:
            failed_events = ", ".join(
                sorted([k for k in result.event_counts if k.endswith(".failed")])
            )
            reasons.append(f"failed stream event(s): {failed_events}")
        if has_item_error:
            reasons.append(f"mcp item error(s): {result.mcp_item_errors[-1]}")
        if has_tool_payload_failure:
            payload_error = (
                result.tool_payload_errors[-1]
                if result.tool_payload_errors
                else "tool returned ok=false"
            )
            reasons.append(f"tool payload error: {payload_error}")
        if has_api_error:
            reasons.append(f"API error: {result.api_error}")
        result.reason = "; ".join(reasons) if reasons else "Unknown failure"

    return result


def parse_headers(raw_headers: list[str]) -> dict[str, str]:
    headers: dict[str, str] = {}
    for pair in raw_headers:
        if ":" not in pair:
            raise ValueError(f"Invalid header format (expect 'Key: Value'): {pair}")
        key, value = pair.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Invalid header key in: {pair}")
        headers[key] = value
    return headers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stream-verify OpenAI remote MCP calls against market_data server."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model name.")
    parser.add_argument(
        "--case-set",
        default=DEFAULT_CASE_SET,
        choices=("smoke", "full24"),
        help="Test case set. 'smoke'=8 cases, 'full24'=4 markets x 2 symbols x 3 tools.",
    )
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL, help="MCP server URL.")
    parser.add_argument("--server-label", default=DEFAULT_SERVER_LABEL, help="MCP server label.")
    parser.add_argument(
        "--require-approval",
        default="never",
        help="MCP require_approval value. Usually 'never' or 'always'.",
    )
    parser.add_argument(
        "--authorization",
        default=os.getenv("MCP_AUTHORIZATION"),
        help="Optional MCP authorization token (or set MCP_AUTHORIZATION env).",
    )
    parser.add_argument(
        "--header",
        action="append",
        default=[],
        help="Optional HTTP header for MCP server. Repeatable, format: 'Key: Value'.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=90.0,
        help="Per-attempt request timeout in seconds.",
    )
    parser.add_argument(
        "--json-report",
        default="",
        help="Optional path to write full JSON report.",
    )
    parser.add_argument(
        "--always-zero",
        action="store_true",
        help="Always exit 0 even when some attempts fail.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        headers = parse_headers(args.header)
    except ValueError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2

    client = OpenAI()
    cases = build_cases(args.case_set)

    print(
        f"[info] Start MCP verification at {datetime.now(UTC).isoformat()} "
        f"(model={args.model}, server_label={args.server_label}, server_url={args.server_url})"
    )
    print(f"[info] Planned attempts: {len(cases)}\n")

    results: list[AttemptResult] = []
    for idx, case in enumerate(cases, start=1):
        print(
            f"[{idx:02d}/{len(cases):02d}] {case.tool_name} ({case.tag}) "
            f"params={json.dumps(case.params, ensure_ascii=False)}"
        )
        attempt = run_attempt(
            client,
            model=args.model,
            server_label=args.server_label,
            server_url=args.server_url,
            require_approval=args.require_approval,
            authorization=args.authorization,
            headers=headers,
            case=case,
            timeout_seconds=args.timeout_seconds,
        )
        results.append(attempt)

        status = "PASS" if attempt.ok else "FAIL"
        print(f"  -> {status}: {attempt.reason}")
        if attempt.event_counts:
            print(f"  -> events: {json.dumps(attempt.event_counts, ensure_ascii=False)}")
        if attempt.called_tools:
            print(f"  -> called_tools: {attempt.called_tools}")
        if attempt.mcp_item_statuses:
            print(f"  -> mcp_statuses: {attempt.mcp_item_statuses}")
        if attempt.api_error:
            print(f"  -> api_error: {attempt.api_error}")
        print()

    by_tool: dict[str, list[AttemptResult]] = defaultdict(list)
    for result in results:
        by_tool[result.tool_name].append(result)

    print("=== Summary by tool ===")
    total_ok = 0
    total_n = len(results)
    tool_order: list[str] = []
    for case in cases:
        if case.tool_name not in tool_order:
            tool_order.append(case.tool_name)
    for tool_name in tool_order:
        bucket = by_tool.get(tool_name, [])
        ok_count = sum(1 for item in bucket if item.ok)
        total_ok += ok_count
        print(f"- {tool_name}: {ok_count}/{len(bucket)} passed")
    print(f"\n=== Overall ===\n- Passed: {total_ok}/{total_n}")

    if args.json_report:
        report = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "model": args.model,
            "case_set": args.case_set,
            "server_label": args.server_label,
            "server_url": args.server_url,
            "results": [item.to_json() for item in results],
        }
        with open(args.json_report, "w", encoding="utf-8") as fp:
            json.dump(report, fp, ensure_ascii=False, indent=2)
        print(f"- JSON report written: {args.json_report}")

    if args.always_zero:
        return 0
    return 0 if total_ok == total_n else 1


if __name__ == "__main__":
    raise SystemExit(main())
