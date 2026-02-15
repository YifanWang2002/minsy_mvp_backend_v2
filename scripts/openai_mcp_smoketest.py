#!/usr/bin/env python3
"""Smoke test MCP access through OpenAI Responses API.

What it validates:
1) OpenAI can fetch MCP tools list from the target `server_url`.
2) OpenAI can call at least one simple MCP tool.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import APIError, OpenAI

DEFAULT_SERVER_URL = "https://dev.minsyai.com/mcp"
DEFAULT_MODEL = "gpt-5.2"


@dataclass
class McpCallRecord:
    call_id: str
    name: str = ""
    status: str = ""
    error: str | None = None
    output_preview: str | None = None


@dataclass
class ProbeResult:
    phase: str
    response_id: str | None = None
    response_text: str = ""
    mcp_list_tools_seen: bool = False
    mcp_list_tools_status: str | None = None
    mcp_list_tools_error: str | None = None
    discovered_tools: list[str] = field(default_factory=list)
    mcp_calls: list[McpCallRecord] = field(default_factory=list)
    api_error: str | None = None
    api_error_body: Any = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify MCP list-tools and tool-call via OpenAI Responses API."
    )
    parser.add_argument(
        "--server-url",
        default=os.getenv("MCP_SERVER_URL", DEFAULT_SERVER_URL),
        help="MCP endpoint URL. Default: env MCP_SERVER_URL or dev URL.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_RESPONSE_MODEL", DEFAULT_MODEL),
        help="Responses model. Default: env OPENAI_RESPONSE_MODEL or gpt-5.2.",
    )
    parser.add_argument(
        "--server-label",
        default="remote_mcp",
        help="MCP server label in OpenAI tool config.",
    )
    parser.add_argument(
        "--show-all-events",
        action="store_true",
        help="Print all stream events (default only prints MCP-related events).",
    )
    parser.add_argument(
        "--allowed-tool",
        action="append",
        default=[],
        help="Optional allow-list tool name. Can be passed multiple times.",
    )
    return parser.parse_args()


def _load_env() -> None:
    project_root = Path(__file__).resolve().parents[1]
    env_file = project_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value

    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}

    if isinstance(value, list | tuple | set):
        return [_to_jsonable(v) for v in value]

    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump(mode="json", exclude_none=True, warnings=False)
            return _to_jsonable(dumped)
        except Exception:  # noqa: BLE001
            pass

    return str(value)


def _coerce_event_payload(event: Any) -> dict[str, Any]:
    payload = _to_jsonable(event)
    if isinstance(payload, dict):
        return payload
    return {"type": str(getattr(event, "type", "unknown"))}


def _truncate(value: Any, max_len: int = 280) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if len(text) <= max_len:
        return text
    return text[:max_len] + "...<truncated>"


def _print_event(*, payload: dict[str, Any], show_all_events: bool) -> None:
    event_type = str(payload.get("type", "unknown"))
    if not show_all_events and "mcp" not in event_type.lower():
        return
    print(f"[event] {event_type}")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _extract_tool_names(item: dict[str, Any]) -> list[str]:
    output: list[str] = []
    tools = item.get("tools")
    if not isinstance(tools, list):
        return output
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name")
        if isinstance(name, str) and name.strip():
            output.append(name.strip())
    return output


def _upsert_mcp_call(
    calls_by_id: dict[str, McpCallRecord],
    *,
    item: dict[str, Any],
) -> None:
    call_id_raw = item.get("id")
    call_id = str(call_id_raw).strip() if call_id_raw is not None else ""
    if not call_id:
        return

    record = calls_by_id.get(call_id)
    if record is None:
        record = McpCallRecord(call_id=call_id)
        calls_by_id[call_id] = record

    name = item.get("name")
    if isinstance(name, str) and name.strip():
        record.name = name.strip()

    status = item.get("status")
    if isinstance(status, str) and status.strip():
        record.status = status.strip()

    error = _truncate(item.get("error"), max_len=600)
    if error:
        record.error = error

    output_preview = _truncate(item.get("output"), max_len=350)
    if output_preview:
        record.output_preview = output_preview


def _run_probe(
    *,
    client: OpenAI,
    phase: str,
    model: str,
    server_label: str,
    server_url: str,
    prompt: str,
    tool_choice: dict[str, str],
    show_all_events: bool,
    allowed_tools: list[str] | None,
) -> ProbeResult:
    result = ProbeResult(phase=phase)
    calls_by_id: dict[str, McpCallRecord] = {}

    tool_def: dict[str, Any] = {
        "type": "mcp",
        "server_label": server_label,
        "server_url": server_url,
        "require_approval": "never",
    }
    if allowed_tools:
        tool_def["allowed_tools"] = allowed_tools

    stream_kwargs: dict[str, Any] = {
        "model": model,
        "input": prompt,
        "tools": [tool_def],
        "tool_choice": tool_choice,
        "reasoning": {"effort": "low"},
    }

    try:
        with client.responses.stream(**stream_kwargs) as stream:
            for event in stream:
                payload = _coerce_event_payload(event)
                _print_event(payload=payload, show_all_events=show_all_events)

                event_type = str(payload.get("type", ""))
                if "mcp_list_tools" in event_type:
                    result.mcp_list_tools_seen = True
                    if event_type.endswith(".completed"):
                        result.mcp_list_tools_status = "completed"
                    elif event_type.endswith(".failed"):
                        result.mcp_list_tools_status = "failed"

                if "mcp_call" in event_type and event_type.endswith(".failed"):
                    # Detailed error is usually attached in output_item.done's item.error.
                    pass

                if event_type in {
                    "response.output_item.added",
                    "response.output_item.done",
                }:
                    item = payload.get("item")
                    if not isinstance(item, dict):
                        continue

                    item_type = str(item.get("type", "")).strip().lower()
                    if item_type == "mcp_list_tools":
                        result.mcp_list_tools_seen = True
                        discovered = _extract_tool_names(item)
                        if discovered:
                            merged = set(result.discovered_tools)
                            merged.update(discovered)
                            result.discovered_tools = sorted(merged)
                        error_text = _truncate(item.get("error"), max_len=700)
                        if error_text:
                            result.mcp_list_tools_status = "failed"
                            result.mcp_list_tools_error = error_text
                        elif result.mcp_list_tools_status is None:
                            result.mcp_list_tools_status = "completed"

                    if item_type == "mcp_call":
                        _upsert_mcp_call(calls_by_id, item=item)

            final_response = stream.get_final_response()
            result.response_id = getattr(final_response, "id", None)
            final_text = getattr(final_response, "output_text", "")
            result.response_text = final_text.strip() if isinstance(final_text, str) else ""

    except APIError as exc:
        result.api_error = f"{type(exc).__name__}: {exc}"
        result.api_error_body = _to_jsonable(getattr(exc, "body", None))

    result.mcp_calls = sorted(calls_by_id.values(), key=lambda record: record.call_id)
    return result


def _preferred_simple_tool_prompt(tool_name: str) -> str:
    prompts: dict[str, str] = {
        "get_indicator_catalog": (
            "Call MCP tool get_indicator_catalog with category as empty string. "
            "Then answer in one sentence."
        ),
        "get_indicator_detail": (
            "Call MCP tool get_indicator_detail with indicator='ema'. "
            "Then answer in one sentence."
        ),
        "check_symbol_available": (
            "Call MCP tool check_symbol_available with symbol='SPY' and market='us_stocks'. "
            "Then answer in one sentence."
        ),
        "get_available_symbols": (
            "Call MCP tool get_available_symbols with market='us_stocks'. "
            "Then answer in one sentence."
        ),
    }
    return prompts.get(
        tool_name,
        (
            f"Call MCP tool {tool_name} with the minimal valid input for a read-only "
            "request. Then answer in one sentence."
        ),
    )


def _select_simple_tool(discovered_tools: list[str]) -> str | None:
    preference = [
        "get_indicator_catalog",
        "check_symbol_available",
        "get_available_symbols",
        "get_indicator_detail",
    ]
    discovered_set = set(discovered_tools)
    for name in preference:
        if name in discovered_set:
            return name
    return discovered_tools[0] if discovered_tools else None


def _print_phase_summary(result: ProbeResult) -> None:
    print(f"\n=== {result.phase} summary ===")
    print(f"response_id: {result.response_id or 'N/A'}")
    print(f"mcp_list_tools_seen: {result.mcp_list_tools_seen}")
    print(f"mcp_list_tools_status: {result.mcp_list_tools_status or 'unknown'}")

    if result.discovered_tools:
        print(f"discovered_tools_count: {len(result.discovered_tools)}")
        print("discovered_tools:")
        for name in result.discovered_tools:
            print(f"  - {name}")
    else:
        print("discovered_tools_count: 0")

    if result.mcp_list_tools_error:
        print(f"mcp_list_tools_error: {result.mcp_list_tools_error}")

    if result.mcp_calls:
        print("mcp_calls:")
        for call in result.mcp_calls:
            status = call.status or "unknown"
            print(f"  - id={call.call_id} name={call.name or 'N/A'} status={status}")
            if call.error:
                print(f"    error={call.error}")
            if call.output_preview:
                print(f"    output_preview={call.output_preview}")
    else:
        print("mcp_calls: []")

    if result.response_text:
        print(f"response_text: {result.response_text}")

    if result.api_error:
        print(f"api_error: {result.api_error}")
        if result.api_error_body is not None:
            body = json.dumps(result.api_error_body, ensure_ascii=False)
            print(f"api_error_body: {_truncate(body, max_len=1600)}")


def main() -> int:
    _load_env()
    args = _parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY is missing.")
        return 2

    client = OpenAI(api_key=api_key)

    # Phase 1: verify list-tools path and attempt one automatic MCP call.
    phase1_prompt = (
        "Call one simple read-only MCP tool. Prefer tools like "
        "get_indicator_catalog or check_symbol_available if available. "
        "After tool call, reply briefly."
    )
    phase1 = _run_probe(
        client=client,
        phase="phase_1_discovery",
        model=args.model,
        server_label=args.server_label,
        server_url=args.server_url,
        prompt=phase1_prompt,
        tool_choice={"type": "mcp", "server_label": args.server_label},
        show_all_events=args.show_all_events,
        allowed_tools=args.allowed_tool or None,
    )
    _print_phase_summary(phase1)

    selected_tool = _select_simple_tool(phase1.discovered_tools)
    phase2: ProbeResult | None = None

    # Phase 2: if we found tools, force call one simple tool for deterministic validation.
    if selected_tool is not None:
        phase2_prompt = _preferred_simple_tool_prompt(selected_tool)
        phase2 = _run_probe(
            client=client,
            phase=f"phase_2_forced_call:{selected_tool}",
            model=args.model,
            server_label=args.server_label,
            server_url=args.server_url,
            prompt=phase2_prompt,
            tool_choice={
                "type": "mcp",
                "server_label": args.server_label,
                "name": selected_tool,
            },
            show_all_events=args.show_all_events,
            allowed_tools=[selected_tool],
        )
        _print_phase_summary(phase2)
    else:
        print("\nSkip phase 2: no tools discovered in phase 1.")

    results = [phase1] + ([phase2] if phase2 else [])
    list_ok = any(
        result is not None
        and result.mcp_list_tools_status == "completed"
        and len(result.discovered_tools) > 0
        for result in results
    )
    call_ok = any(
        result is not None
        and any((call.status == "completed" and not call.error) for call in result.mcp_calls)
        for result in results
    )

    print("\n=== final verdict ===")
    print(f"list_tools_ok: {list_ok}")
    print(f"tool_call_ok: {call_ok}")

    return 0 if list_ok and call_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
