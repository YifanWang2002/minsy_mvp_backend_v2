"""Debug probe for frontend-like backend stream vs direct OpenAI endpoint.

Runs three checks in one command:
1) Frontend-like multi-turn flow through backend SSE endpoints.
2) Direct OpenAI stream with a light reasoning prompt.
3) Direct OpenAI stream with MCP tool_choice (for tool-name mismatch diagnosis).

Usage:
    set -a; source .env; set +a
    uv run python -m src.scripts.debug_stream_path_probe --model gpt-5.2
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import httpx
from openai import OpenAI

DEFAULT_BASE_URL = "http://127.0.0.1:8000/api/v1"
DEFAULT_MODEL = "gpt-5.2"
DEFAULT_MCP_SERVER_URL = "http://127.0.0.1:8111/mcp"

_AGENT_UI_PATTERN = re.compile(
    r"<\s*AGENT_UI_JSON\s*>([\s\S]*?)</\s*AGENT_UI_JSON\s*>",
    flags=re.IGNORECASE,
)

_PREFERRED_IDS: dict[str, str] = {
    "trading_years_bucket": "years_5_plus",
    "risk_tolerance": "aggressive",
    "return_expectation": "high_growth",
    "target_market": "us_stocks",
    "target_instrument": "SPY",
    "opportunity_frequency_bucket": "few_per_week",
    "holding_period_bucket": "swing_days",
}


@dataclass
class ChoicePrompt:
    choice_id: str
    options: list[dict[str, str]]


@dataclass
class BackendTurnTrace:
    turn: int
    sent_message: str
    phase: str
    stream_error: str | None
    text_preview: str
    openai_stream_error: dict[str, Any] | None
    mcp_events: dict[str, int]


@dataclass
class DirectProbeResult:
    name: str
    ok: bool
    event_counts: dict[str, int]
    text_preview: str
    error: str | None = None


def _iter_sse_events(response: httpx.Response):
    event_name = "message"
    data_lines: list[str] = []
    for line in response.iter_lines():
        if line == "":
            if data_lines:
                yield event_name, "\n".join(data_lines)
            event_name = "message"
            data_lines = []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_name = line.removeprefix("event:").strip() or "message"
            continue
        if line.startswith("data:"):
            data_lines.append(line.removeprefix("data:").lstrip())
    if data_lines:
        yield event_name, "\n".join(data_lines)


def _parse_choice_prompt(payload: Any) -> ChoicePrompt | None:
    if not isinstance(payload, dict):
        return None
    if payload.get("type") != "choice_prompt":
        return None
    choice_id = payload.get("choice_id")
    options_raw = payload.get("options")
    if not isinstance(choice_id, str) or not choice_id.strip():
        return None
    if not isinstance(options_raw, list):
        return None
    options: list[dict[str, str]] = []
    for item in options_raw:
        if not isinstance(item, dict):
            continue
        option_id = item.get("id")
        label = item.get("label")
        if isinstance(option_id, str) and option_id.strip() and isinstance(label, str) and label.strip():
            options.append({"id": option_id.strip(), "label": label.strip()})
    if len(options) < 2:
        return None
    return ChoicePrompt(choice_id=choice_id.strip(), options=options)


def _extract_choice_from_text(full_text: str) -> ChoicePrompt | None:
    for raw in _AGENT_UI_PATTERN.findall(full_text):
        try:
            payload = json.loads(raw.strip())
        except json.JSONDecodeError:
            continue
        prompt = _parse_choice_prompt(payload)
        if prompt is not None:
            return prompt
    return None


def _register(client: httpx.Client, base_url: str) -> tuple[str, str]:
    email = f"stream_diag_{uuid4().hex[:10]}@example.com"
    resp = client.post(
        f"{base_url}/auth/register",
        json={"email": email, "password": "probe123456", "name": "Stream Diag"},
        timeout=30.0,
    )
    if resp.status_code != 201:
        raise RuntimeError(f"register failed: {resp.status_code} {resp.text}")
    token = resp.json().get("access_token")
    if not isinstance(token, str) or not token:
        raise RuntimeError("register failed: missing access_token")
    return email, token


def _create_thread(client: httpx.Client, base_url: str, headers: dict[str, str]) -> str:
    resp = client.post(
        f"{base_url}/chat/new-thread",
        headers=headers,
        json={"metadata": {}},
        timeout=30.0,
    )
    if resp.status_code != 201:
        raise RuntimeError(f"new-thread failed: {resp.status_code} {resp.text}")
    session_id = str(resp.json().get("session_id") or "")
    if not session_id:
        raise RuntimeError("new-thread returned empty session_id")
    return session_id


def _run_backend_flow(
    *,
    base_url: str,
    language: str,
    max_turns: int,
    timeout_seconds: float,
) -> tuple[str, list[BackendTurnTrace]]:
    traces: list[BackendTurnTrace] = []
    with httpx.Client(timeout=30.0, trust_env=False) as client:
        _, token = _register(client, base_url)
        headers = {"Authorization": f"Bearer {token}"}
        session_id = _create_thread(client, base_url, headers)

        next_message = "Hi"
        latest_prompt: ChoicePrompt | None = None

        for turn in range(1, max_turns + 1):
            if latest_prompt is not None:
                preferred = _PREFERRED_IDS.get(latest_prompt.choice_id)
                selected_label = None
                if preferred is not None:
                    selected_label = next(
                        (opt["label"] for opt in latest_prompt.options if opt["id"] == preferred),
                        None,
                    )
                if selected_label is None and latest_prompt.options:
                    selected_label = latest_prompt.options[0]["label"]
                next_message = selected_label or "Continue"

            assistant_text = ""
            done_payload: dict[str, Any] = {}
            mcp_events: Counter[str] = Counter()
            latest_prompt = None
            openai_stream_error: dict[str, Any] | None = None

            with client.stream(
                "POST",
                f"{base_url}/chat/send-openai-stream",
                headers=headers,
                params={"language": language},
                json={"session_id": session_id, "message": next_message},
                timeout=timeout_seconds,
            ) as resp:
                if resp.status_code != 200:
                    body = resp.read().decode("utf-8", errors="replace")
                    raise RuntimeError(f"turn {turn} failed: {resp.status_code} {body}")

                for _, raw_data in _iter_sse_events(resp):
                    try:
                        payload = json.loads(raw_data)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(payload, dict):
                        continue

                    event_type = payload.get("type")
                    if event_type == "text_delta":
                        delta = payload.get("delta")
                        if isinstance(delta, str):
                            assistant_text += delta
                        continue

                    if event_type == "genui":
                        prompt = _parse_choice_prompt(payload.get("payload"))
                        if prompt is not None:
                            latest_prompt = prompt
                        continue

                    if event_type == "mcp_event":
                        openai_type = payload.get("openai_type")
                        if isinstance(openai_type, str) and openai_type:
                            mcp_events[openai_type] += 1
                        continue

                    if event_type == "openai_event" and payload.get("openai_type") == "response.stream_error":
                        raw_error = payload.get("payload")
                        if isinstance(raw_error, dict):
                            openai_stream_error = raw_error.get("error")
                        continue

                    if event_type == "done":
                        done_payload = payload
                        continue

            if latest_prompt is None:
                latest_prompt = _extract_choice_from_text(assistant_text)

            phase = str(done_payload.get("phase") or "?")
            stream_error = done_payload.get("stream_error")
            stream_error_text = stream_error if isinstance(stream_error, str) and stream_error else None
            text_preview = assistant_text.replace("\n", " ").strip()[:220]

            traces.append(
                BackendTurnTrace(
                    turn=turn,
                    sent_message=next_message,
                    phase=phase,
                    stream_error=stream_error_text,
                    text_preview=text_preview,
                    openai_stream_error=openai_stream_error,
                    mcp_events=dict(mcp_events),
                )
            )

            if phase == "strategy":
                break

            if latest_prompt is None and phase == "pre_strategy":
                next_message = "SPY"
            elif latest_prompt is None:
                next_message = "Continue"

        return session_id, traces


def _run_direct_reasoning_probe(model: str) -> DirectProbeResult:
    client = OpenAI()
    question = (
        "A portfolio rises 12% in year 1 and falls 10% in year 2. "
        "Starting from 100, what is the ending value and net return? "
        "Answer briefly with calculation."
    )
    event_counts: Counter[str] = Counter()
    text = ""
    try:
        with client.responses.stream(model=model, input=question, timeout=90) as stream:
            for event in stream:
                event_type = getattr(event, "type", "unknown")
                event_counts[event_type] += 1
                if event_type == "response.output_text.delta":
                    delta = getattr(event, "delta", "")
                    if delta:
                        text += delta
            stream.get_final_response()
        return DirectProbeResult(
            name="direct_reasoning",
            ok=bool(text.strip()),
            event_counts=dict(event_counts),
            text_preview=text.replace("\n", " ").strip()[:220],
        )
    except Exception as exc:  # noqa: BLE001
        return DirectProbeResult(
            name="direct_reasoning",
            ok=False,
            event_counts=dict(event_counts),
            text_preview=text.replace("\n", " ").strip()[:220],
            error=f"{type(exc).__name__}: {exc}",
        )


def _run_direct_mcp_probe(
    *,
    model: str,
    mcp_server_url: str,
    tool_choice_name: str,
) -> DirectProbeResult:
    client = OpenAI()
    prompt = (
        f"Call MCP tool `{tool_choice_name}` exactly once for symbol SPY. "
        "Then summarize the result in one sentence."
    )
    event_counts: Counter[str] = Counter()
    text = ""
    try:
        with client.responses.stream(
            model=model,
            input=prompt,
            tools=[
                {
                    "type": "mcp",
                    "server_label": "market_data",
                    "server_url": mcp_server_url,
                    "allowed_tools": [tool_choice_name, "get_symbol_quote"],
                    "require_approval": "never",
                }
            ],
            tool_choice={
                "type": "mcp",
                "server_label": "market_data",
                "name": tool_choice_name,
            },
            timeout=120,
        ) as stream:
            for event in stream:
                event_type = getattr(event, "type", "unknown")
                event_counts[event_type] += 1
                if event_type == "response.output_text.delta":
                    delta = getattr(event, "delta", "")
                    if delta:
                        text += delta
            stream.get_final_response()
        return DirectProbeResult(
            name=f"direct_mcp_tool_choice_{tool_choice_name}",
            ok=True,
            event_counts=dict(event_counts),
            text_preview=text.replace("\n", " ").strip()[:220],
        )
    except Exception as exc:  # noqa: BLE001
        return DirectProbeResult(
            name=f"direct_mcp_tool_choice_{tool_choice_name}",
            ok=False,
            event_counts=dict(event_counts),
            text_preview=text.replace("\n", " ").strip()[:220],
            error=f"{type(exc).__name__}: {exc}",
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe backend SSE vs direct OpenAI stream path.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="API base URL")
    parser.add_argument("--language", default="en", help="ISO language code")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model for direct probes")
    parser.add_argument("--max-turns", type=int, default=10, help="Max backend turns")
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=180.0,
        help="Per-turn backend stream timeout",
    )
    parser.add_argument(
        "--mcp-server-url",
        default=os.getenv("MCP_SERVER_URL", DEFAULT_MCP_SERVER_URL),
        help="MCP server URL used in direct MCP probe",
    )
    parser.add_argument(
        "--mcp-tool-choice",
        default="check_symbol_available",
        help="Tool name to force in direct MCP probe",
    )
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("[FAIL] OPENAI_API_KEY is missing in environment.")
        return 2

    print(f"[INFO] backend_base_url={args.base_url.rstrip('/')}")
    print(f"[INFO] direct_model={args.model}")
    print(f"[INFO] direct_mcp_server_url={args.mcp_server_url}")
    print(f"[INFO] direct_mcp_tool_choice={args.mcp_tool_choice}")

    session_id, backend_traces = _run_backend_flow(
        base_url=args.base_url.rstrip("/"),
        language=args.language,
        max_turns=args.max_turns,
        timeout_seconds=args.timeout_seconds,
    )
    print(f"[INFO] backend_session_id={session_id}")
    for trace in backend_traces:
        print(
            f"[BACKEND TURN {trace.turn}] phase={trace.phase} "
            f"stream_error={trace.stream_error!r} msg={trace.sent_message!r}"
        )
        if trace.mcp_events:
            print(f"  mcp_events={trace.mcp_events}")
        if trace.openai_stream_error:
            print(f"  openai_stream_error={trace.openai_stream_error}")
        print(f"  text_preview={trace.text_preview!r}")

    direct_reasoning = _run_direct_reasoning_probe(args.model)
    print(
        f"[DIRECT] {direct_reasoning.name} ok={direct_reasoning.ok} "
        f"events={direct_reasoning.event_counts}"
    )
    if direct_reasoning.error:
        print(f"  error={direct_reasoning.error}")
    print(f"  text_preview={direct_reasoning.text_preview!r}")

    direct_mcp = _run_direct_mcp_probe(
        model=args.model,
        mcp_server_url=args.mcp_server_url,
        tool_choice_name=args.mcp_tool_choice,
    )
    print(
        f"[DIRECT] {direct_mcp.name} ok={direct_mcp.ok} "
        f"events={direct_mcp.event_counts}"
    )
    if direct_mcp.error:
        print(f"  error={direct_mcp.error}")
    print(f"  text_preview={direct_mcp.text_preview!r}")

    backend_errors = [item for item in backend_traces if item.stream_error]
    print(
        f"[SUMMARY] backend_turns={len(backend_traces)} "
        f"backend_stream_error_turns={len(backend_errors)} "
        f"direct_reasoning_ok={direct_reasoning.ok} "
        f"direct_mcp_ok={direct_mcp.ok}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
