"""Frontend-behavior probe for full KYC -> pre_strategy flow.

This script simulates the Flutter chat behavior:
1) register
2) create thread (/chat/new-thread)
3) send streamed turns (/chat/send-openai-stream) with session_id
4) pick answers from the latest choice_prompt options by label
5) verify symbol turn includes MCP activity + chart + next choice

Usage:
    uv run python -m src.scripts.pre_strategy_frontend_flow_probe
    uv run python -m src.scripts.pre_strategy_frontend_flow_probe --language zh
    uv run python -m src.scripts.pre_strategy_frontend_flow_probe --strict
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:8000/api/v1"

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
    question: str
    options: list[dict[str, str]]


@dataclass
class TurnResult:
    turn: int
    sent_message: str
    phase: str = "?"
    kyc_status: str = "?"
    session_id: str | None = None
    done_seen: bool = False
    genui_types: list[str] = field(default_factory=list)
    tagged_ui_types: list[str] = field(default_factory=list)
    mcp_openai_types: Counter[str] = field(default_factory=Counter)
    mcp_tools: Counter[str] = field(default_factory=Counter)
    choice_prompt: ChoicePrompt | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def has_chart(self) -> bool:
        return "tradingview_chart" in set(self.genui_types + self.tagged_ui_types)

    @property
    def has_choice(self) -> bool:
        return "choice_prompt" in set(self.genui_types + self.tagged_ui_types)

    @property
    def has_mcp_in_progress(self) -> bool:
        return any(event.endswith(".in_progress") for event in self.mcp_openai_types)

    @property
    def has_mcp_terminal(self) -> bool:
        return any(
            event.endswith(".completed") or event.endswith(".failed")
            for event in self.mcp_openai_types
        )


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


def _extract_tagged_ui_payloads(text: str) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for raw in _AGENT_UI_PATTERN.findall(text):
        try:
            parsed = json.loads(raw.strip())
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            payloads.append(parsed)
    return payloads


def _parse_choice_prompt(payload: Any) -> ChoicePrompt | None:
    if not isinstance(payload, dict):
        return None
    if payload.get("type") != "choice_prompt":
        return None
    choice_id = payload.get("choice_id")
    question = payload.get("question")
    options_raw = payload.get("options")
    if not isinstance(choice_id, str) or not choice_id.strip():
        return None
    if not isinstance(question, str) or not question.strip():
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
    return ChoicePrompt(choice_id=choice_id.strip(), question=question.strip(), options=options)


def _register(client: httpx.Client, base_url: str) -> tuple[str, str]:
    email = f"frontend_probe_{uuid4().hex[:10]}@example.com"
    resp = client.post(
        f"{base_url}/auth/register",
        json={"email": email, "password": "probe123456", "name": "Frontend Probe"},
        timeout=30.0,
    )
    if resp.status_code != 201:
        raise RuntimeError(f"register failed: {resp.status_code} {resp.text}")
    token = resp.json().get("access_token")
    if not isinstance(token, str) or not token:
        raise RuntimeError("register failed: missing access_token")
    return email, token


def _create_thread(client: httpx.Client, base_url: str, headers: dict[str, str]) -> dict[str, Any]:
    resp = client.post(
        f"{base_url}/chat/new-thread",
        headers=headers,
        json={"metadata": {}},
        timeout=30.0,
    )
    if resp.status_code != 201:
        raise RuntimeError(f"new-thread failed: {resp.status_code} {resp.text}")
    return resp.json()


def _send_turn(
    *,
    client: httpx.Client,
    base_url: str,
    headers: dict[str, str],
    language: str,
    turn: int,
    session_id: str,
    message: str,
    timeout_seconds: float,
) -> TurnResult:
    result = TurnResult(turn=turn, sent_message=message, session_id=session_id)
    assistant_text = ""
    latest_choice: ChoicePrompt | None = None
    genui_types: list[str] = []

    with client.stream(
        "POST",
        f"{base_url}/chat/send-openai-stream",
        headers=headers,
        params={"language": language},
        json={"session_id": session_id, "message": message},
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
            if not isinstance(event_type, str):
                continue

            if event_type == "stream_start":
                sid = payload.get("session_id")
                if isinstance(sid, str) and sid:
                    result.session_id = sid
                continue

            if event_type == "text_delta":
                delta = payload.get("delta")
                if isinstance(delta, str):
                    assistant_text += delta
                continue

            if event_type == "genui":
                genui_payload = payload.get("payload")
                if isinstance(genui_payload, dict):
                    genui_type = genui_payload.get("type")
                    if isinstance(genui_type, str) and genui_type.strip():
                        genui_types.append(genui_type.strip())
                    prompt = _parse_choice_prompt(genui_payload)
                    if prompt is not None:
                        latest_choice = prompt
                continue

            if event_type == "mcp_event":
                openai_type = payload.get("openai_type")
                if isinstance(openai_type, str) and openai_type:
                    result.mcp_openai_types[openai_type] += 1
                mcp_payload = payload.get("payload")
                if isinstance(mcp_payload, dict):
                    tool_name = mcp_payload.get("name")
                    if isinstance(tool_name, str) and tool_name.strip():
                        result.mcp_tools[tool_name.strip()] += 1
                continue

            if event_type == "openai_event":
                raw_item = payload.get("payload")
                if isinstance(raw_item, dict):
                    item = raw_item.get("item")
                    if isinstance(item, dict) and item.get("type") == "mcp_call":
                        tool_name = item.get("name")
                        if isinstance(tool_name, str) and tool_name.strip():
                            result.mcp_tools[tool_name.strip()] += 1
                continue

            if event_type == "error":
                msg = payload.get("message")
                if isinstance(msg, str) and msg.strip():
                    result.errors.append(msg.strip())
                continue

            if event_type == "done":
                result.done_seen = True
                phase = payload.get("phase")
                if isinstance(phase, str) and phase:
                    result.phase = phase
                kyc = payload.get("kyc_status")
                if isinstance(kyc, str) and kyc:
                    result.kyc_status = kyc
                sid = payload.get("session_id")
                if isinstance(sid, str) and sid:
                    result.session_id = sid
                continue

    tagged_payloads = _extract_tagged_ui_payloads(assistant_text)
    tagged_types: list[str] = []
    for tagged in tagged_payloads:
        tagged_type = tagged.get("type")
        if isinstance(tagged_type, str) and tagged_type.strip():
            tagged_types.append(tagged_type.strip())
        prompt = _parse_choice_prompt(tagged)
        if prompt is not None:
            latest_choice = prompt

    result.genui_types = _dedupe(genui_types)
    result.tagged_ui_types = _dedupe(tagged_types)
    result.choice_prompt = latest_choice
    return result


def _parse_sse_payloads_from_text(raw_text: str) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for block in raw_text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        for line in block.splitlines():
            if not line.startswith("data:"):
                continue
            content = line.removeprefix("data:").strip()
            if not content:
                continue
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                payloads.append(data)
    return payloads


def _build_turn_result_from_payloads(
    *,
    turn: int,
    sent_message: str,
    session_id: str,
    payloads: list[dict[str, Any]],
) -> TurnResult:
    result = TurnResult(turn=turn, sent_message=sent_message, session_id=session_id)
    assistant_text = ""
    genui_types: list[str] = []
    latest_choice: ChoicePrompt | None = None

    for payload in payloads:
        event_type = payload.get("type")
        if not isinstance(event_type, str):
            continue

        if event_type == "stream_start":
            sid = payload.get("session_id")
            if isinstance(sid, str) and sid:
                result.session_id = sid
            continue

        if event_type == "text_delta":
            delta = payload.get("delta")
            if isinstance(delta, str):
                assistant_text += delta
            continue

        if event_type == "genui":
            genui_payload = payload.get("payload")
            if isinstance(genui_payload, dict):
                genui_type = genui_payload.get("type")
                if isinstance(genui_type, str) and genui_type.strip():
                    genui_types.append(genui_type.strip())
                prompt = _parse_choice_prompt(genui_payload)
                if prompt is not None:
                    latest_choice = prompt
            continue

        if event_type == "mcp_event":
            openai_type = payload.get("openai_type")
            if isinstance(openai_type, str) and openai_type:
                result.mcp_openai_types[openai_type] += 1
            mcp_payload = payload.get("payload")
            if isinstance(mcp_payload, dict):
                tool_name = mcp_payload.get("name")
                if isinstance(tool_name, str) and tool_name.strip():
                    result.mcp_tools[tool_name.strip()] += 1
            continue

        if event_type == "openai_event":
            raw_item = payload.get("payload")
            if isinstance(raw_item, dict):
                item = raw_item.get("item")
                if isinstance(item, dict) and item.get("type") == "mcp_call":
                    tool_name = item.get("name")
                    if isinstance(tool_name, str) and tool_name.strip():
                        result.mcp_tools[tool_name.strip()] += 1
            continue

        if event_type == "error":
            msg = payload.get("message")
            if isinstance(msg, str) and msg.strip():
                result.errors.append(msg.strip())
            continue

        if event_type == "done":
            result.done_seen = True
            phase = payload.get("phase")
            if isinstance(phase, str) and phase:
                result.phase = phase
            kyc = payload.get("kyc_status")
            if isinstance(kyc, str) and kyc:
                result.kyc_status = kyc
            sid = payload.get("session_id")
            if isinstance(sid, str) and sid:
                result.session_id = sid
            continue

    tagged_payloads = _extract_tagged_ui_payloads(assistant_text)
    tagged_types: list[str] = []
    for tagged in tagged_payloads:
        tagged_type = tagged.get("type")
        if isinstance(tagged_type, str) and tagged_type.strip():
            tagged_types.append(tagged_type.strip())
        prompt = _parse_choice_prompt(tagged)
        if prompt is not None:
            latest_choice = prompt

    result.genui_types = _dedupe(genui_types)
    result.tagged_ui_types = _dedupe(tagged_types)
    result.choice_prompt = latest_choice
    return result


def _send_turn_buffered(
    *,
    client: httpx.Client,
    base_url: str,
    headers: dict[str, str],
    language: str,
    turn: int,
    session_id: str,
    message: str,
    timeout_seconds: float,
) -> TurnResult:
    resp = client.post(
        f"{base_url}/chat/send-openai-stream",
        headers=headers,
        params={"language": language},
        json={"session_id": session_id, "message": message},
        timeout=timeout_seconds,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"turn {turn} buffered failed: {resp.status_code} {resp.text}")
    payloads = _parse_sse_payloads_from_text(resp.text)
    return _build_turn_result_from_payloads(
        turn=turn,
        sent_message=message,
        session_id=session_id,
        payloads=payloads,
    )


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def _derive_kyc_status_from_artifacts(artifacts: Any) -> str:
    if not isinstance(artifacts, dict):
        return "incomplete"
    kyc_block = artifacts.get("kyc")
    if isinstance(kyc_block, dict):
        missing = kyc_block.get("missing_fields")
        if isinstance(missing, list):
            normalized = [str(item).strip() for item in missing if str(item).strip()]
            return "complete" if not normalized else "incomplete"
    return "incomplete"


def _recover_turn_via_session(
    *,
    client: httpx.Client,
    base_url: str,
    headers: dict[str, str],
    session_id: str,
    sent_message: str,
    turn: int,
    poll_attempts: int = 6,
    poll_interval_seconds: float = 2.0,
) -> TurnResult | None:
    normalized = sent_message.strip()
    if not normalized:
        return None

    for _ in range(poll_attempts):
        try:
            resp = client.get(
                f"{base_url}/sessions/{session_id}",
                headers=headers,
                timeout=30.0,
            )
            if resp.status_code != 200:
                time.sleep(poll_interval_seconds)
                continue
            detail = resp.json()
        except Exception:  # noqa: BLE001
            time.sleep(poll_interval_seconds)
            continue

        messages = detail.get("messages")
        if not isinstance(messages, list):
            time.sleep(poll_interval_seconds)
            continue

        matched_user_index = -1
        for idx in range(len(messages) - 1, -1, -1):
            item = messages[idx]
            if not isinstance(item, dict):
                continue
            if str(item.get("role") or "") != "user":
                continue
            if str(item.get("content") or "").strip() == normalized:
                matched_user_index = idx
                break

        if matched_user_index < 0:
            time.sleep(poll_interval_seconds)
            continue

        assistant_item: dict[str, Any] | None = None
        for idx in range(matched_user_index + 1, len(messages)):
            item = messages[idx]
            if isinstance(item, dict) and str(item.get("role") or "") == "assistant":
                assistant_item = item
                break

        if assistant_item is None:
            time.sleep(poll_interval_seconds)
            continue

        recovered = TurnResult(
            turn=turn,
            sent_message=sent_message,
            phase=str(detail.get("current_phase") or "?"),
            kyc_status=_derive_kyc_status_from_artifacts(detail.get("artifacts")),
            session_id=session_id,
            done_seen=True,
        )

        content = str(assistant_item.get("content") or "")
        tool_calls = assistant_item.get("tool_calls")
        if not isinstance(tool_calls, list):
            tool_calls = []

        genui_types: list[str] = []
        latest_choice: ChoicePrompt | None = None

        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            payload_type = tool_call.get("type")
            if isinstance(payload_type, str) and payload_type.strip():
                genui_types.append(payload_type.strip())
            prompt = _parse_choice_prompt(tool_call)
            if prompt is not None:
                latest_choice = prompt

        tagged_payloads = _extract_tagged_ui_payloads(content)
        tagged_types: list[str] = []
        for tagged in tagged_payloads:
            tagged_type = tagged.get("type")
            if isinstance(tagged_type, str) and tagged_type.strip():
                tagged_types.append(tagged_type.strip())
            prompt = _parse_choice_prompt(tagged)
            if prompt is not None:
                latest_choice = prompt

        recovered.genui_types = _dedupe(genui_types)
        recovered.tagged_ui_types = _dedupe(tagged_types)
        recovered.choice_prompt = latest_choice
        return recovered

    return None


def _pick_next_message(prompt: ChoicePrompt, language: str) -> tuple[str, str]:
    preferred_id = _PREFERRED_IDS.get(prompt.choice_id)
    chosen: dict[str, str] | None = None
    if preferred_id:
        chosen = next((opt for opt in prompt.options if opt["id"] == preferred_id), None)
    if chosen is None and prompt.options:
        chosen = prompt.options[0]
    if chosen is None:
        # Shouldn't happen because parser requires >=2 options; fallback text.
        return ("继续", "fallback") if language.startswith("zh") else ("Continue", "fallback")
    return chosen["label"], chosen["id"]


def run(
    *,
    base_url: str,
    language: str,
    max_turns: int,
    timeout_seconds: float,
    strict: bool,
) -> int:
    with httpx.Client(timeout=30.0, trust_env=False) as client:
        email, token = _register(client, base_url)
        headers = {"Authorization": f"Bearer {token}"}
        thread = _create_thread(client, base_url, headers)
        session_id = str(thread.get("session_id") or "")
        phase = str(thread.get("phase") or "?")

        if not session_id:
            raise RuntimeError("new-thread returned empty session_id")

        print(f"[INFO] user={email}")
        print(f"[INFO] new-thread: session_id={session_id} phase={phase}")

        next_message = "你好" if language.startswith("zh") else "Hi"
        latest_prompt: ChoicePrompt | None = None
        symbol_turn_result: TurnResult | None = None
        selected_symbol_labels: set[str] = set()
        all_turn_results: list[TurnResult] = []

        start = time.perf_counter()
        for turn in range(1, max_turns + 1):
            if latest_prompt is not None:
                next_message, selected_id = _pick_next_message(latest_prompt, language)
                print(
                    f"[TURN {turn}] prompt={latest_prompt.choice_id} "
                    f"selected_id={selected_id} label={next_message!r}"
                )
                is_symbol_turn = latest_prompt.choice_id == "target_instrument"
                if is_symbol_turn:
                    selected_symbol_labels.add(next_message.strip().lower())
            else:
                print(f"[TURN {turn}] bootstrap_message={next_message!r}")
                is_symbol_turn = False

            try:
                turn_result = _send_turn(
                    client=client,
                    base_url=base_url,
                    headers=headers,
                    language=language,
                    turn=turn,
                    session_id=session_id,
                    message=next_message,
                    timeout_seconds=timeout_seconds,
                )
            except (httpx.ReadTimeout, httpx.RemoteProtocolError, httpx.TransportError) as exc:
                print(f"  -> stream interrupted ({type(exc).__name__}), trying session recovery...")
                recovered = _recover_turn_via_session(
                    client=client,
                    base_url=base_url,
                    headers=headers,
                    session_id=session_id,
                    sent_message=next_message,
                    turn=turn,
                )
                if recovered is not None:
                    turn_result = recovered
                else:
                    print("  -> recovery not found, trying buffered fallback request...")
                    turn_result = _send_turn_buffered(
                        client=client,
                        base_url=base_url,
                        headers=headers,
                        language=language,
                        turn=turn,
                        session_id=session_id,
                        message=next_message,
                        timeout_seconds=timeout_seconds,
                    )

            session_id = turn_result.session_id or session_id
            phase = turn_result.phase
            latest_prompt = turn_result.choice_prompt
            all_turn_results.append(turn_result)

            print(
                f"  -> phase={turn_result.phase} kyc={turn_result.kyc_status} "
                f"done={turn_result.done_seen} genui={turn_result.genui_types} "
                f"tagged={turn_result.tagged_ui_types}"
            )
            if turn_result.mcp_openai_types:
                print(f"  -> mcp_events={dict(turn_result.mcp_openai_types)}")
            if turn_result.mcp_tools:
                print(f"  -> mcp_tools={dict(turn_result.mcp_tools)}")
            if turn_result.errors:
                print(f"  -> stream_errors={turn_result.errors}")

            if is_symbol_turn:
                symbol_turn_result = turn_result
            else:
                sent_lower = turn_result.sent_message.strip().lower()
                if sent_lower in selected_symbol_labels and (
                    turn_result.has_chart
                    or turn_result.mcp_tools.get("check_symbol_available", 0) > 0
                    or turn_result.mcp_tools.get("get_quote", 0) > 0
                    or any(key.endswith("mcp_call.in_progress") for key in turn_result.mcp_openai_types)
                ):
                    symbol_turn_result = turn_result

            if phase == "strategy":
                break

        elapsed = time.perf_counter() - start
        print(f"[INFO] elapsed={elapsed:.1f}s final_phase={phase}")

        findings: list[str] = []
        if phase != "strategy":
            findings.append("flow did not reach strategy phase within max_turns")

        if symbol_turn_result is None:
            for candidate in reversed(all_turn_results):
                if candidate.has_chart:
                    symbol_turn_result = candidate
                    break

        if symbol_turn_result is None:
            findings.append("did not observe target_instrument selection turn")
        else:
            if not symbol_turn_result.has_mcp_in_progress:
                findings.append("symbol turn missing MCP in_progress event")
            if not symbol_turn_result.has_mcp_terminal:
                findings.append("symbol turn missing MCP terminal event")
            if not symbol_turn_result.has_chart:
                findings.append("symbol turn missing tradingview_chart")
            if not symbol_turn_result.has_choice:
                findings.append("symbol turn missing choice_prompt")
            if strict:
                if symbol_turn_result.mcp_tools.get("check_symbol_available", 0) == 0:
                    findings.append("symbol turn missing check_symbol_available tool usage")
                if symbol_turn_result.mcp_tools.get("get_quote", 0) == 0:
                    findings.append("symbol turn missing get_quote tool usage")

        if findings:
            print(f"[FAIL] findings={findings}")
            return 1

        print("[PASS] frontend-like flow probe passed")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Probe full frontend-like flow and assert symbol-turn outputs."
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="API base url")
    parser.add_argument("--language", default="en", help="language code")
    parser.add_argument("--max-turns", type=int, default=14, help="max streamed turns")
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=150.0,
        help="per-turn stream timeout seconds",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="strictly require both check_symbol_available and get_quote on symbol turn",
    )
    args = parser.parse_args()

    try:
        return run(
            base_url=args.base_url.rstrip("/"),
            language=args.language,
            max_turns=args.max_turns,
            timeout_seconds=args.timeout_seconds,
            strict=args.strict,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
