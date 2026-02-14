"""Benchmark per-turn OpenAI token usage through real /chat/send-openai-stream calls.

Usage examples:
  uv run python -m src.scripts.benchmark_openai_turn_tokens
  uv run python -m src.scripts.benchmark_openai_turn_tokens --language zh
  uv run python -m src.scripts.benchmark_openai_turn_tokens --base-url http://127.0.0.1:8000/api/v1
  uv run python -m src.scripts.benchmark_openai_turn_tokens --messages-json '["你好，我有5年经验", "继续"]'
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:8000/api/v1"
DEFAULT_MESSAGES: list[str] = [
    "我有超过5年的交易经验，风险偏好激进，并且追求高增长收益。",
    "继续。",
    "我想交易美股。",
    "SPY。",
    "我希望每天都有交易机会。",
    "我通常会持有几天。",
    "请给我一个趋势跟踪的策略初稿。",
]


@dataclass
class TurnUsage:
    turn: int
    sent_message: str
    session_id: str
    phase: str
    status: str
    kyc_status: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    stream_error: str | None


class BenchmarkError(RuntimeError):
    """Raised when benchmark flow cannot proceed."""


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


def _register(client: httpx.Client, base_url: str) -> str:
    email = f"token_bench_{uuid4().hex[:10]}@example.com"
    response = client.post(
        f"{base_url}/auth/register",
        json={"email": email, "password": "bench123456", "name": "Token Benchmark"},
        timeout=30.0,
    )
    if response.status_code != 201:
        raise BenchmarkError(f"register failed: {response.status_code} {response.text}")
    payload = response.json()
    token = payload.get("access_token")
    if not isinstance(token, str) or not token.strip():
        raise BenchmarkError("register failed: missing access_token")
    return token.strip()


def _create_thread(client: httpx.Client, base_url: str, headers: dict[str, str]) -> str:
    response = client.post(
        f"{base_url}/chat/new-thread",
        headers=headers,
        json={"metadata": {"source": "token_benchmark"}},
        timeout=30.0,
    )
    if response.status_code != 201:
        raise BenchmarkError(f"new-thread failed: {response.status_code} {response.text}")
    payload = response.json()
    session_id = payload.get("session_id")
    if not isinstance(session_id, str) or not session_id.strip():
        raise BenchmarkError("new-thread failed: missing session_id")
    return session_id.strip()


def _parse_usage(payload: dict[str, Any]) -> tuple[int, int, int]:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return (0, 0, 0)

    def _to_int(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    input_tokens = _to_int(usage.get("input_tokens"))
    output_tokens = _to_int(usage.get("output_tokens"))
    total_tokens = _to_int(usage.get("total_tokens"))
    if total_tokens <= 0:
        total_tokens = input_tokens + output_tokens
    return input_tokens, output_tokens, total_tokens


def _send_turn(
    *,
    client: httpx.Client,
    base_url: str,
    headers: dict[str, str],
    session_id: str,
    language: str,
    message: str,
    timeout_seconds: float,
) -> TurnUsage:
    done_payload: dict[str, Any] | None = None
    resolved_session_id = session_id

    with client.stream(
        "POST",
        f"{base_url}/chat/send-openai-stream",
        headers=headers,
        params={"language": language},
        json={"session_id": session_id, "message": message},
        timeout=timeout_seconds,
    ) as response:
        if response.status_code != 200:
            body = response.read().decode("utf-8", errors="replace")
            raise BenchmarkError(
                f"send-openai-stream failed: {response.status_code} {body}",
            )

        for _, raw_data in _iter_sse_events(response):
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
                if isinstance(sid, str) and sid.strip():
                    resolved_session_id = sid.strip()
                continue

            if event_type == "done":
                done_payload = payload
                break

    if done_payload is None:
        raise BenchmarkError("stream finished without done payload")

    input_tokens, output_tokens, total_tokens = _parse_usage(done_payload)
    return TurnUsage(
        turn=0,
        sent_message=message,
        session_id=resolved_session_id,
        phase=str(done_payload.get("phase", "?")),
        status=str(done_payload.get("status", "?")),
        kyc_status=str(done_payload.get("kyc_status", "?")),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        stream_error=(
            str(done_payload.get("stream_error"))
            if isinstance(done_payload.get("stream_error"), str)
            and str(done_payload.get("stream_error")).strip()
            else None
        ),
    )


def _format_markdown(
    *,
    base_url: str,
    language: str,
    model_hint: str,
    results: list[TurnUsage],
) -> str:
    total_input = sum(item.input_tokens for item in results)
    total_output = sum(item.output_tokens for item in results)
    total_all = sum(item.total_tokens for item in results)

    lines = [
        "# OpenAI Turn Token Benchmark",
        "",
        f"- Generated at (UTC): {datetime.now(UTC).isoformat()}",
        f"- Base URL: `{base_url}`",
        f"- Language: `{language}`",
        f"- Model hint: `{model_hint}`",
        "",
        "## Per Turn",
        "",
        "| Turn | Phase(done) | input_tokens | output_tokens | total_tokens | stream_error | message |",
        "|---:|---|---:|---:|---:|---|---|",
    ]

    for item in results:
        message = item.sent_message.replace("|", "\\|").strip()
        if len(message) > 60:
            message = message[:57] + "..."
        lines.append(
            "| "
            f"{item.turn} | {item.phase} | {item.input_tokens} | {item.output_tokens} | "
            f"{item.total_tokens} | {item.stream_error or ''} | {message} |"
        )

    lines.extend(
        [
            "",
            "## Totals",
            "",
            f"- total_input_tokens: **{total_input}**",
            f"- total_output_tokens: **{total_output}**",
            f"- total_tokens: **{total_all}**",
        ]
    )
    return "\n".join(lines) + "\n"


def _parse_messages(messages_json: str | None) -> list[str]:
    if messages_json is None:
        return list(DEFAULT_MESSAGES)
    try:
        payload = json.loads(messages_json)
    except json.JSONDecodeError as exc:
        raise BenchmarkError(f"invalid --messages-json: {exc}") from exc
    if not isinstance(payload, list):
        raise BenchmarkError("--messages-json must decode to a JSON list")
    messages = [str(item).strip() for item in payload if str(item).strip()]
    if not messages:
        raise BenchmarkError("messages list is empty")
    return messages


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark OpenAI per-turn token usage")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--language", default="zh")
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--messages-json", default=None)
    parser.add_argument(
        "--output",
        default="backend/benchmark_reports/openai_turn_token_benchmark.md",
        help="Markdown output path",
    )
    parser.add_argument(
        "--model-hint",
        default="settings.openai_response_model",
        help="For annotation only; script reads real usage from done payload",
    )
    args = parser.parse_args()

    try:
        messages = _parse_messages(args.messages_json)
        base_url = args.base_url.rstrip("/")

        with httpx.Client(timeout=30.0, trust_env=False) as client:
            token = _register(client, base_url)
            headers = {"Authorization": f"Bearer {token}"}
            session_id = _create_thread(client, base_url, headers)

            results: list[TurnUsage] = []
            for idx, message in enumerate(messages, start=1):
                usage = _send_turn(
                    client=client,
                    base_url=base_url,
                    headers=headers,
                    session_id=session_id,
                    language=args.language,
                    message=message,
                    timeout_seconds=args.timeout_seconds,
                )
                usage.turn = idx
                session_id = usage.session_id
                results.append(usage)
                print(
                    f"turn={idx} phase={usage.phase} input={usage.input_tokens} "
                    f"output={usage.output_tokens} total={usage.total_tokens}"
                )

        output_text = _format_markdown(
            base_url=base_url,
            language=args.language,
            model_hint=args.model_hint,
            results=results,
        )
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_text, encoding="utf-8")

        print(f"\nSaved benchmark report: {output_path}")
        return 0

    except BenchmarkError as exc:
        print(f"[benchmark-error] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
