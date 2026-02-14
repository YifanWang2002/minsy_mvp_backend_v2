"""Benchmark per-turn OpenAI token usage via FastAPI TestClient (real OpenAI calls).

Usage:
  uv run python -m src.scripts.benchmark_openai_turn_tokens_testclient
  uv run python -m src.scripts.benchmark_openai_turn_tokens_testclient --language zh
  uv run python -m src.scripts.benchmark_openai_turn_tokens_testclient --messages-json '["...", "..."]'
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

from fastapi.testclient import TestClient

from src.main import app

DEFAULT_MESSAGES: list[str] = [
    "我的交易经验是 years_5_plus，风险偏好 aggressive，收益预期 high_growth。请继续。",
    "目标市场 target_market=us_stocks。",
    "目标标的是 target_instrument=SPY。",
    "机会频率是 opportunity_frequency_bucket=daily。",
    "持有周期是 holding_period_bucket=swing_days。",
    "请基于以上范围创建并保存一个最简趋势策略，返回 strategy_id。",
    "继续。",
]
DEFAULT_OUTPUT_PATH = Path("benchmark_reports/openai_turn_token_benchmark_testclient.md")


@dataclass
class TurnUsage:
    turn: int
    message: str
    phase: str
    status: str
    kyc_status: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    stream_error: str | None


class BenchmarkError(RuntimeError):
    """Benchmark flow error."""


def _parse_sse_payloads(raw_text: str) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    blocks = [block.strip() for block in raw_text.split("\n\n") if block.strip()]
    for block in blocks:
        for line in block.splitlines():
            if not line.startswith("data: "):
                continue
            try:
                parsed = json.loads(line.removeprefix("data: "))
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                payloads.append(parsed)
    return payloads


def _to_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _parse_usage(done_payload: dict[str, Any]) -> tuple[int, int, int]:
    usage = done_payload.get("usage")
    if not isinstance(usage, dict):
        return (0, 0, 0)
    input_tokens = _to_int(usage.get("input_tokens"))
    output_tokens = _to_int(usage.get("output_tokens"))
    total_tokens = _to_int(usage.get("total_tokens"))
    if total_tokens <= 0:
        total_tokens = input_tokens + output_tokens
    return input_tokens, output_tokens, total_tokens


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
    parser = argparse.ArgumentParser(description="Benchmark OpenAI per-turn token usage via TestClient")
    parser.add_argument("--language", default="zh")
    parser.add_argument("--messages-json", default=None)
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Markdown output path",
    )
    parser.add_argument(
        "--max-turn-retries",
        type=int,
        default=3,
        help="Retry the same turn when usage is zero and stream_error is present",
    )
    args = parser.parse_args()
    messages = _parse_messages(args.messages_json)
    output_path = Path(args.output)

    with TestClient(app) as client:
        email = f"token_bench_tc_{uuid4().hex[:10]}@example.com"
        register_resp = client.post(
            "/api/v1/auth/register",
            json={"email": email, "password": "bench123456", "name": "Token Benchmark TC"},
        )
        if register_resp.status_code != 201:
            raise BenchmarkError(f"register failed: {register_resp.status_code} {register_resp.text}")
        token = register_resp.json().get("access_token")
        if not isinstance(token, str) or not token.strip():
            raise BenchmarkError("register failed: missing access_token")

        headers = {"Authorization": f"Bearer {token.strip()}"}

        new_thread_resp = client.post(
            "/api/v1/chat/new-thread",
            headers=headers,
            json={"metadata": {"source": "token_benchmark_testclient"}},
        )
        if new_thread_resp.status_code != 201:
            raise BenchmarkError(
                f"new-thread failed: {new_thread_resp.status_code} {new_thread_resp.text}"
            )
        session_id = str(new_thread_resp.json().get("session_id", "")).strip()
        if not session_id:
            raise BenchmarkError("new-thread failed: missing session_id")

        results: list[TurnUsage] = []
        for idx, message in enumerate(messages, start=1):
            result: TurnUsage | None = None
            attempts = max(1, int(args.max_turn_retries))
            for attempt in range(1, attempts + 1):
                response = client.post(
                    f"/api/v1/chat/send-openai-stream?language={args.language}",
                    headers=headers,
                    json={"session_id": session_id, "message": message},
                )
                if response.status_code != 200:
                    raise BenchmarkError(
                        f"turn {idx} failed: {response.status_code} {response.text}"
                    )

                payloads = _parse_sse_payloads(response.text)
                done_payload = next((item for item in payloads if item.get("type") == "done"), None)
                if not isinstance(done_payload, dict):
                    raise BenchmarkError(f"turn {idx} missing done payload")

                sid_from_done = done_payload.get("session_id")
                if isinstance(sid_from_done, str) and sid_from_done.strip():
                    session_id = sid_from_done.strip()

                input_tokens, output_tokens, total_tokens = _parse_usage(done_payload)
                candidate = TurnUsage(
                    turn=idx,
                    message=message,
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
                if candidate.input_tokens > 0 or candidate.total_tokens > 0:
                    result = candidate
                    break
                if attempt < attempts:
                    print(
                        "turn="
                        f"{idx} retry={attempt + 1}/{attempts} because usage=0 "
                        f"(stream_error={candidate.stream_error!r})"
                    )
                    continue
                raise BenchmarkError(
                    "turn "
                    f"{idx} returned zero usage tokens after {attempts} attempts "
                    f"(stream_error={candidate.stream_error!r})",
                )

            if result is None:
                raise BenchmarkError(f"turn {idx} failed with unknown retry flow")
            results.append(result)
            print(
                f"turn={result.turn} phase={result.phase} "
                f"input={result.input_tokens} output={result.output_tokens} total={result.total_tokens}"
            )

    total_input = sum(item.input_tokens for item in results)
    total_output = sum(item.output_tokens for item in results)
    total_tokens = sum(item.total_tokens for item in results)

    lines = [
        "# OpenAI Turn Token Benchmark (TestClient)",
        "",
        f"- Generated at (UTC): {datetime.now(UTC).isoformat()}",
        f"- Language: `{args.language}`",
        f"- Turns: `{len(messages)}`",
        "",
        "## Per Turn",
        "",
        "| Turn | Phase(done) | input_tokens | output_tokens | total_tokens | stream_error | message |",
        "|---:|---|---:|---:|---:|---|---|",
    ]
    for item in results:
        msg = item.message.replace("|", "\\|")
        if len(msg) > 60:
            msg = msg[:57] + "..."
        lines.append(
            "| "
            f"{item.turn} | {item.phase} | {item.input_tokens} | {item.output_tokens} | "
            f"{item.total_tokens} | {item.stream_error or ''} | {msg} |"
        )

    lines.extend(
        [
            "",
            "## Totals",
            "",
            f"- total_input_tokens: **{total_input}**",
            f"- total_output_tokens: **{total_output}**",
            f"- total_tokens: **{total_tokens}**",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved benchmark report: {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BenchmarkError as exc:
        print(f"[benchmark-error] {exc}", file=sys.stderr)
        raise SystemExit(2)
