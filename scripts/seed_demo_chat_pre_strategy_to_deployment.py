#!/usr/bin/env python3
"""Seed one real OpenAI+MCP chat session from pre-strategy to deployment.

This script:
1) logs in as an existing user,
2) drives real `/chat/send-openai-stream` turns in English,
3) waits for persisted assistant messages after every turn,
4) stores an artifact with transcript + MCP calls.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests


@dataclass
class TurnResult:
    user_message: str
    phase_after: str
    done_payload: dict[str, Any]
    streamed_text: str
    assistant_text: str
    openai_event_count: int
    mcp_event_count: int
    mcp_calls: list[dict[str, Any]]
    genui_payloads: list[dict[str, Any]]


class LiveApiClient:
    def __init__(
        self,
        *,
        base_url: str,
        email: str,
        password: str,
        language: str,
        stream_timeout_seconds: float,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.email = email
        self.password = password
        self.language = language
        self.stream_timeout_seconds = stream_timeout_seconds
        self.http = requests.Session()
        self.token: str | None = None

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def login(self) -> dict[str, Any]:
        response = self.http.post(
            self._url("/auth/login"),
            json={"email": self.email, "password": self.password},
            timeout=30,
        )
        _assert_status(response, 200, "auth_login")
        payload = response.json()
        token = payload.get("access_token")
        if not isinstance(token, str) or not token.strip():
            raise RuntimeError("auth_login returned empty access_token")
        self.token = token.strip()
        return payload

    def create_thread(self, *, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        response = self.http.post(
            self._url("/chat/new-thread"),
            headers=self._headers(),
            json={"metadata": metadata or {}},
            timeout=30,
        )
        _assert_status(response, 201, "chat_new_thread")
        payload = response.json()
        if not isinstance(payload.get("session_id"), str):
            raise RuntimeError(f"chat_new_thread missing session_id: {payload}")
        return payload

    def get_session(self, session_id: str) -> dict[str, Any]:
        response = self.http.get(
            self._url(f"/sessions/{session_id}"),
            headers=self._headers(),
            timeout=30,
        )
        _assert_status(response, 200, "sessions_get")
        return response.json()

    def get_strategy_draft(self, strategy_draft_id: str) -> dict[str, Any]:
        response = self.http.get(
            self._url(f"/strategies/drafts/{strategy_draft_id}"),
            headers=self._headers(),
            timeout=30,
        )
        _assert_status(response, 200, "strategy_draft_get")
        return response.json()

    def confirm_strategy(
        self,
        *,
        session_id: str,
        dsl_json: dict[str, Any],
    ) -> dict[str, Any]:
        response = self.http.post(
            self._url("/strategies/confirm"),
            headers=self._headers(),
            json={
                "session_id": session_id,
                "dsl_json": dsl_json,
                "auto_start_backtest": False,
                "language": self.language,
            },
            timeout=120,
        )
        _assert_status(response, 200, "strategy_confirm")
        return response.json()

    def send_turn(self, *, session_id: str, message: str) -> TurnResult:
        before = self.get_session(session_id)
        assistant_count_before = len(_assistant_messages(before))

        response = self.http.post(
            self._url(f"/chat/send-openai-stream?language={self.language}"),
            headers=self._headers(),
            json={"session_id": session_id, "message": message},
            timeout=(30, self.stream_timeout_seconds + 30),
        )
        _assert_status(response, 200, "chat_send_openai_stream")

        payloads = _parse_sse_payloads(response.text)
        streamed_text_parts = [
            str(item.get("delta", ""))
            for item in payloads
            if isinstance(item, dict) and item.get("type") == "text_delta"
        ]
        openai_event_count = sum(
            1
            for item in payloads
            if isinstance(item, dict) and item.get("type") == "openai_event"
        )
        mcp_event_count = sum(
            1
            for item in payloads
            if isinstance(item, dict) and item.get("type") == "mcp_event"
        )
        done_payload = next(
            (
                item
                for item in payloads
                if isinstance(item, dict) and item.get("type") == "done"
            ),
            None,
        )
        if not isinstance(done_payload, dict):
            raise RuntimeError("chat turn ended without done payload")

        after = self._wait_for_assistant_message(
            session_id=session_id,
            previous_assistant_count=assistant_count_before,
            timeout_seconds=90.0,
        )
        assistant = _assistant_messages(after)[-1]

        tool_calls = assistant.get("tool_calls")
        tool_call_list = tool_calls if isinstance(tool_calls, list) else []
        mcp_calls = [
            _coerce_tool_call(item)
            for item in tool_call_list
            if isinstance(item, dict) and str(item.get("type", "")).strip().lower() == "mcp_call"
        ]
        genui_payloads = [
            _coerce_tool_call(item)
            for item in tool_call_list
            if isinstance(item, dict) and str(item.get("type", "")).strip().lower() != "mcp_call"
        ]

        phase_after_raw = done_payload.get("phase")
        phase_after = (
            phase_after_raw.strip()
            if isinstance(phase_after_raw, str) and phase_after_raw.strip()
            else str(after.get("current_phase", "unknown"))
        )

        assistant_text = str(assistant.get("content", "")).strip()
        streamed_text = "".join(streamed_text_parts).strip()
        return TurnResult(
            user_message=message,
            phase_after=phase_after,
            done_payload=done_payload,
            streamed_text=streamed_text,
            assistant_text=assistant_text,
            openai_event_count=openai_event_count,
            mcp_event_count=mcp_event_count,
            mcp_calls=mcp_calls,
            genui_payloads=genui_payloads,
        )

    def _wait_for_assistant_message(
        self,
        *,
        session_id: str,
        previous_assistant_count: int,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        latest: dict[str, Any] | None = None
        while time.monotonic() < deadline:
            snapshot = self.get_session(session_id)
            latest = snapshot
            if len(_assistant_messages(snapshot)) >= previous_assistant_count + 1:
                return snapshot
            time.sleep(1.2)
        if latest is not None:
            return latest
        raise RuntimeError("timed out waiting for assistant message persistence")


def _assert_status(response: requests.Response, expected_status: int, step: str) -> None:
    if response.status_code == expected_status:
        return
    raise RuntimeError(
        f"{step} failed: status={response.status_code} expected={expected_status} body={response.text[:1200]}"
    )


def _assistant_messages(session_payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw = session_payload.get("messages")
    if not isinstance(raw, list):
        return []
    return [
        item
        for item in raw
        if isinstance(item, dict) and str(item.get("role", "")).strip() == "assistant"
    ]


def _parse_sse_payloads(raw_text: str) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    blocks = [block.strip() for block in raw_text.split("\n\n") if block.strip()]
    for block in blocks:
        for line in block.splitlines():
            if not line.startswith("data: "):
                continue
            raw = line.removeprefix("data: ").strip()
            if not raw:
                continue
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                payloads.append(parsed)
    return payloads


def _coerce_tool_call(value: dict[str, Any]) -> dict[str, Any]:
    return dict(value)


def _parse_json_like(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _extract_strategy_draft_id(turn: TurnResult) -> str | None:
    for payload in reversed(turn.genui_payloads):
        if str(payload.get("type", "")).strip().lower() != "strategy_ref":
            continue
        draft_id = payload.get("strategy_draft_id")
        if isinstance(draft_id, str) and draft_id.strip():
            return draft_id.strip()

    for call in reversed(turn.mcp_calls):
        if str(call.get("name", "")).strip() != "strategy_validate_dsl":
            continue
        output = _parse_json_like(call.get("output"))
        if not isinstance(output, dict):
            continue
        draft_id = output.get("strategy_draft_id")
        if isinstance(draft_id, str) and draft_id.strip():
            return draft_id.strip()
    return None


def _has_mcp_success(turn: TurnResult, tool_name: str) -> bool:
    for call in turn.mcp_calls:
        if str(call.get("name", "")).strip() != tool_name:
            continue
        if str(call.get("status", "")).strip().lower() == "success":
            return True
    return False


def _extract_strategy_id_from_session(session_payload: dict[str, Any]) -> str | None:
    artifacts = session_payload.get("artifacts")
    if not isinstance(artifacts, dict):
        return None
    for phase in ("strategy", "deployment", "stress_test"):
        phase_block = artifacts.get(phase)
        if not isinstance(phase_block, dict):
            continue
        profile = phase_block.get("profile")
        if not isinstance(profile, dict):
            continue
        raw = profile.get("strategy_id")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return None


def _has_choice_prompt(turn: TurnResult) -> bool:
    return any(str(payload.get("type", "")).strip().lower() == "choice_prompt" for payload in turn.genui_payloads)


def _extract_backtest_total_return_pct(turn: TurnResult) -> float | None:
    for call in reversed(turn.mcp_calls):
        if str(call.get("name", "")).strip() != "backtest_get_job":
            continue
        if str(call.get("status", "")).strip().lower() != "success":
            continue
        output = _parse_json_like(call.get("output"))
        if not isinstance(output, dict):
            continue
        result = output.get("result")
        if not isinstance(result, dict):
            continue
        summary = result.get("summary")
        if not isinstance(summary, dict):
            continue
        raw = summary.get("total_return_pct")
        if isinstance(raw, int | float):
            return float(raw)
    return None


def _print_turn(label: str, turn: TurnResult) -> None:
    mcp_names = [str(item.get("name", "")).strip() for item in turn.mcp_calls]
    print(
        f"[{label}] phase_after={turn.phase_after} "
        f"openai_events={turn.openai_event_count} mcp_events={turn.mcp_event_count} "
        f"mcp_calls={mcp_names}",
        flush=True,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/api/v1")
    parser.add_argument("--email", default="2@test.com")
    parser.add_argument("--password", default="123456")
    parser.add_argument("--language", default="en")
    parser.add_argument("--stream-timeout-seconds", type=float, default=300.0)
    parser.add_argument(
        "--stop-after-tune",
        action="store_true",
        help="Stop once strategy tuning + backtest turn is completed.",
    )
    parser.add_argument(
        "--artifact-file",
        default="artifacts/demo_seed_pre_strategy_to_deployment/latest.json",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    client = LiveApiClient(
        base_url=args.base_url,
        email=args.email,
        password=args.password,
        language=args.language,
        stream_timeout_seconds=float(args.stream_timeout_seconds),
    )
    login_payload = client.login()
    user_id = str(login_payload.get("user_id"))
    thread = client.create_thread(
        metadata={
            "seed_source": "seed_demo_chat_pre_strategy_to_deployment.py",
            "language": args.language,
            "created_at": datetime.now(UTC).isoformat(),
        }
    )
    session_id = str(thread["session_id"])
    print(f"[session] id={session_id} initial_phase={thread.get('phase')}")

    turns: list[TurnResult] = []
    actions: list[dict[str, Any]] = []

    def send(label: str, message: str) -> TurnResult:
        turn = client.send_turn(session_id=session_id, message=message)
        turns.append(turn)
        _print_turn(label, turn)
        return turn

    # 1) Pre-strategy (crypto + 1m + high-frequency intent).
    pre_messages = [
        "I want to build a crypto 1-minute high-frequency strategy. Start with the scope setup first.",
        "Use BTCUSD as the main trading instrument.",
        "I want multiple opportunities per day and intraday scalp holding style.",
    ]
    latest_turn: TurnResult | None = None
    for idx, message in enumerate(pre_messages, start=1):
        latest_turn = send(f"pre_strategy_{idx}", message)
        if latest_turn.phase_after == "strategy":
            break

    if latest_turn is None:
        raise RuntimeError("no pre-strategy turns were sent")
    if latest_turn.phase_after != "strategy":
        latest_turn = send(
            "pre_strategy_force_complete",
            (
                "Please set these values exactly and move us to strategy phase: "
                "target_market=crypto, target_instrument=BTCUSD, "
                "opportunity_frequency_bucket=multiple_per_day, "
                "holding_period_bucket=intraday_scalp."
            ),
        )
    if latest_turn.phase_after != "strategy":
        raise RuntimeError("failed to enter strategy phase from pre-strategy")

    # 2) Strategy creation (draft validate).
    draft_id: str | None = None
    latest_turn = None
    for attempt in range(1, 6):
        prompt = (
            "Please create and validate a complete 1-minute BTCUSD high-frequency strategy DSL draft now."
            if attempt == 1
            else (
                "Retry in this same session. Validate one complete DSL draft and return strategy_ref. "
                "Use real MCP tool calls only."
            )
        )
        latest_turn = send(f"strategy_draft_{attempt}", prompt)
        draft_id = _extract_strategy_draft_id(latest_turn)
        if draft_id and _has_mcp_success(latest_turn, "strategy_validate_dsl"):
            break
    if not draft_id:
        raise RuntimeError("strategy draft id was not produced")

    # 3) Try chat-based baseline save (without immediate deploy).
    latest_turn = send(
        "strategy_save_baseline",
        (
            "Save this as a baseline strategy so we can do one tuning iteration first. "
            "Do not finalize for deployment yet."
        ),
    )
    session_after_save = client.get_session(session_id)
    strategy_id = _extract_strategy_id_from_session(session_after_save)
    current_phase = str(session_after_save.get("current_phase", ""))

    # Fallback: persist draft via strategy confirm API if strategy_id still missing.
    if not strategy_id:
        draft_payload = client.get_strategy_draft(draft_id)
        dsl_json = draft_payload.get("dsl_json")
        if not isinstance(dsl_json, dict):
            raise RuntimeError("strategy draft payload missing dsl_json object")
        confirm_payload = client.confirm_strategy(session_id=session_id, dsl_json=dsl_json)
        actions.append(
            {
                "action": "strategy_confirm_api",
                "strategy_id": str(confirm_payload.get("strategy_id")),
                "phase": confirm_payload.get("phase"),
            }
        )
        session_after_save = client.get_session(session_id)
        strategy_id = _extract_strategy_id_from_session(session_after_save)
        current_phase = str(session_after_save.get("current_phase", ""))

    if not strategy_id:
        raise RuntimeError("strategy_id is still missing after baseline save/confirm")

    # If already in deployment phase, force blocked -> back to strategy for one micro-tune turn.
    if current_phase == "deployment":
        latest_turn = send(
            "deployment_back_to_strategy",
            (
                "Set deployment_status to blocked because I need one more strategy tuning pass now, "
                "then route me back to strategy."
            ),
        )
        if latest_turn.phase_after != "strategy":
            latest_turn = send(
                "deployment_back_to_strategy_retry",
                "Mark deployment as blocked and return to strategy immediately.",
            )
        if latest_turn.phase_after != "strategy":
            raise RuntimeError("failed to transition deployment -> strategy for tuning")

    # 4) One micro-tune iteration in strategy.
    latest_turn = None
    for attempt in range(1, 5):
        prompt = (
            "Please fine-tune the saved strategy with minimal patch operations: "
            "reduce risk exposure by around 30%, tighten exits for fast loss control, "
            "and run a short backtest using available 1-minute BTCUSD coverage."
            if attempt == 1
            else (
                "Retry the tuning turn. Fetch current DSL, apply a minimal patch, "
                "and keep the strategy in strategy phase."
            )
        )
        latest_turn = send(f"strategy_tune_{attempt}", prompt)
        if _has_mcp_success(latest_turn, "strategy_patch_dsl") or _has_mcp_success(
            latest_turn, "strategy_upsert_dsl"
        ):
            break
    if latest_turn is None:
        raise RuntimeError("strategy tune step not executed")
    tuned_total_return_pct = _extract_backtest_total_return_pct(latest_turn)

    # Ensure still strategy before final deployment confirmation.
    session_after_tune = client.get_session(session_id)
    if str(session_after_tune.get("current_phase", "")) != "strategy":
        latest_turn = send(
            "strategy_return_retry",
            "I still need to stay in strategy phase for this tuning cycle.",
        )
        if latest_turn.phase_after != "strategy":
            raise RuntimeError("unexpected phase after tuning; expected strategy")

    if bool(args.stop_after_tune):
        final_session = client.get_session(session_id)
        final_phase = str(final_session.get("current_phase", ""))
        strategy_id = _extract_strategy_id_from_session(final_session)
        transcript = [
            {
                "turn_index": index + 1,
                **asdict(turn),
            }
            for index, turn in enumerate(turns)
        ]
        return {
            "started_at": datetime.now(UTC).isoformat(),
            "base_url": args.base_url,
            "email": args.email,
            "language": args.language,
            "user_id": user_id,
            "session_id": session_id,
            "final_phase": final_phase,
            "strategy_id": strategy_id,
            "turn_count": len(turns),
            "actions": actions,
            "tuned_total_return_pct": tuned_total_return_pct,
            "checks": {
                "started_from_pre_strategy": str(thread.get("phase", "")) == "pre_strategy",
                "strategy_draft_generated": any(
                    _has_mcp_success(turn, "strategy_validate_dsl") for turn in turns
                ),
                "strategy_tuned": any(
                    _has_mcp_success(turn, "strategy_patch_dsl")
                    or _has_mcp_success(turn, "strategy_upsert_dsl")
                    for turn in turns
                ),
                "has_backtest_result": tuned_total_return_pct is not None,
                "all_turns_have_openai_events": all(turn.openai_event_count > 0 for turn in turns),
            },
            "transcript": transcript,
            "final_session_snapshot": {
                "current_phase": final_phase,
                "session_title": final_session.get("session_title"),
                "message_count": len(final_session.get("messages", []))
                if isinstance(final_session.get("messages"), list)
                else 0,
            },
            "finished_at": datetime.now(UTC).isoformat(),
            "result": "ok",
        }

    # 5) Finalize strategy and enter deployment.
    latest_turn = None
    for attempt in range(1, 5):
        latest_turn = send(
            f"strategy_finalize_{attempt}",
            "This tuned strategy is finalized and ready to deploy now.",
        )
        if latest_turn.phase_after == "deployment":
            break
    if latest_turn is None or latest_turn.phase_after != "deployment":
        raise RuntimeError("failed to enter deployment phase after finalization")

    # 6) Deployment create + start.
    latest_turn = None
    for attempt in range(1, 5):
        latest_turn = send(
            f"deployment_create_start_{attempt}",
            (
                "Please create a paper deployment and start it now. "
                "Reuse my existing active paper broker account and summarize deployment id + status."
            ),
        )
        has_create = _has_mcp_success(latest_turn, "trading_create_paper_deployment")
        has_start = _has_mcp_success(latest_turn, "trading_start_deployment")
        if has_create and (has_start or _has_mcp_success(latest_turn, "trading_list_deployments")):
            break

    # 7) End on AgentUI choice prompt.
    latest_turn = None
    for attempt in range(1, 6):
        latest_turn = send(
            f"deployment_choice_{attempt}",
            (
                "For the next step, ask me using AGENT_UI_JSON choice_prompt with exactly three options: "
                "inspect_positions, pause_deployment, stop_deployment."
            ),
        )
        if _has_choice_prompt(latest_turn):
            break
    if latest_turn is None or not _has_choice_prompt(latest_turn):
        raise RuntimeError("failed to end on agentui choice prompt")

    final_session = client.get_session(session_id)
    final_phase = str(final_session.get("current_phase", ""))
    strategy_id = _extract_strategy_id_from_session(final_session)

    transcript = [
        {
            "turn_index": index + 1,
            **asdict(turn),
        }
        for index, turn in enumerate(turns)
    ]
    return {
        "started_at": datetime.now(UTC).isoformat(),
        "base_url": args.base_url,
        "email": args.email,
        "language": args.language,
        "user_id": user_id,
        "session_id": session_id,
        "final_phase": final_phase,
        "strategy_id": strategy_id,
        "turn_count": len(turns),
        "actions": actions,
        "checks": {
            "started_from_pre_strategy": str(thread.get("phase", "")) == "pre_strategy",
            "contains_pre_strategy_mcp": any(
                any(call.get("name") in {"check_symbol_available", "get_symbol_quote"} for call in turn.mcp_calls)
                for turn in turns
            ),
            "strategy_draft_generated": any(
                _has_mcp_success(turn, "strategy_validate_dsl") for turn in turns
            ),
            "strategy_tuned": any(
                _has_mcp_success(turn, "strategy_patch_dsl") or _has_mcp_success(turn, "strategy_upsert_dsl")
                for turn in turns
            ),
            "deployment_tools_called": any(
                _has_mcp_success(turn, "trading_create_paper_deployment")
                or _has_mcp_success(turn, "trading_start_deployment")
                for turn in turns
            ),
            "ended_with_choice_prompt": bool(turns and _has_choice_prompt(turns[-1])),
            "all_turns_have_openai_events": all(turn.openai_event_count > 0 for turn in turns),
        },
        "transcript": transcript,
        "final_session_snapshot": {
            "current_phase": final_phase,
            "session_title": final_session.get("session_title"),
            "message_count": len(final_session.get("messages", []))
            if isinstance(final_session.get("messages"), list)
            else 0,
        },
        "finished_at": datetime.now(UTC).isoformat(),
        "result": "ok",
    }


def main() -> None:
    args = _parse_args()
    report = run(args)

    artifact_path = Path(args.artifact_file).resolve()
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"artifact={artifact_path}")


if __name__ == "__main__":
    main()
