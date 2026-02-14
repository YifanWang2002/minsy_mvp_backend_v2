#!/usr/bin/env python3
"""Verify OpenAI tool-call semantics for strategy_get_dsl / strategy_patch_dsl."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

from openai import APIError, OpenAI

from src.config import settings

DEFAULT_MODEL = settings.openai_response_model


@dataclass(frozen=True, slots=True)
class Case:
    name: str
    prompt: str
    required_tool: str
    expected_fields: tuple[str, ...]


@dataclass
class CaseResult:
    name: str
    required_tool: str
    pass_expectation: bool = False
    response_id: str | None = None
    called_tool: str | None = None
    parsed_arguments: dict[str, Any] | None = None
    output_text: str = ""
    error: str | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify tool-call semantics for strategy patch workflow.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", default="logs/strategy_patch_tool_semantics_report.json")
    parser.add_argument("--always-zero", action="store_true")
    return parser.parse_args()


def _tools_schema() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "name": "strategy_get_dsl",
            "description": "Fetch latest strategy DSL JSON and metadata for a session-bound strategy.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "session_id": {"type": "string"},
                    "strategy_id": {"type": "string"},
                },
                "required": ["session_id", "strategy_id"],
            },
        },
        {
            "type": "function",
            "name": "strategy_patch_dsl",
            "description": "Apply RFC 6902 patch operations to an existing strategy.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "session_id": {"type": "string"},
                    "strategy_id": {"type": "string"},
                    "patch_json": {"type": "string"},
                    "expected_version": {"type": "integer"},
                },
                "required": ["session_id", "strategy_id", "patch_json"],
            },
        },
    ]


def _extract_response_text(response: Any) -> str:
    text = getattr(response, "output_text", "")
    if isinstance(text, str):
        return text
    return ""


def _extract_first_function_call(response: Any) -> tuple[str | None, dict[str, Any] | None]:
    output = getattr(response, "output", None)
    if not isinstance(output, list):
        return None, None
    for item in output:
        item_type = getattr(item, "type", None)
        if item_type != "function_call":
            continue
        name = getattr(item, "name", None)
        raw_args = getattr(item, "arguments", None)
        if not isinstance(name, str):
            continue
        if not isinstance(raw_args, str):
            return name, None
        try:
            parsed = json.loads(raw_args)
        except json.JSONDecodeError:
            return name, None
        if not isinstance(parsed, dict):
            return name, None
        return name, parsed
    return None, None


def _run_case(client: OpenAI, *, model: str, case: Case) -> CaseResult:
    result = CaseResult(name=case.name, required_tool=case.required_tool)
    try:
        response = client.responses.create(
            model=model,
            input=case.prompt,
            tools=_tools_schema(),
            tool_choice={"type": "function", "name": case.required_tool},
            timeout=60,
        )
        result.response_id = getattr(response, "id", None)
        result.output_text = _extract_response_text(response)

        tool_name, args = _extract_first_function_call(response)
        result.called_tool = tool_name
        result.parsed_arguments = args

        if tool_name != case.required_tool or not isinstance(args, dict):
            result.pass_expectation = False
            return result

        fields_ok = all(
            field in args and args[field] not in (None, "")
            for field in case.expected_fields
        )
        if not fields_ok:
            result.pass_expectation = False
            return result

        if case.required_tool == "strategy_patch_dsl":
            patch_raw = args.get("patch_json")
            if not isinstance(patch_raw, str):
                result.pass_expectation = False
                return result
            try:
                patch_ops = json.loads(patch_raw)
            except json.JSONDecodeError:
                result.pass_expectation = False
                return result
            if not isinstance(patch_ops, list) or not patch_ops:
                result.pass_expectation = False
                return result
            for operation in patch_ops:
                if not isinstance(operation, dict):
                    result.pass_expectation = False
                    return result
                if not isinstance(operation.get("op"), str):
                    result.pass_expectation = False
                    return result
                if not isinstance(operation.get("path"), str):
                    result.pass_expectation = False
                    return result

        result.pass_expectation = True
    except APIError as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        result.pass_expectation = False
    except Exception as exc:  # noqa: BLE001
        result.error = f"{type(exc).__name__}: {exc}"
        result.pass_expectation = False
    return result


def main() -> int:
    args = _parse_args()
    api_key = os.getenv("OPENAI_API_KEY") or settings.openai_api_key
    if not api_key:
        print("OPENAI_API_KEY is missing.", file=sys.stderr)
        return 2

    cases = [
        Case(
            name="semantic_get_dsl_call",
            required_tool="strategy_get_dsl",
            expected_fields=("session_id", "strategy_id"),
            prompt=(
                "Call strategy_get_dsl once to fetch latest DSL for "
                "session_id=11111111-1111-1111-1111-111111111111 and "
                "strategy_id=22222222-2222-2222-2222-222222222222."
            ),
        ),
        Case(
            name="semantic_patch_dsl_call",
            required_tool="strategy_patch_dsl",
            expected_fields=("session_id", "strategy_id", "patch_json", "expected_version"),
            prompt=(
                "Call strategy_patch_dsl once for session_id=11111111-1111-1111-1111-111111111111 "
                "and strategy_id=22222222-2222-2222-2222-222222222222. "
                "Update long stop pct from 0.02 to 0.015 and set expected_version=12."
            ),
        ),
    ]

    client = OpenAI(api_key=api_key)
    started_at = datetime.now(UTC).isoformat()

    results = [_run_case(client, model=args.model, case=case) for case in cases]
    passed = sum(1 for item in results if item.pass_expectation)

    report = {
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "model": args.model,
        "total_cases": len(results),
        "passed_cases": passed,
        "failed_cases": len(results) - passed,
        "results": [item.to_json() for item in results],
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)

    output_path = args.output
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(text)

    if args.always_zero:
        return 0
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
