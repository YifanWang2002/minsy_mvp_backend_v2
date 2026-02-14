#!/usr/bin/env python3
"""Verify OpenAI tool-call semantics for strategy version history tools."""

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
        description="Verify tool-call semantics for strategy revision workflow.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", default="logs/strategy_revision_tool_semantics_report.json")
    parser.add_argument("--always-zero", action="store_true")
    return parser.parse_args()


def _tools_schema() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "name": "strategy_list_versions",
            "description": "List strategy revision metadata in reverse-chronological order.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "session_id": {"type": "string"},
                    "strategy_id": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["session_id", "strategy_id"],
            },
        },
        {
            "type": "function",
            "name": "strategy_get_version_dsl",
            "description": "Fetch one historical strategy DSL by version number.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "session_id": {"type": "string"},
                    "strategy_id": {"type": "string"},
                    "version": {"type": "integer"},
                },
                "required": ["session_id", "strategy_id", "version"],
            },
        },
        {
            "type": "function",
            "name": "strategy_diff_versions",
            "description": "Compute RFC 6902 patch operations from one strategy version to another.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "session_id": {"type": "string"},
                    "strategy_id": {"type": "string"},
                    "from_version": {"type": "integer"},
                    "to_version": {"type": "integer"},
                },
                "required": ["session_id", "strategy_id", "from_version", "to_version"],
            },
        },
        {
            "type": "function",
            "name": "strategy_rollback_dsl",
            "description": "Rollback to a historical strategy version by creating a new head version.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "session_id": {"type": "string"},
                    "strategy_id": {"type": "string"},
                    "target_version": {"type": "integer"},
                    "expected_version": {"type": "integer"},
                },
                "required": ["session_id", "strategy_id", "target_version"],
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
        if getattr(item, "type", None) != "function_call":
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


def _validate_case_fields(case: Case, args: dict[str, Any]) -> bool:
    for field in case.expected_fields:
        if field not in args or args[field] in (None, ""):
            return False

    if case.required_tool == "strategy_list_versions":
        limit = args.get("limit")
        if limit is not None and (not isinstance(limit, int) or limit <= 0):
            return False
    if case.required_tool == "strategy_get_version_dsl":
        if not isinstance(args.get("version"), int) or int(args["version"]) <= 0:
            return False
    if case.required_tool == "strategy_diff_versions":
        if not isinstance(args.get("from_version"), int):
            return False
        if not isinstance(args.get("to_version"), int):
            return False
        if int(args["from_version"]) <= 0 or int(args["to_version"]) <= 0:
            return False
    if case.required_tool == "strategy_rollback_dsl":
        if not isinstance(args.get("target_version"), int) or int(args["target_version"]) <= 0:
            return False
        expected = args.get("expected_version")
        if expected is not None and (not isinstance(expected, int) or int(expected) < 0):
            return False
    return True


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

        result.pass_expectation = _validate_case_fields(case, args)
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
            name="semantic_list_versions_call",
            required_tool="strategy_list_versions",
            expected_fields=("session_id", "strategy_id"),
            prompt=(
                "Call strategy_list_versions once for "
                "session_id=11111111-1111-1111-1111-111111111111 and "
                "strategy_id=22222222-2222-2222-2222-222222222222 with limit=20."
            ),
        ),
        Case(
            name="semantic_get_version_dsl_call",
            required_tool="strategy_get_version_dsl",
            expected_fields=("session_id", "strategy_id", "version"),
            prompt=(
                "Call strategy_get_version_dsl once for "
                "session_id=11111111-1111-1111-1111-111111111111 and "
                "strategy_id=22222222-2222-2222-2222-222222222222 at version=3."
            ),
        ),
        Case(
            name="semantic_diff_versions_call",
            required_tool="strategy_diff_versions",
            expected_fields=("session_id", "strategy_id", "from_version", "to_version"),
            prompt=(
                "Call strategy_diff_versions once for "
                "session_id=11111111-1111-1111-1111-111111111111 and "
                "strategy_id=22222222-2222-2222-2222-222222222222 with "
                "from_version=2 and to_version=5."
            ),
        ),
        Case(
            name="semantic_rollback_call",
            required_tool="strategy_rollback_dsl",
            expected_fields=("session_id", "strategy_id", "target_version", "expected_version"),
            prompt=(
                "Call strategy_rollback_dsl once for "
                "session_id=11111111-1111-1111-1111-111111111111 and "
                "strategy_id=22222222-2222-2222-2222-222222222222 with "
                "target_version=4 and expected_version=9."
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
