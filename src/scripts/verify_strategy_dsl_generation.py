#!/usr/bin/env python3
"""Validate real OpenAI-generated strategy DSL against local validators."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from openai import APIError, OpenAI

from src.config import settings
from src.engine.strategy import (
    EXAMPLE_PATH,
    load_strategy_payload,
    validate_strategy_payload,
)

DEFAULT_MODEL = settings.openai_response_model


@dataclass
class DslPromptCase:
    name: str
    prompt: str
    expect_valid: bool
    expected_error_codes: list[str] = field(default_factory=list)


@dataclass
class DslPromptResult:
    case_name: str
    expect_valid: bool
    expected_error_codes: list[str]
    model: str
    ok: bool = False
    response_id: str | None = None
    raw_text: str = ""
    parsed_json_ok: bool = False
    parse_error: str | None = None
    validation_is_valid: bool = False
    validation_error_codes: list[str] = field(default_factory=list)
    validation_errors: list[dict[str, Any]] = field(default_factory=list)
    pass_expectation: bool = False
    api_error: str | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _build_cases() -> list[DslPromptCase]:
    base_payload = load_strategy_payload(EXAMPLE_PATH)
    base_json = json.dumps(base_payload, ensure_ascii=False)
    guard = (
        "Return ONLY a JSON object. Keep the same top-level keys and overall structure as the template unless the case "
        "explicitly asks you to violate one rule."
    )

    return [
        DslPromptCase(
            name="valid_strategy",
            expect_valid=True,
            prompt=(
                f"{guard} Produce a VALID strategy DSL by minimally editing this template to keep it valid. "
                f"Template: {base_json}"
            ),
        ),
        DslPromptCase(
            name="missing_required_timeframe",
            expect_valid=False,
            expected_error_codes=["MISSING_REQUIRED_FIELD"],
            prompt=(
                f"{guard} Return JSON derived from this template, but intentionally remove top-level timeframe. "
                f"Template: {base_json}"
            ),
        ),
        DslPromptCase(
            name="timeframe_type_mismatch",
            expect_valid=False,
            expected_error_codes=["TYPE_MISMATCH"],
            prompt=(
                f"{guard} Return JSON derived from this template, but intentionally set top-level timeframe to number 4 "
                "instead of a string. "
                f"Template: {base_json}"
            ),
        ),
        DslPromptCase(
            name="invalid_not_structure",
            expect_valid=False,
            expected_error_codes=["SCHEMA_VALIDATION_ERROR"],
            prompt=(
                f"{guard} Return JSON derived from this template, but intentionally make one condition invalid by using "
                "{\"not\": [ ... ]} (array under not). "
                f"Template: {base_json}"
            ),
        ),
        DslPromptCase(
            name="unknown_factor_ref",
            expect_valid=False,
            expected_error_codes=["UNKNOWN_FACTOR_REF"],
            prompt=(
                f"{guard} Return JSON derived from this template, but intentionally replace one entry ref with "
                "'ema_fast' without defining ema_fast in factors. "
                f"Template: {base_json}"
            ),
        ),
        DslPromptCase(
            name="invalid_output_name",
            expect_valid=False,
            expected_error_codes=["INVALID_OUTPUT_NAME"],
            prompt=(
                f"{guard} Return JSON derived from this template, but intentionally use invalid output ref "
                "macd_12_26_9.badline in one cmp condition. "
                f"Template: {base_json}"
            ),
        ),
        DslPromptCase(
            name="temporal_condition_reserved",
            expect_valid=False,
            expected_error_codes=["TEMPORAL_NOT_SUPPORTED", "SCHEMA_VALIDATION_ERROR"],
            prompt=(
                f"{guard} Return JSON derived from this template, but intentionally replace one entry condition "
                "with temporal node in this exact schema shape: "
                "{\"temporal\":{\"type\":\"within_bars\",\"bars\":3,\"condition\":{...}}}. "
                f"Template: {base_json}"
            ),
        ),
        DslPromptCase(
            name="bracket_rr_conflict",
            expect_valid=False,
            expected_error_codes=["SCHEMA_VALIDATION_ERROR", "BRACKET_RR_CONFLICT"],
            prompt=(
                f"{guard} Return JSON derived from this template, but intentionally make one bracket_rr exit invalid by "
                "setting both stop and take in that same bracket_rr rule. "
                f"Template: {base_json}"
            ),
        ),
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify OpenAI DSL generation with real endpoint")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-cases", type=int, default=0, help="0 means run all cases")
    parser.add_argument("--output", default="", help="Optional JSON report path")
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=60.0,
        help="Timeout per OpenAI response create call.",
    )
    parser.add_argument("--always-zero", action="store_true", help="Always return status code 0")
    return parser.parse_args()


def _extract_json_text(raw: str) -> str:
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, flags=re.DOTALL)
    if fenced:
        raw = fenced.group(1)

    start = raw.find("{")
    if start == -1:
        return raw

    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(raw)):
        ch = raw[idx]
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return raw[start : idx + 1]

    return raw[start:]


def _get_response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    chunks: list[str] = []
    output_items = getattr(response, "output", None)
    if isinstance(output_items, list):
        for item in output_items:
            content = getattr(item, "content", None)
            if not isinstance(content, list):
                continue
            for part in content:
                text = getattr(part, "text", None)
                if isinstance(text, str) and text:
                    chunks.append(text)
    return "\n".join(chunks).strip()


def run_case(
    client: OpenAI,
    case: DslPromptCase,
    model: str,
    *,
    request_timeout_seconds: float,
) -> DslPromptResult:
    result = DslPromptResult(
        case_name=case.name,
        expect_valid=case.expect_valid,
        expected_error_codes=case.expected_error_codes,
        model=model,
    )

    try:
        response = client.responses.create(
            model=model,
            input=case.prompt,
            timeout=request_timeout_seconds,
        )
        result.response_id = getattr(response, "id", None)
        raw_text = _get_response_text(response)
        result.raw_text = raw_text

        json_text = _extract_json_text(raw_text)
        payload = json.loads(json_text)
        if not isinstance(payload, dict):
            raise ValueError("Model output is not a JSON object")
        result.parsed_json_ok = True

        validation = validate_strategy_payload(payload)
        result.validation_is_valid = validation.is_valid
        result.validation_error_codes = [item.code for item in validation.errors]
        result.validation_errors = [
            {
                "code": item.code,
                "message": item.message,
                "path": item.path,
            }
            for item in validation.errors
        ]

        if case.expect_valid:
            result.pass_expectation = validation.is_valid
        else:
            result.pass_expectation = not validation.is_valid and any(
                code in result.validation_error_codes for code in case.expected_error_codes
            )

        result.ok = True
        return result
    except (json.JSONDecodeError, ValueError) as exc:
        result.parse_error = str(exc)
        result.pass_expectation = False
        return result
    except APIError as exc:
        result.api_error = f"{type(exc).__name__}: {exc}"
        result.pass_expectation = False
        return result


def main() -> int:
    args = _parse_args()

    api_key = os.getenv("OPENAI_API_KEY") or settings.openai_api_key
    if not api_key:
        print("OPENAI_API_KEY is missing.", file=sys.stderr)
        return 2

    client = OpenAI(api_key=api_key)
    cases = _build_cases()
    if args.max_cases > 0:
        cases = cases[: args.max_cases]

    started_at = datetime.now(UTC).isoformat()
    results: list[DslPromptResult] = []
    for index, case in enumerate(cases, start=1):
        print(
            f"[verify] running case {index}/{len(cases)}: {case.name}",
            file=sys.stderr,
        )
        result = run_case(
            client,
            case,
            model=args.model,
            request_timeout_seconds=args.request_timeout_seconds,
        )
        results.append(result)
        print(
            f"[verify] done case {index}/{len(cases)}: {case.name} pass={result.pass_expectation}",
            file=sys.stderr,
        )
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

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")

    if args.always_zero:
        return 0
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
