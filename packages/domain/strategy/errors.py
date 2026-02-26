"""Error models for strategy DSL validation/parsing."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True, slots=True)
class StrategyDslError:
    """A single validation/parsing error item."""

    code: str
    message: str
    path: str = ""
    value: Any = None
    stage: str = ""
    pointer: str = ""
    expected: Any = None
    actual: Any = None
    suggestion: str = ""


@dataclass(frozen=True, slots=True)
class StrategyDslValidationResult:
    """Validation response for schema + semantic phases."""

    is_valid: bool
    errors: tuple[StrategyDslError, ...] = ()


class StrategyDslValidationError(ValueError):
    """Raised when a DSL payload does not pass validation."""

    def __init__(self, errors: list[StrategyDslError]) -> None:
        self.errors = errors
        summary = "; ".join(f"{item.code}@{item.path}: {item.message}" for item in errors)
        super().__init__(summary or "Strategy DSL validation failed")


# Backward-compatible alias used by existing call sites/tests.
StrategyDslValidationException = StrategyDslValidationError


def json_path_to_pointer(path: str) -> str:
    """Convert a simple '$.a.b[0]' jsonpath into JSON Pointer '/a/b/0'."""
    if not isinstance(path, str):
        return ""
    text = path.strip()
    if not text or text == "$":
        return ""
    if not text.startswith("$"):
        return ""

    pointer_parts: list[str] = []
    token = ""
    index_mode = False
    index_buffer = ""

    for ch in text[1:]:
        if index_mode:
            if ch == "]":
                if index_buffer:
                    pointer_parts.append(index_buffer)
                index_mode = False
                index_buffer = ""
                continue
            index_buffer += ch
            continue

        if ch == ".":
            if token:
                pointer_parts.append(token)
                token = ""
            continue
        if ch == "[":
            if token:
                pointer_parts.append(token)
                token = ""
            index_mode = True
            index_buffer = ""
            continue
        token += ch

    if token:
        pointer_parts.append(token)

    if not pointer_parts:
        return ""

    escaped = [part.replace("~", "~0").replace("/", "~1") for part in pointer_parts]
    return "/" + "/".join(escaped)


def normalize_strategy_error(
    error: StrategyDslError,
    *,
    stage: str = "",
    expected: Any = None,
    actual: Any = None,
    suggestion: str = "",
) -> StrategyDslError:
    """Fill optional metadata fields without mutating original immutable object."""
    resolved_stage = error.stage or stage
    resolved_pointer = error.pointer or json_path_to_pointer(error.path)
    resolved_expected = error.expected if error.expected is not None else expected
    resolved_actual = error.actual if error.actual is not None else actual
    resolved_suggestion = error.suggestion or suggestion

    if (
        resolved_stage == error.stage
        and resolved_pointer == error.pointer
        and resolved_expected is error.expected
        and resolved_actual is error.actual
        and resolved_suggestion == error.suggestion
    ):
        return error

    return replace(
        error,
        stage=resolved_stage,
        pointer=resolved_pointer,
        expected=resolved_expected,
        actual=resolved_actual,
        suggestion=resolved_suggestion,
    )
