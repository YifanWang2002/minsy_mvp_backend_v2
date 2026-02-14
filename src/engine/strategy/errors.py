"""Error models for strategy DSL validation/parsing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class StrategyDslError:
    """A single validation/parsing error item."""

    code: str
    message: str
    path: str = ""
    value: Any = None


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
