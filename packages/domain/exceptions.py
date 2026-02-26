"""Domain-layer exceptions shared across apps and transports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class DomainError(Exception):
    """Framework-agnostic domain exception carrying HTTP-like error metadata."""

    status_code: int
    code: str
    message: str

    @property
    def detail(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
        }

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"

