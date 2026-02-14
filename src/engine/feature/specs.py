"""Shared factor specifications for the engine feature layer."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class FactorKind(StrEnum):
    """Top-level factor namespaces supported by the platform."""

    INDICATOR = "indicator"
    PRICE_EVENT = "price_event"
    ML_SIGNAL = "ml_signal"
    CUSTOM = "custom"


@dataclass(frozen=True, slots=True)
class FactorKey:
    """Stable key for a registered factor."""

    kind: str
    name: str

    @classmethod
    def build(cls, *, kind: str | FactorKind, name: str) -> "FactorKey":
        kind_value = kind.value if isinstance(kind, FactorKind) else str(kind).strip().lower()
        return cls(kind=kind_value, name=name.strip().lower())


@dataclass(frozen=True, slots=True)
class FactorRecord:
    """A registry record for one factor implementation."""

    key: FactorKey
    metadata: Any
    calculator: Any = None
    custom_instance: Any = None
