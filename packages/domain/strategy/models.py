"""Typed model objects for validated strategy DSL payloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class FactorDefinition:
    """Single factor specification."""

    factor_id: str
    factor_type: str
    params: dict[str, Any]
    outputs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StrategyInfo:
    """Basic strategy metadata from DSL."""

    name: str
    description: str


@dataclass(frozen=True, slots=True)
class StrategyUniverse:
    """Universe and bar resolution settings."""

    market: str
    tickers: tuple[str, ...]
    timeframe: str


@dataclass(frozen=True, slots=True)
class ParsedStrategyDsl:
    """A parsed, validation-safe strategy DSL object."""

    dsl_version: str
    strategy: StrategyInfo
    universe: StrategyUniverse
    factors: dict[str, FactorDefinition]
    trade: dict[str, Any]
    raw: dict[str, Any]
