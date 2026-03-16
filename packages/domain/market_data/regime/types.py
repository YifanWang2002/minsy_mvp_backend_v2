"""Typed contracts for pre-strategy regime analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

StrategyFamilyId = Literal[
    "trend_continuation",
    "mean_reversion",
    "volatility_regime",
]


@dataclass(slots=True, frozen=True)
class TimeframePlan:
    """Mapped timeframe candidates for pre-strategy regime probing."""

    primary: str
    secondary: str
    candidates: tuple[str, str]
    mapping_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary": self.primary,
            "secondary": self.secondary,
            "candidates": list(self.candidates),
            "mapping_reason": self.mapping_reason,
        }


@dataclass(slots=True, frozen=True)
class FamilyScores:
    """Programmatic probabilities for three strategy families."""

    trend_continuation: float
    mean_reversion: float
    volatility_regime: float
    recommended_family: StrategyFamilyId
    confidence: float
    evidence_for: dict[str, list[str]] = field(default_factory=dict)
    evidence_against: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["recommended_family"] = self.recommended_family
        return payload


@dataclass(slots=True, frozen=True)
class RegimeSnapshot:
    """Structured analysis payload for one symbol/timeframe."""

    timeframe: str
    bars: int
    source_mode: str
    source_label: str
    asof_utc: str
    snapshot: dict[str, Any]
    family_scores: FamilyScores
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "timeframe": self.timeframe,
            "bars": self.bars,
            "source_mode": self.source_mode,
            "source_label": self.source_label,
            "asof_utc": self.asof_utc,
            "snapshot": self.snapshot,
            "family_scores": self.family_scores.to_dict(),
            "summary": self.summary,
        }

