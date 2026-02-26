"""Build typed model objects from validated strategy DSL payloads."""

from __future__ import annotations

from typing import Any

from packages.domain.strategy.models import (
    FactorDefinition,
    ParsedStrategyDsl,
    StrategyInfo,
    StrategyUniverse,
)


def build_parsed_strategy(payload: dict[str, Any]) -> ParsedStrategyDsl:
    raw_factors = payload.get("factors", {})
    factor_map: dict[str, FactorDefinition] = {}
    if isinstance(raw_factors, dict):
        for factor_id, factor_def in raw_factors.items():
            if not isinstance(factor_id, str) or factor_id.startswith("x-"):
                continue
            if not isinstance(factor_def, dict):
                continue
            outputs_value = factor_def.get("outputs")
            outputs: tuple[str, ...] = tuple(
                item
                for item in (outputs_value if isinstance(outputs_value, list) else [])
                if isinstance(item, str)
            )
            factor_map[factor_id] = FactorDefinition(
                factor_id=factor_id,
                factor_type=str(factor_def.get("type", "")).strip().lower(),
                params=dict(factor_def.get("params", {})) if isinstance(factor_def.get("params"), dict) else {},
                outputs=outputs,
            )

    strategy_section = payload.get("strategy", {})
    universe_section = payload.get("universe", {})
    trade_section = payload.get("trade", {})

    return ParsedStrategyDsl(
        dsl_version=str(payload.get("dsl_version", "")).strip(),
        strategy=StrategyInfo(
            name=str(strategy_section.get("name", "")).strip(),
            description=str(strategy_section.get("description", "")).strip(),
        ),
        universe=StrategyUniverse(
            market=str(universe_section.get("market", "")).strip(),
            tickers=tuple(
                item
                for item in (universe_section.get("tickers", []) if isinstance(universe_section, dict) else [])
                if isinstance(item, str)
            ),
            timeframe=str(payload.get("timeframe", "")).strip(),
        ),
        factors=factor_map,
        trade=dict(trade_section) if isinstance(trade_section, dict) else {},
        raw=dict(payload),
    )
