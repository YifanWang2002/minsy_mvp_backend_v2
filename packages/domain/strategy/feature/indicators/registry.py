"""Indicator registry adapter backed by the global feature registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Type

from packages.domain.strategy.feature.registry import FeatureRegistry
from packages.domain.strategy.feature.specs import FactorKind

from .base import BaseIndicator, IndicatorCategory, IndicatorMetadata


class IndicatorRegistry:
    """Compatibility adapter for indicator modules.

    Indicator implementations can keep importing ``IndicatorRegistry`` while the
    underlying data is stored in ``FeatureRegistry`` for future non-indicator
    factor kinds.
    """

    @classmethod
    def register(
        cls,
        metadata: IndicatorMetadata,
        calculator: Callable[..., Any] | None = None,
    ) -> None:
        FeatureRegistry.register(
            kind=FactorKind.INDICATOR,
            name=metadata.name,
            metadata=metadata,
            calculator=calculator,
        )

    @classmethod
    def register_custom(cls, indicator_class: Type[BaseIndicator]):
        instance = indicator_class()
        metadata = instance.metadata
        FeatureRegistry.register(
            kind=FactorKind.INDICATOR,
            name=metadata.name,
            metadata=metadata,
            custom_instance=instance,
        )
        return indicator_class

    @classmethod
    def get(cls, name: str) -> IndicatorMetadata | None:
        metadata = FeatureRegistry.get_metadata(kind=FactorKind.INDICATOR, name=name)
        if metadata is None:
            return None
        return metadata

    @classmethod
    def get_calculator(cls, name: str) -> Callable[..., Any] | None:
        return FeatureRegistry.get_calculator(kind=FactorKind.INDICATOR, name=name)

    @classmethod
    def get_custom(cls, name: str) -> BaseIndicator | None:
        custom = FeatureRegistry.get_custom(kind=FactorKind.INDICATOR, name=name)
        if custom is None:
            return None
        return custom

    @classmethod
    def has(cls, name: str) -> bool:
        return FeatureRegistry.has(kind=FactorKind.INDICATOR, name=name)

    @classmethod
    def is_custom(cls, name: str) -> bool:
        return FeatureRegistry.is_custom(kind=FactorKind.INDICATOR, name=name)

    @classmethod
    def list_all(cls) -> list[str]:
        return FeatureRegistry.list_names(kind=FactorKind.INDICATOR)

    @classmethod
    def list_by_category(cls, category: IndicatorCategory) -> list[str]:
        names: list[str] = []
        for name in cls.list_all():
            metadata = cls.get(name)
            if metadata and metadata.category == category:
                names.append(name)
        return sorted(names)

    @classmethod
    def get_categories(cls) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        for category in IndicatorCategory:
            indicators = cls.list_by_category(category)
            if indicators:
                result[category.value] = indicators
        return result

    @classmethod
    def get_all_signatures(
        cls,
        exclude_categories: list[IndicatorCategory] | None = None,
        include_categories: list[IndicatorCategory] | None = None,
    ) -> dict[str, dict[str, Any]]:
        signatures: dict[str, dict[str, Any]] = {}
        for name in cls.list_all():
            metadata = cls.get(name)
            if metadata is None:
                continue
            if exclude_categories and metadata.category in exclude_categories:
                continue
            if include_categories and metadata.category not in include_categories:
                continue
            signatures[name] = metadata.get_signature()
        return signatures

    @classmethod
    def get_category_signatures(cls, category: IndicatorCategory) -> dict[str, dict[str, Any]]:
        return {
            name: metadata.get_signature()
            for name in cls.list_all()
            if (metadata := cls.get(name)) is not None and metadata.category == category
        }

    @classmethod
    def print_available(cls) -> str:
        lines = ["=" * 60, "Available Technical Indicators", "=" * 60, ""]

        for category in IndicatorCategory:
            indicators = cls.list_by_category(category)
            if not indicators:
                continue
            lines.append(f"\n{category.value.upper()} ({len(indicators)} indicators)")
            lines.append("-" * 40)
            for name in indicators:
                metadata = cls.get(name)
                if metadata is None:
                    continue
                lines.append(f"  {name:20} - {metadata.full_name}")

        lines.append("")
        lines.append(f"Total: {len(cls.list_all())} indicators")
        lines.append("=" * 60)
        return "\n".join(lines)

    @classmethod
    def get_indicator_info(cls, name: str) -> str | None:
        metadata = cls.get(name)
        if metadata is None:
            return None

        lines = [
            f"Indicator: {metadata.name}",
            f"Full Name: {metadata.full_name}",
            f"Category: {metadata.category.value}",
            f"Description: {metadata.description}",
            "",
            "Parameters:",
        ]

        for param in metadata.params:
            param_info = f"  {param.name}: {param.type}"
            if param.default is not None:
                param_info += f" = {param.default}"
            if param.min_value is not None or param.max_value is not None:
                param_info += f" (range: {param.min_value or '-∞'} to {param.max_value or '∞'})"
            if param.description:
                param_info += f"\n    {param.description}"
            lines.append(param_info)

        lines.append("")
        lines.append("Outputs:")
        for output in metadata.outputs:
            lines.append(f"  {output.name}: {output.description}")

        lines.append("")
        lines.append(f"Required columns: {metadata.required_columns}")
        lines.append(f"TA-Lib function: {metadata.talib_func or 'N/A'}")
        lines.append(f"pandas-ta function: {metadata.pandas_ta_func or 'N/A'}")
        return "\n".join(lines)

    @classmethod
    def clear(cls) -> None:
        FeatureRegistry.clear(kind=FactorKind.INDICATOR)
