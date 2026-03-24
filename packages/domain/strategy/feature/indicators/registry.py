"""Indicator registry adapter backed by the global feature registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, Type
import warnings

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
        *,
        registration_mode: Literal["legacy", "decorator"] = "legacy",
    ) -> None:
        cls._validate_metadata(metadata)
        existing = cls.get(metadata.name)
        if existing is not None and existing.get_signature() != metadata.get_signature():
            raise ValueError(
                f"Duplicate indicator name with conflicting metadata: '{metadata.name}'"
            )
        if registration_mode == "legacy":
            warnings.warn(
                (
                    f"Indicator '{metadata.name}' registered via legacy register block. "
                    "Please migrate to @indicator decorator."
                ),
                DeprecationWarning,
                stacklevel=2,
            )
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
        cls._validate_metadata(metadata)
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

    @classmethod
    def list_multi_output(cls) -> list[str]:
        names: list[str] = []
        for name in cls.list_all():
            metadata = cls.get(name)
            if metadata is None:
                continue
            if len(metadata.outputs) >= 2:
                names.append(name)
        return sorted(names)

    @classmethod
    def validate_registry(cls) -> tuple[bool, list[str]]:
        errors: list[str] = []
        seen: set[str] = set()
        for name in cls.list_all():
            metadata = cls.get(name)
            if metadata is None:
                errors.append(f"indicator '{name}' missing metadata")
                continue
            key = metadata.name.strip().lower()
            if key in seen:
                errors.append(f"duplicate indicator name '{key}'")
            seen.add(key)
            try:
                cls._validate_metadata(metadata)
            except ValueError as exc:
                errors.append(f"{name}: {exc}")
        return len(errors) == 0, errors

    @classmethod
    def _validate_metadata(cls, metadata: IndicatorMetadata) -> None:
        name = str(metadata.name).strip().lower()
        if not name:
            raise ValueError("name cannot be empty")
        if name != metadata.name:
            raise ValueError(f"name must be lowercase/trimmed: '{metadata.name}'")

        full_name = str(metadata.full_name).strip()
        if not full_name:
            raise ValueError("full_name cannot be empty")

        required_columns = [str(column).strip().lower() for column in metadata.required_columns]
        if not required_columns:
            raise ValueError("required_columns cannot be empty")
        if any(not column for column in required_columns):
            raise ValueError("required_columns cannot contain empty entries")

        status = str(metadata.status).strip().lower()
        if status not in {"active", "deprecated", "removed"}:
            raise ValueError(f"status must be one of active/deprecated/removed, got '{metadata.status}'")
        version = str(metadata.version).strip()
        if not version:
            raise ValueError("version cannot be empty")
        if status == "deprecated" and not str(metadata.deprecated_since or "").strip():
            raise ValueError("deprecated status requires deprecated_since")
        if status == "removed" and not str(metadata.remove_after or "").strip():
            raise ValueError("removed status requires remove_after")

        param_names: set[str] = set()
        for param in metadata.params:
            param_name = str(param.name).strip()
            if not param_name:
                raise ValueError(f"{name}: parameter name cannot be empty")
            lowered = param_name.lower()
            if lowered in param_names:
                raise ValueError(f"{name}: duplicated parameter '{param_name}'")
            param_names.add(lowered)
            if param.min_value is not None and param.max_value is not None:
                if float(param.min_value) > float(param.max_value):
                    raise ValueError(
                        f"{name}.{param_name}: min_value cannot be greater than max_value"
                    )
            if (
                param.default is not None
                and isinstance(param.default, int | float)
                and not isinstance(param.default, bool)
            ):
                default_numeric = float(param.default)
                if param.min_value is not None and default_numeric < float(param.min_value):
                    raise ValueError(
                        f"{name}.{param_name}: default < min_value"
                    )
                if param.max_value is not None and default_numeric > float(param.max_value):
                    raise ValueError(
                        f"{name}.{param_name}: default > max_value"
                    )

        output_names: set[str] = set()
        for output in metadata.outputs:
            output_name = str(output.name).strip()
            if not output_name:
                raise ValueError(f"{name}: output name cannot be empty")
            lowered = output_name.lower()
            if lowered in output_names:
                raise ValueError(f"{name}: duplicated output '{output_name}'")
            output_names.add(lowered)
