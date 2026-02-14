"""Unified factor registry for indicators and future factor families."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.engine.feature.specs import FactorKey, FactorKind, FactorRecord


class FeatureRegistry:
    """Global registry keyed by ``(factor_kind, factor_name)``."""

    _records: dict[FactorKey, FactorRecord] = {}

    @classmethod
    def register(
        cls,
        *,
        kind: str | FactorKind,
        name: str,
        metadata: Any,
        calculator: Callable[..., Any] | None = None,
        custom_instance: Any = None,
    ) -> None:
        key = FactorKey.build(kind=kind, name=name)
        cls._records[key] = FactorRecord(
            key=key,
            metadata=metadata,
            calculator=calculator,
            custom_instance=custom_instance,
        )

    @classmethod
    def get_record(cls, *, kind: str | FactorKind, name: str) -> FactorRecord | None:
        key = FactorKey.build(kind=kind, name=name)
        return cls._records.get(key)

    @classmethod
    def get_metadata(cls, *, kind: str | FactorKind, name: str) -> Any | None:
        record = cls.get_record(kind=kind, name=name)
        return record.metadata if record else None

    @classmethod
    def get_calculator(
        cls,
        *,
        kind: str | FactorKind,
        name: str,
    ) -> Callable[..., Any] | None:
        record = cls.get_record(kind=kind, name=name)
        if record is None:
            return None
        calculator = record.calculator
        if calculator is None or not callable(calculator):
            return None
        return calculator

    @classmethod
    def get_custom(cls, *, kind: str | FactorKind, name: str) -> Any | None:
        record = cls.get_record(kind=kind, name=name)
        return record.custom_instance if record else None

    @classmethod
    def has(cls, *, kind: str | FactorKind, name: str) -> bool:
        return cls.get_record(kind=kind, name=name) is not None

    @classmethod
    def is_custom(cls, *, kind: str | FactorKind, name: str) -> bool:
        record = cls.get_record(kind=kind, name=name)
        return bool(record and record.custom_instance is not None)

    @classmethod
    def list_records(cls, *, kind: str | FactorKind | None = None) -> list[FactorRecord]:
        if kind is None:
            return list(cls._records.values())
        kind_value = kind.value if isinstance(kind, FactorKind) else str(kind).strip().lower()
        return [record for record in cls._records.values() if record.key.kind == kind_value]

    @classmethod
    def list_names(cls, *, kind: str | FactorKind) -> list[str]:
        return sorted(record.key.name for record in cls.list_records(kind=kind))

    @classmethod
    def clear(cls, *, kind: str | FactorKind | None = None) -> None:
        if kind is None:
            cls._records.clear()
            return

        kind_value = kind.value if isinstance(kind, FactorKind) else str(kind).strip().lower()
        to_delete = [key for key in cls._records if key.kind == kind_value]
        for key in to_delete:
            cls._records.pop(key, None)
