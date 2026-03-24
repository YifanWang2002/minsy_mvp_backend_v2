"""Decorator helpers for indicator registration."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .base import IndicatorCategory, IndicatorMetadata, IndicatorOutput, IndicatorParam
from .registry import IndicatorRegistry


def indicator(
    *,
    name: str | None = None,
    full_name: str | None = None,
    category: IndicatorCategory | None = None,
    description: str = "",
    params: Sequence[IndicatorParam] | None = None,
    outputs: Sequence[IndicatorOutput] | None = None,
    talib_func: str | None = None,
    pandas_ta_func: str | None = None,
    required_columns: Sequence[str] | None = None,
    metadata: IndicatorMetadata | None = None,
) -> Any:
    """Register one indicator function via decorator.

    Args:
        metadata: Optional prebuilt metadata. If provided, all other metadata
            arguments are ignored except registration behavior.
    """

    resolved_metadata = metadata
    if resolved_metadata is None:
        if category is None:
            raise ValueError("indicator decorator requires `category` when metadata is not provided")
        indicator_name = str(name or "").strip().lower()
        if not indicator_name:
            raise ValueError("indicator decorator requires non-empty `name`")
        display_name = str(full_name or "").strip()
        if not display_name:
            raise ValueError("indicator decorator requires non-empty `full_name`")
        resolved_metadata = IndicatorMetadata(
            name=indicator_name,
            full_name=display_name,
            category=category,
            description=str(description or ""),
            params=list(params or []),
            outputs=list(outputs or []),
            talib_func=talib_func,
            pandas_ta_func=pandas_ta_func,
            required_columns=list(required_columns or ["close"]),
        )

    def _decorator(func: Any) -> Any:
        IndicatorRegistry.register(
            resolved_metadata,
            func,
            registration_mode="decorator",
        )
        setattr(func, "__indicator_metadata__", resolved_metadata)
        return func

    return _decorator


__all__ = ["indicator"]
