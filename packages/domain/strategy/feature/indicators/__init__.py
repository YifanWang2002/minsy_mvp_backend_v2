"""Technical Indicators Module.

This module provides a unified interface for calculating technical indicators
using TA-Lib (preferred) or pandas-ta as a fallback.

Usage:
    from backend.indicators import calculate, get_categories, print_available
    
    # Calculate a single indicator
    sma = calculate("sma", data, length=20)
    
    # Calculate a multi-output indicator
    macd = calculate("macd", data, fast=12, slow=26, signal=9)
    macd_line = calculate("macd", data, output="MACD")  # Get specific output
    
    # List available indicators
    print(print_available())
    
    # Get categories
    categories = get_categories()
"""

# Import base classes
from .base import (
    BaseIndicator,
    IndicatorCategory,
    IndicatorMetadata,
    IndicatorOutput,
    IndicatorParam,
    IndicatorSource,
)

# Import registry
from .registry import IndicatorRegistry

# Import wrapper
from .wrapper import (
    IndicatorWrapper,
    calculate,
    get_wrapper,
    is_pandas_ta_available,
    is_talib_available,
)

# Register all built-in indicators by importing the categories module
# This triggers the registration of all indicators
from . import categories  # noqa: F401

# Convenience functions
def get_categories() -> dict[str, list[str]]:
    """Get all indicator categories with their indicators."""
    return IndicatorRegistry.get_categories()


def print_available() -> str:
    """Print all available indicators grouped by category."""
    return IndicatorRegistry.print_available()


def get_indicator_info(name: str) -> str | None:
    """Get detailed info for a single indicator."""
    return IndicatorRegistry.get_indicator_info(name)


def list_indicators(category: IndicatorCategory | None = None) -> list[str]:
    """List available indicators, optionally filtered by category."""
    if category is None:
        return IndicatorRegistry.list_all()
    return IndicatorRegistry.list_by_category(category)


def get_signature(name: str) -> dict | None:
    """Get the signature/metadata for an indicator."""
    meta = IndicatorRegistry.get(name)
    if meta:
        return meta.get_signature()
    return None


def get_all_signatures(
    exclude_categories: list[IndicatorCategory] | None = None,
    include_categories: list[IndicatorCategory] | None = None
) -> dict[str, dict]:
    """Get signatures for all indicators with optional filtering.
    
    Args:
        exclude_categories: List of categories to exclude (e.g., [IndicatorCategory.CANDLE, IndicatorCategory.UTILS])
        include_categories: List of categories to include (if specified, only these are returned)
        
    Note: If both are specified, exclude_categories takes precedence.
    
    Example:
        # Get all indicators except candle patterns and utils
        signatures = get_all_signatures(exclude_categories=[IndicatorCategory.CANDLE, IndicatorCategory.UTILS])
        
        # Get only momentum and overlap indicators
        signatures = get_all_signatures(include_categories=[IndicatorCategory.MOMENTUM, IndicatorCategory.OVERLAP])
    """
    return IndicatorRegistry.get_all_signatures(
        exclude_categories=exclude_categories,
        include_categories=include_categories
    )


__all__ = [
    # Base classes
    "BaseIndicator",
    "IndicatorCategory",
    "IndicatorMetadata",
    "IndicatorOutput",
    "IndicatorParam",
    "IndicatorSource",
    # Registry
    "IndicatorRegistry",
    # Wrapper
    "IndicatorWrapper",
    "calculate",
    "get_wrapper",
    "is_talib_available",
    "is_pandas_ta_available",
    # Convenience functions
    "get_categories",
    "print_available",
    "get_indicator_info",
    "list_indicators",
    "get_signature",
    "get_all_signatures",
]
