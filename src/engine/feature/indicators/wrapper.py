"""Unified indicator calculation wrapper.

This module provides a unified interface for calculating indicators using
either TA-Lib (preferred) or pandas-ta as fallback.
"""

from typing import Any, Union

import numpy as np
import pandas as pd

from .base import IndicatorCategory, IndicatorMetadata, IndicatorSource
from .registry import IndicatorRegistry

# Check library availability
_TALIB_AVAILABLE = False
_PANDAS_TA_AVAILABLE = False

try:
    import talib
    _TALIB_AVAILABLE = True
except ImportError:
    pass

try:
    import pandas_ta as ta
    _PANDAS_TA_AVAILABLE = True
except ImportError:
    pass


def is_talib_available() -> bool:
    """Check if TA-Lib is available."""
    return _TALIB_AVAILABLE


def is_pandas_ta_available() -> bool:
    """Check if pandas-ta is available."""
    return _PANDAS_TA_AVAILABLE


class IndicatorWrapper:
    """Unified indicator calculation wrapper.
    
    This class provides a single interface for calculating any registered
    indicator. It automatically selects the best available library
    (TA-Lib preferred, pandas-ta as fallback).
    """
    
    def __init__(self, prefer_talib: bool = True):
        """Initialize the wrapper.
        
        Args:
            prefer_talib: Whether to prefer TA-Lib over pandas-ta when both are available
        """
        self.prefer_talib = prefer_talib
    
    def calculate(
        self,
        name: str,
        data: pd.DataFrame,
        output: str | None = None,
        source: str = "close",
        **params: Any,
    ) -> Union[pd.Series, pd.DataFrame]:
        """Calculate an indicator.
        
        Args:
            name: Indicator name (e.g., "sma", "rsi", "macd")
            data: OHLCV DataFrame
            output: For multi-output indicators, which output to return.
                   If None, returns all outputs as DataFrame
            source: For single-source indicators, which column to use
            **params: Indicator-specific parameters
            
        Returns:
            Series for single-output indicators, or DataFrame for multi-output
            
        Raises:
            ValueError: If indicator is unknown or params are invalid
        """
        name = name.lower()
        
        # Check if indicator exists
        metadata = IndicatorRegistry.get(name)
        if metadata is None:
            raise ValueError(f"Unknown indicator: {name}")
        
        # Check for custom indicator
        if IndicatorRegistry.is_custom(name):
            custom = IndicatorRegistry.get_custom(name)
            if custom is not None:
                # Merge default params
                final_params = custom.get_default_params()
                final_params.update(params)
                # Validate
                is_valid, error = custom.validate_params(final_params)
                if not is_valid:
                    raise ValueError(error)
                return custom.calculate(data, **final_params)
        
        # Get calculator function
        calculator = IndicatorRegistry.get_calculator(name)
        if calculator is None:
            raise ValueError(f"No calculator registered for indicator: {name}")
        
        # Determine which library to use
        use_talib = self._should_use_talib(metadata)
        
        # Calculate
        result = calculator(
            data=data,
            use_talib=use_talib,
            source=source,
            **params
        )
        
        # Handle output selection
        if output is not None and isinstance(result, pd.DataFrame):
            if output not in result.columns:
                raise ValueError(
                    f"Output '{output}' not found. Available: {list(result.columns)}"
                )
            return result[output]
        
        return result
    
    def _should_use_talib(self, metadata: IndicatorMetadata) -> bool:
        """Determine whether to use TA-Lib for this indicator."""
        has_talib = metadata.talib_func is not None and _TALIB_AVAILABLE
        has_pandas_ta = metadata.pandas_ta_func is not None and _PANDAS_TA_AVAILABLE
        
        if self.prefer_talib:
            return has_talib or (not has_pandas_ta)
        else:
            return has_talib and (not has_pandas_ta)
    
    def get_source(self, name: str) -> IndicatorSource:
        """Get which library will be used for an indicator."""
        metadata = IndicatorRegistry.get(name)
        if metadata is None:
            raise ValueError(f"Unknown indicator: {name}")
        
        if IndicatorRegistry.is_custom(name):
            return IndicatorSource.CUSTOM
        
        if self._should_use_talib(metadata):
            return IndicatorSource.TALIB
        return IndicatorSource.PANDAS_TA
    
    def validate(self, name: str, params: dict[str, Any]) -> tuple[bool, str | None]:
        """Validate indicator name and parameters.
        
        Args:
            name: Indicator name
            params: Parameters to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        metadata = IndicatorRegistry.get(name)
        if metadata is None:
            return False, f"Unknown indicator: {name}"
        
        # Check for custom indicator
        if IndicatorRegistry.is_custom(name):
            custom = IndicatorRegistry.get_custom(name)
            if custom is not None:
                return custom.validate_params(params)
        
        # Validate params against metadata
        for param_def in metadata.params:
            value = params.get(param_def.name, param_def.default)
            
            if value is None:
                return False, f"Missing required parameter: {param_def.name}"
            
            if param_def.min_value is not None and value < param_def.min_value:
                return False, f"Parameter {param_def.name} must be >= {param_def.min_value}"
            
            if param_def.max_value is not None and value > param_def.max_value:
                return False, f"Parameter {param_def.name} must be <= {param_def.max_value}"
        
        return True, None
    
    def get_metadata(self, name: str) -> IndicatorMetadata | None:
        """Get metadata for an indicator."""
        return IndicatorRegistry.get(name)
    
    def list_indicators(self, category: IndicatorCategory | None = None) -> list[str]:
        """List available indicators, optionally filtered by category."""
        if category is None:
            return IndicatorRegistry.list_all()
        return IndicatorRegistry.list_by_category(category)
    
    def get_categories(self) -> dict[str, list[str]]:
        """Get all categories with their indicators."""
        return IndicatorRegistry.get_categories()
    
    def print_available(self) -> str:
        """Print all available indicators."""
        return IndicatorRegistry.print_available()
    
    def get_signature(self, name: str) -> dict[str, Any] | None:
        """Get signature for an indicator."""
        metadata = IndicatorRegistry.get(name)
        if metadata is None:
            return None
        return metadata.get_signature()


# Global default wrapper instance
_default_wrapper: IndicatorWrapper | None = None


def get_wrapper() -> IndicatorWrapper:
    """Get the default indicator wrapper."""
    global _default_wrapper
    if _default_wrapper is None:
        _default_wrapper = IndicatorWrapper()
    return _default_wrapper


def calculate(
    name: str,
    data: pd.DataFrame,
    output: str | None = None,
    source: str = "close",
    **params: Any,
) -> Union[pd.Series, pd.DataFrame]:
    """Calculate an indicator using the default wrapper.
    
    This is a convenience function for quick calculations.
    
    Args:
        name: Indicator name
        data: OHLCV DataFrame
        output: For multi-output indicators, which output to return
        source: For single-source indicators, which column to use
        **params: Indicator-specific parameters
        
    Returns:
        Series or DataFrame with indicator values
    """
    return get_wrapper().calculate(name, data, output=output, source=source, **params)
