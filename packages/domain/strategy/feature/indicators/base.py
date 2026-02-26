"""Base classes for the indicator system."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Union

import pandas as pd


class IndicatorCategory(str, Enum):
    """Indicator categories."""
    
    OVERLAP = "overlap"  # Moving averages & overlay indicators
    MOMENTUM = "momentum"  # Momentum & trend indicators
    VOLATILITY = "volatility"  # Volatility indicators
    VOLUME = "volume"  # Volume indicators
    CANDLE = "candle"  # Candlestick patterns
    UTILS = "utils"  # Statistics, math, cycle, etc.


class IndicatorSource(str, Enum):
    """Source library for indicator calculation."""
    
    TALIB = "talib"
    PANDAS_TA = "pandas_ta"
    CUSTOM = "custom"


@dataclass
class IndicatorParam:
    """Indicator parameter definition."""
    
    name: str
    type: str  # "int", "float", "str", "bool"
    default: Any
    min_value: float | None = None
    max_value: float | None = None
    description: str = ""
    choices: list[str] | None = None  # For enum-like params


@dataclass
class IndicatorOutput:
    """Indicator output definition."""
    
    name: str
    description: str = ""


@dataclass
class IndicatorMetadata:
    """Complete indicator metadata."""
    
    name: str  # Function name (e.g., "sma", "ema")
    full_name: str  # Full name (e.g., "Simple Moving Average")
    category: IndicatorCategory
    description: str = ""
    params: list[IndicatorParam] = field(default_factory=list)
    outputs: list[IndicatorOutput] = field(default_factory=list)
    talib_func: str | None = None  # TA-Lib function name (e.g., "SMA")
    pandas_ta_func: str | None = None  # pandas-ta function name (e.g., "sma")
    required_columns: list[str] = field(default_factory=lambda: ["close"])
    
    def get_signature(self) -> dict[str, Any]:
        """Get function signature for documentation."""
        return {
            "name": self.name,
            "full_name": self.full_name,
            "category": self.category.value,
            "description": self.description,
            "params": [
                {
                    "name": p.name,
                    "type": p.type,
                    "default": p.default,
                    "min": p.min_value,
                    "max": p.max_value,
                    "description": p.description,
                    "choices": p.choices,
                }
                for p in self.params
            ],
            "outputs": [
                {"name": o.name, "description": o.description}
                for o in self.outputs
            ],
            "required_columns": self.required_columns,
            "sources": {
                "talib": self.talib_func,
                "pandas_ta": self.pandas_ta_func,
            },
        }


class BaseIndicator(ABC):
    """Abstract base class for custom indicators."""
    
    metadata: IndicatorMetadata
    
    @abstractmethod
    def calculate(
        self, 
        data: pd.DataFrame, 
        **params: Any
    ) -> Union[pd.Series, pd.DataFrame]:
        """Calculate indicator values.
        
        Args:
            data: OHLCV DataFrame with columns: open, high, low, close, volume
            **params: Indicator parameters
            
        Returns:
            pandas Series or DataFrame with indicator values
        """
        pass
    
    def validate_params(self, params: dict[str, Any]) -> tuple[bool, str | None]:
        """Validate indicator parameters.
        
        Args:
            params: Parameters to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        for param_def in self.metadata.params:
            # Get value or default
            value = params.get(param_def.name, param_def.default)
            
            if value is None:
                return False, f"Missing required parameter: {param_def.name}"
            
            # Check min/max constraints
            if param_def.min_value is not None and value < param_def.min_value:
                return False, f"Parameter {param_def.name} must be >= {param_def.min_value}"
            
            if param_def.max_value is not None and value > param_def.max_value:
                return False, f"Parameter {param_def.name} must be <= {param_def.max_value}"
            
            # Check type
            if param_def.type == 'int' and not isinstance(value, int):
                if not (isinstance(value, float) and value.is_integer()):
                    return False, f"Parameter {param_def.name} must be an integer"
            elif param_def.type == 'float' and not isinstance(value, (int, float)):
                return False, f"Parameter {param_def.name} must be a number"
            elif param_def.type == 'str' and not isinstance(value, str):
                return False, f"Parameter {param_def.name} must be a string"
            elif param_def.type == 'bool' and not isinstance(value, bool):
                return False, f"Parameter {param_def.name} must be a boolean"
            
            # Check choices
            if param_def.choices is not None and value not in param_def.choices:
                return False, f"Parameter {param_def.name} must be one of {param_def.choices}"
        
        return True, None
    
    def get_default_params(self) -> dict[str, Any]:
        """Get default parameter values."""
        return {p.name: p.default for p in self.metadata.params}
    
    def validate_data(self, data: pd.DataFrame) -> tuple[bool, str | None]:
        """Validate input data has required columns."""
        missing = [col for col in self.metadata.required_columns if col not in data.columns]
        if missing:
            return False, f"Missing required columns: {missing}"
        return True, None
