"""Strategy DSL validation/parsing engine."""

from pathlib import Path

from src.engine.strategy.errors import (
    StrategyDslError,
    StrategyDslValidationError,
    StrategyDslValidationException,
    StrategyDslValidationResult,
)
from src.engine.strategy.models import (
    FactorDefinition,
    ParsedStrategyDsl,
    StrategyInfo,
    StrategyUniverse,
)
from src.engine.strategy.pipeline import (
    load_strategy_payload,
    parse_strategy_payload,
    validate_strategy_payload,
)
from src.engine.strategy.storage import (
    StrategyMetadataReceipt,
    StrategyPersistenceResult,
    StrategyStorageNotFoundError,
    get_session_user_id,
    get_strategy_or_raise,
    upsert_strategy_dsl,
    validate_stored_strategy,
)

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
SCHEMA_PATH = ASSETS_DIR / "strategy_dsl_schema.json"
EXAMPLE_PATH = ASSETS_DIR / "example_strategy.json"

__all__ = [
    "ASSETS_DIR",
    "EXAMPLE_PATH",
    "SCHEMA_PATH",
    "FactorDefinition",
    "ParsedStrategyDsl",
    "StrategyDslError",
    "StrategyDslValidationError",
    "StrategyDslValidationException",
    "StrategyDslValidationResult",
    "StrategyMetadataReceipt",
    "StrategyPersistenceResult",
    "StrategyStorageNotFoundError",
    "StrategyInfo",
    "StrategyUniverse",
    "get_session_user_id",
    "get_strategy_or_raise",
    "load_strategy_payload",
    "parse_strategy_payload",
    "upsert_strategy_dsl",
    "validate_strategy_payload",
    "validate_stored_strategy",
]
