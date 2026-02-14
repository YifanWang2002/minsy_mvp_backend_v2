"""Strategy DSL validation/parsing engine."""

from pathlib import Path

from src.engine.strategy.errors import (
    StrategyDslError,
    StrategyDslValidationError,
    StrategyDslValidationException,
    StrategyDslValidationResult,
)
from src.engine.strategy.draft_store import (
    StrategyDraftRecord,
    create_strategy_draft,
    get_strategy_draft,
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
    StrategyPatchApplyError,
    StrategyPersistenceResult,
    StrategyRevisionNotFoundError,
    StrategyRevisionReceipt,
    StrategyStorageNotFoundError,
    StrategyVersionConflictError,
    StrategyVersionDiff,
    StrategyVersionPayload,
    apply_strategy_json_patch,
    diff_strategy_versions,
    get_session_user_id,
    get_strategy_or_raise,
    get_strategy_version_payload,
    list_strategy_versions,
    patch_strategy_dsl,
    rollback_strategy_dsl,
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
    "StrategyDraftRecord",
    "StrategyMetadataReceipt",
    "StrategyPatchApplyError",
    "StrategyPersistenceResult",
    "StrategyRevisionNotFoundError",
    "StrategyRevisionReceipt",
    "StrategyStorageNotFoundError",
    "StrategyInfo",
    "StrategyUniverse",
    "StrategyVersionDiff",
    "StrategyVersionPayload",
    "StrategyVersionConflictError",
    "apply_strategy_json_patch",
    "diff_strategy_versions",
    "create_strategy_draft",
    "get_session_user_id",
    "get_strategy_draft",
    "get_strategy_or_raise",
    "get_strategy_version_payload",
    "list_strategy_versions",
    "load_strategy_payload",
    "patch_strategy_dsl",
    "parse_strategy_payload",
    "rollback_strategy_dsl",
    "upsert_strategy_dsl",
    "validate_strategy_payload",
    "validate_stored_strategy",
]
