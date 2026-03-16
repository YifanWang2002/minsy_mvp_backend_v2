"""Market-data domain package exports."""

from packages.domain.market_data.aggregator import AggregatedBar, BarAggregator
from packages.domain.market_data.catalog_service import (
    CatalogEntry,
    get_symbol_coverage,
    mark_accessed,
    scan_and_sync_catalog,
    upsert_catalog_entry,
    upsert_catalog_entry_from_parquet,
)
from packages.domain.market_data.factor_cache import (
    FactorCache,
    FactorCacheStats,
    factor_signature,
)
from packages.domain.market_data.ring_buffer import OhlcvRing
from packages.domain.market_data.runtime import (
    MarketDataRuntime,
    RuntimeBar,
    market_data_runtime,
)
from packages.domain.market_data.subscription_registry import (
    SubscriptionDelta,
    SubscriptionRegistry,
)
from packages.domain.market_data.sync_service import (
    MarketDataNoMissingDataError,
    MarketDataProviderUnavailableError,
    MarketDataSyncInputError,
    MarketDataSyncJobNotFoundError,
    MarketDataSyncJobReceipt,
    MarketDataSyncJobView,
    create_market_data_sync_job,
    execute_market_data_sync_job,
    execute_market_data_sync_job_with_fresh_session,
    get_market_data_sync_job_view,
    schedule_market_data_sync_job,
)
from packages.domain.market_data.regime import (
    FamilyScores,
    RegimeSnapshot,
    SUPPORTED_REGIME_TIMEFRAMES,
    StrategyFamilyId,
    TimeframePlan,
    build_regime_feature_snapshot,
    map_pre_strategy_timeframes,
    score_strategy_families,
)

__all__ = [
    "AggregatedBar",
    "BarAggregator",
    "CatalogEntry",
    "FactorCache",
    "FactorCacheStats",
    "MarketDataNoMissingDataError",
    "MarketDataProviderUnavailableError",
    "MarketDataRuntime",
    "MarketDataSyncInputError",
    "MarketDataSyncJobNotFoundError",
    "MarketDataSyncJobReceipt",
    "MarketDataSyncJobView",
    "OhlcvRing",
    "RuntimeBar",
    "SubscriptionDelta",
    "SubscriptionRegistry",
    "SUPPORTED_REGIME_TIMEFRAMES",
    "StrategyFamilyId",
    "TimeframePlan",
    "FamilyScores",
    "RegimeSnapshot",
    "build_regime_feature_snapshot",
    "create_market_data_sync_job",
    "execute_market_data_sync_job",
    "execute_market_data_sync_job_with_fresh_session",
    "factor_signature",
    "get_market_data_sync_job_view",
    "get_symbol_coverage",
    "mark_accessed",
    "market_data_runtime",
    "map_pre_strategy_timeframes",
    "scan_and_sync_catalog",
    "schedule_market_data_sync_job",
    "score_strategy_families",
    "upsert_catalog_entry",
    "upsert_catalog_entry_from_parquet",
]
