"""Market-to-provider routing for incremental sync."""

from __future__ import annotations

_MARKET_ALIASES: dict[str, str] = {
    "stock": "us_stocks",
    "stocks": "us_stocks",
    "us_stock": "us_stocks",
    "us_stocks": "us_stocks",
    "us_equity": "us_stocks",
    "us_equities": "us_stocks",
    "equity": "us_stocks",
    "equities": "us_stocks",
    "crypto": "crypto",
    "cryptos": "crypto",
    "forex": "forex",
    "fx": "forex",
    "futures": "futures",
    "future": "futures",
}

_PROVIDER_BY_MARKET: dict[str, str] = {
    "crypto": "alpaca",
    "us_stocks": "alpaca",
    "forex": "ibkr",
    "futures": "ibkr",
}


def normalize_incremental_market(value: str) -> str:
    normalized = str(value).strip().lower()
    if not normalized:
        raise ValueError("market cannot be empty")
    mapped = _MARKET_ALIASES.get(normalized, normalized)
    if mapped not in _PROVIDER_BY_MARKET:
        raise ValueError(f"Unsupported incremental market: {value}")
    return mapped


def resolve_provider_for_market(value: str) -> str:
    market = normalize_incremental_market(value)
    return _PROVIDER_BY_MARKET[market]
