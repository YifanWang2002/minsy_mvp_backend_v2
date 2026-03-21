"""CCXT exchange metadata catalog used by broker settings and validation."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any


@dataclass(frozen=True, slots=True)
class CcxtExchangeStaticConfig:
    """Product-level static metadata overrides for one CCXT exchange."""

    exchange_id: str
    name: str
    description: str
    website_url: str
    website_title: str
    paper_trading_status: str = "not_supported"
    supports_demo: bool = False


_STATIC_EXCHANGES: tuple[CcxtExchangeStaticConfig, ...] = (
    CcxtExchangeStaticConfig(
        exchange_id="binance",
        name="Binance",
        description="Global crypto exchange with spot and derivatives.",
        website_url="https://www.binance.com",
        website_title="Binance - Cryptocurrency Exchange for Bitcoin, Ethereum & Altcoins",
        paper_trading_status="supported",
        supports_demo=True,
    ),
    CcxtExchangeStaticConfig(
        exchange_id="okx",
        name="OKX",
        description="Crypto exchange with robust demo-trading support.",
        website_url="https://www.okx.com",
        website_title="OKX: Buy Bitcoin & Crypto | Crypto Exchange, App & Wallet",
        paper_trading_status="supported",
        supports_demo=True,
    ),
    CcxtExchangeStaticConfig(
        exchange_id="bybit",
        name="Bybit",
        description="Spot and derivatives exchange with testnet/demo environments.",
        website_url="https://www.bybit.com",
        website_title="Bybit: Buy & Trade Crypto",
        paper_trading_status="supported",
        supports_demo=True,
    ),
    CcxtExchangeStaticConfig(
        exchange_id="bitget",
        name="Bitget",
        description="Crypto exchange with demo trading for derivatives.",
        website_url="https://www.bitget.com",
        website_title="Bitget Exchange: Crypto Trading Platform | Buy and Sell Bitcoin, Ethereum",
        paper_trading_status="supported",
        supports_demo=True,
    ),
    CcxtExchangeStaticConfig(
        exchange_id="coinbase",
        name="Coinbase",
        description="Coinbase Advanced Trade API.",
        website_url="https://www.coinbase.com",
        website_title="Coinbase - Buy and Sell Bitcoin, Ethereum, and more with trust",
        paper_trading_status="not_supported",
    ),
    CcxtExchangeStaticConfig(
        exchange_id="kucoin",
        name="KuCoin",
        description="Crypto exchange with futures-focused paper-trading support.",
        website_url="https://www.kucoin.com",
        website_title="Crypto Exchange | Bitcoin Exchange | Bitcoin Trading | KuCoin",
        paper_trading_status="in_progress",
        supports_demo=True,
    ),
    CcxtExchangeStaticConfig(
        exchange_id="kraken",
        name="Kraken",
        description="Long-standing exchange; public sandbox support is limited by market.",
        website_url="https://www.kraken.com",
        website_title="Kraken: Buy and sell crypto securely",
        paper_trading_status="in_progress",
        supports_demo=True,
    ),
    CcxtExchangeStaticConfig(
        exchange_id="krakenfutures",
        name="Kraken Futures",
        description="Kraken Futures with full demo/testnet sandbox support.",
        website_url="https://futures.kraken.com",
        website_title="Kraken Futures: Crypto Derivatives Trading",
        paper_trading_status="supported",
        supports_demo=True,
    ),
    CcxtExchangeStaticConfig(
        exchange_id="gemini",
        name="Gemini",
        description="US-focused exchange with sandbox environment.",
        website_url="https://www.gemini.com",
        website_title="Buy, Sell & Trade Bitcoin, Solana, & Other Cryptos with Gemini's Best-in-class Platform | Gemini",
        paper_trading_status="in_progress",
        supports_demo=True,
    ),
    CcxtExchangeStaticConfig(
        exchange_id="bitmex",
        name="BitMEX",
        description="Derivatives-focused venue with long-running testnet.",
        website_url="https://www.bitmex.com",
        website_title="BitMEX | Crypto Derivatives Exchange",
        paper_trading_status="in_progress",
        supports_demo=True,
    ),
    CcxtExchangeStaticConfig(
        exchange_id="gateio",
        name="Gate.io",
        description="Global exchange with published testnet environment.",
        website_url="https://www.gate.io",
        website_title="Buy/Sell Bitcoin, Ethereum and 4,400+ Altcoins | Cryptocurrency Exchange | Gate.com",
        paper_trading_status="in_progress",
        supports_demo=True,
    ),
    CcxtExchangeStaticConfig(
        exchange_id="mexc",
        name="MEXC",
        description="Global exchange with live API integration.",
        website_url="https://www.mexc.com",
        website_title="MEXC Exchange: Your Easiest Way to Crypto - Trade Bitcoin, Ethereum & Most Trending Tokens",
        paper_trading_status="not_supported",
    ),
    CcxtExchangeStaticConfig(
        exchange_id="cryptocom",
        name="Crypto.com Exchange",
        description="Crypto.com Exchange API integration.",
        website_url="https://crypto.com/exchange",
        website_title="Crypto.com Exchange",
        paper_trading_status="not_supported",
    ),
)


_FIELD_MAPPING: dict[str, str] = {
    "apiKey": "api_key",
    "secret": "api_secret",
    "password": "password",
    "uid": "uid",
}


def _normalize_fields(raw_fields: list[str]) -> list[str]:
    seen: set[str] = set()
    normalized: list[str] = []
    for field in raw_fields:
        key = str(field).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        normalized.append(key)
    return normalized


def _extract_ccxt_required_and_optional_fields(
    required_credentials: dict[str, Any],
) -> tuple[list[str], list[str]]:
    required_fields: list[str] = []
    optional_fields: list[str] = []
    for ccxt_field, ui_field in _FIELD_MAPPING.items():
        value = required_credentials.get(ccxt_field)
        if value is True:
            required_fields.append(ui_field)
        elif value is False:
            optional_fields.append(ui_field)

    # Trading account integrations always need key/secret to validate.
    if "api_key" not in required_fields:
        required_fields.insert(0, "api_key")
    if "api_secret" not in required_fields:
        required_fields.append("api_secret")

    # Keep optional extensions discoverable if not explicitly required.
    for fallback_field in ("password", "uid"):
        if (
            fallback_field not in required_fields
            and fallback_field not in optional_fields
        ):
            optional_fields.append(fallback_field)

    return _normalize_fields(required_fields), _normalize_fields(optional_fields)


def _resolve_paper_support_message(status: str) -> str:
    mapping = {
        "supported": "Demo API trading supported.",
        "in_progress": "Supported but development in progress.",
        "not_supported": "Not supported, consider using our built-in sandbox or switch to another exchange.",
    }
    return mapping.get(status, mapping["not_supported"])


def _build_exchange_metadata(config: CcxtExchangeStaticConfig) -> dict[str, Any]:
    exchange_id = config.exchange_id.strip().lower()
    required_fields = ["api_key", "api_secret"]
    optional_fields = ["password", "uid"]
    supports_testnet = False
    supports_sandbox = False

    try:
        import ccxt  # type: ignore[import-not-found]

        if hasattr(ccxt, exchange_id):
            exchange_cls = getattr(ccxt, exchange_id)
            exchange = exchange_cls()
            required_credentials = getattr(exchange, "requiredCredentials", {})
            required_credentials = (
                required_credentials if isinstance(required_credentials, dict) else {}
            )
            required_fields, optional_fields = (
                _extract_ccxt_required_and_optional_fields(required_credentials)
            )

            urls = getattr(exchange, "urls", {})
            urls = urls if isinstance(urls, dict) else {}
            supports_testnet = bool(urls.get("test"))

            has_map = getattr(exchange, "has", {})
            has_map = has_map if isinstance(has_map, dict) else {}
            supports_sandbox = bool(has_map.get("sandbox")) or supports_testnet
    except Exception:
        # Keep static fallback metadata when ccxt runtime introspection is unavailable.
        pass

    paper_trading_status = (
        str(config.paper_trading_status).strip().lower() or "not_supported"
    )
    supports_paper = paper_trading_status == "supported"
    return {
        "exchange_id": exchange_id,
        "name": config.name,
        "required_fields": required_fields,
        "optional_fields": optional_fields,
        "supports_sandbox": supports_sandbox,
        "supports_testnet": supports_testnet,
        "supports_demo": bool(config.supports_demo),
        "supports_paper": supports_paper,
        # Product policy: keep live capability badge disabled for now.
        "supports_live": False,
        "paper_trading_status": paper_trading_status,
        "paper_trading_message": _resolve_paper_support_message(paper_trading_status),
        "live_trading_status": "disabled",
        "live_trading_message": "Live trading is currently disabled on this platform.",
        "description": config.description,
        "website_title": config.website_title,
        "website_url": config.website_url,
    }


@lru_cache(maxsize=1)
def list_supported_ccxt_exchanges() -> tuple[dict[str, Any], ...]:
    """Return product-supported CCXT exchanges for settings UI and validation."""

    rows = [_build_exchange_metadata(config) for config in _STATIC_EXCHANGES]
    return tuple(rows)


@lru_cache(maxsize=64)
def resolve_ccxt_exchange_metadata(exchange_id: str) -> dict[str, Any]:
    """Resolve one exchange metadata entry from allowlist or dynamic ccxt fallback."""

    normalized_exchange_id = str(exchange_id).strip().lower()
    for item in list_supported_ccxt_exchanges():
        if item["exchange_id"] == normalized_exchange_id:
            return item

    # Dynamic fallback for non-allowlisted exchanges so validation remains backward-compatible.
    dynamic_config = CcxtExchangeStaticConfig(
        exchange_id=normalized_exchange_id,
        name=normalized_exchange_id.upper(),
        description="CCXT exchange (dynamic fallback).",
        website_url="",
        website_title=normalized_exchange_id.upper(),
        paper_trading_status="not_supported",
    )
    return _build_exchange_metadata(dynamic_config)
