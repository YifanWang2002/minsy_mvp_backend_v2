"""Optional CCXT REST provider for historical OHLCV fetches."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from packages.shared_settings.schema.settings import settings
from packages.infra.providers.trading.adapters.base import OhlcvBar


class CcxtRestProvider:
    """Thin wrapper around ccxt async client (optional dependency)."""

    def __init__(
        self,
        *,
        exchange_id: str | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        try:
            import ccxt.async_support as ccxt  # type: ignore[import-not-found]
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("ccxt package is not installed") from exc

        resolved_exchange_id = (
            str(exchange_id or settings.ccxt_market_data_exchange_id).strip().lower() or "binance"
        )
        if not hasattr(ccxt, resolved_exchange_id):
            raise RuntimeError(f"Unsupported ccxt exchange: {resolved_exchange_id}")

        exchange_cls = getattr(ccxt, resolved_exchange_id)
        timeout_ms = int(max((timeout_seconds or settings.ccxt_market_data_timeout_seconds), 1.0) * 1000)
        self._exchange = exchange_cls(
            {
                "enableRateLimit": True,
                "timeout": timeout_ms,
            }
        )

    async def fetch_ohlcv(
        self,
        *,
        symbol: str,
        timeframe: str,
        since: datetime | None,
        limit: int,
        market: str,
    ) -> list[OhlcvBar]:
        ccxt_symbol = _to_ccxt_symbol(symbol=symbol, market=market)
        since_ms = None
        if since is not None:
            since_ms = int(since.astimezone(UTC).timestamp() * 1000)

        rows = await self._exchange.fetch_ohlcv(
            symbol=ccxt_symbol,
            timeframe=_to_ccxt_timeframe(timeframe),
            since=since_ms,
            limit=max(1, int(limit)),
        )
        if not isinstance(rows, list):
            return []

        bars: list[OhlcvBar] = []
        for item in rows:
            if not isinstance(item, list) or len(item) < 6:
                continue
            try:
                timestamp = datetime.fromtimestamp(float(item[0]) / 1000.0, tz=UTC)
                bars.append(
                    OhlcvBar(
                        timestamp=timestamp,
                        open=Decimal(str(item[1])),
                        high=Decimal(str(item[2])),
                        low=Decimal(str(item[3])),
                        close=Decimal(str(item[4])),
                        volume=Decimal(str(item[5])),
                    )
                )
            except Exception:  # noqa: BLE001
                continue
        return bars

    async def aclose(self) -> None:
        await self._exchange.close()


def _to_ccxt_timeframe(value: str) -> str:
    normalized = str(value).strip().lower()
    mapping = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1h",
        "2h": "2h",
        "4h": "4h",
        "1d": "1d",
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported timeframe for ccxt: {value}")
    return mapping[normalized]


def _to_ccxt_symbol(*, symbol: str, market: str) -> str:
    normalized = symbol.strip().upper().replace("-", "").replace("/", "")
    if "/" in symbol:
        parts = [item.strip().upper() for item in symbol.split("/", 1)]
        if len(parts) == 2 and parts[0] and parts[1]:
            return f"{parts[0]}/{parts[1]}"

    if market.strip().lower() == "crypto":
        quote_candidates = ("USDT", "USDC", "USD", "BTC", "ETH")
        for quote in quote_candidates:
            if normalized.endswith(quote) and len(normalized) > len(quote):
                base = normalized[: -len(quote)]
                quote_symbol = "USDT" if quote == "USD" else quote
                return f"{base}/{quote_symbol}"
        return f"{normalized}/USDT"

    return normalized
