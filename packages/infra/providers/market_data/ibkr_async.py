"""IBKR market-data provider backed by ib_async."""

from __future__ import annotations

import asyncio
from calendar import monthrange
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from math import ceil
from typing import Any

from packages.infra.providers.trading.adapters.base import OhlcvBar
from packages.shared_settings.schema.settings import settings

_FUTURES_EXCHANGE_ALIASES: dict[str, str] = {
    "GLOBEX": "CME",
    "ECBOT": "CBOT",
}

_FUTURES_DEFAULT_EXCHANGE_BY_SYMBOL: dict[str, str] = {
    "ES": "CME",
    "NQ": "CME",
    "YM": "CBOT",
    "RTY": "CME",
    "CL": "NYMEX",
    "NG": "NYMEX",
    "RB": "NYMEX",
    "GC": "COMEX",
    "SI": "COMEX",
    "HG": "COMEX",
    "ZN": "CBOT",
    "ZB": "CBOT",
    "ZF": "CBOT",
    "ZT": "CBOT",
}


def _normalize_symbol(symbol: str) -> str:
    normalized = str(symbol).strip().upper().replace("/", "").replace("-", "")
    if not normalized:
        raise ValueError("symbol cannot be empty")
    return normalized


def _duration_str_for_window(start: datetime, end: datetime) -> str:
    seconds = max(60, int((end - start).total_seconds()) + 120)
    if seconds <= 86_400:
        return f"{seconds} S"
    days = max(1, int(ceil(seconds / 86_400.0)))
    return f"{days} D"


def _parse_bar_timestamp(raw: Any) -> datetime | None:
    if isinstance(raw, datetime):
        if raw.tzinfo is None:
            return raw.replace(tzinfo=UTC)
        return raw.astimezone(UTC)
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        normalized = text.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    return None


def _normalize_futures_exchange(exchange: str) -> str:
    normalized = str(exchange).strip().upper()
    if not normalized:
        return "CME"
    return _FUTURES_EXCHANGE_ALIASES.get(normalized, normalized)


def _parse_contract_expiry_date(raw: Any) -> date | None:
    text = str(raw or "").strip()
    if not text:
        return None
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) >= 8:
        try:
            year = int(digits[0:4])
            month = int(digits[4:6])
            day = int(digits[6:8])
            return date(year, month, day)
        except ValueError:
            return None
    if len(digits) >= 6:
        try:
            year = int(digits[0:4])
            month = int(digits[4:6])
            last_day = monthrange(year, month)[1]
            return date(year, month, last_day)
        except ValueError:
            return None
    return None


class IbkrAsyncMarketDataProvider:
    """Fetch historical bars from local IBKR gateway via ib_async."""

    def __init__(
        self,
        *,
        host: str | None = None,
        port: int | None = None,
        client_id: int | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        if settings.market_data_incremental_execution_mode != "local_collector":
            raise RuntimeError(
                "IBKR incremental provider can only run in local_collector execution mode."
            )
        self._host = str(host or settings.ibkr_gateway_host).strip()
        self._port = int(port or settings.ibkr_gateway_port)
        self._client_id = int(client_id or settings.ibkr_gateway_client_id)
        self._timeout_seconds = float(timeout_seconds or settings.ibkr_gateway_timeout_seconds)
        self._ib = None
        self._ib_lib: Any = None
        self._contract_lib: Any = None
        self._futures_contract_candidates_cache: dict[tuple[str, str], list[tuple[date, Any]]] = (
            {}
        )

    async def _ensure_connected(self) -> None:
        if self._ib is not None and bool(self._ib.isConnected()):
            return
        try:
            import ib_async as ib_lib  # type: ignore[import-not-found]
            from ib_async import (
                contract as contract_lib,  # type: ignore[import-not-found]
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("ib_async package is not installed") from exc
        self._ib_lib = ib_lib
        self._contract_lib = contract_lib
        self._ib = ib_lib.IB()
        await self._ib.connectAsync(
            self._host,
            self._port,
            clientId=self._client_id,
            timeout=self._timeout_seconds,
        )

    async def aclose(self) -> None:
        if self._ib is None:
            return
        try:
            if bool(self._ib.isConnected()):
                self._ib.disconnect()
        finally:
            self._ib = None

    def _build_forex_contract(self, *, symbol: str):
        assert self._contract_lib is not None
        symbol_key = _normalize_symbol(symbol)
        return self._contract_lib.Forex(symbol_key)

    def _futures_exchange_for_symbol(self, *, symbol_key: str) -> str:
        exchange_map = settings.ibkr_futures_exchange_map_json
        exchange_raw = ""
        if isinstance(exchange_map, dict):
            exchange_raw = str(exchange_map.get(symbol_key, "")).strip()
        if not exchange_raw:
            exchange_raw = _FUTURES_DEFAULT_EXCHANGE_BY_SYMBOL.get(symbol_key, "CME")
        return _normalize_futures_exchange(exchange_raw)

    async def _futures_contract_candidates(
        self,
        *,
        symbol_key: str,
        exchange: str,
    ) -> list[tuple[date, Any]]:
        assert self._ib is not None
        assert self._contract_lib is not None

        cache_key = (symbol_key, exchange)
        cached = self._futures_contract_candidates_cache.get(cache_key)
        if cached is not None:
            return cached

        base_contract = self._contract_lib.Future(
            symbol=symbol_key,
            exchange=exchange,
            currency="USD",
            includeExpired=True,
        )
        details = await self._ib.reqContractDetailsAsync(base_contract)
        candidates: list[tuple[date, Any]] = []
        for detail in details or []:
            contract = getattr(detail, "contract", None)
            if contract is None:
                continue
            expiry = _parse_contract_expiry_date(
                getattr(contract, "lastTradeDateOrContractMonth", None)
            )
            if expiry is None:
                continue
            candidates.append((expiry, contract))
        if not candidates:
            raise ValueError(
                f"Unable to resolve IBKR futures contract for {symbol_key} on exchange {exchange}."
            )

        deduped: dict[str, tuple[date, Any]] = {}
        for expiry, contract in candidates:
            con_id = getattr(contract, "conId", None)
            local_symbol = str(getattr(contract, "localSymbol", "")).strip()
            key = str(con_id) if con_id else f"{expiry.isoformat()}:{local_symbol}"
            deduped[key] = (expiry, contract)

        ordered = sorted(deduped.values(), key=lambda item: item[0])
        self._futures_contract_candidates_cache[cache_key] = ordered
        return ordered

    async def _resolve_futures_contract(
        self,
        *,
        symbol: str,
        end: datetime,
    ):
        symbol_key = _normalize_symbol(symbol)
        exchange = self._futures_exchange_for_symbol(symbol_key=symbol_key)
        candidates = await self._futures_contract_candidates(
            symbol_key=symbol_key,
            exchange=exchange,
        )
        target_day = end.astimezone(UTC).date()
        for expiry, contract in candidates:
            if expiry >= target_day:
                return contract
        return candidates[-1][1]

    async def _fetch_window(
        self,
        *,
        symbol: str,
        market: str,
        start: datetime,
        end: datetime,
        limit: int,
    ) -> list[OhlcvBar]:
        assert self._ib is not None
        market_key = str(market).strip().lower()
        if market_key == "forex":
            contract = self._build_forex_contract(symbol=symbol)
        elif market_key == "futures":
            contract = await self._resolve_futures_contract(symbol=symbol, end=end)
        else:
            raise ValueError(f"Unsupported IBKR market: {market}")
        await self._ib.qualifyContractsAsync(contract)
        duration = _duration_str_for_window(start, end)
        what_to_show = (
            settings.ibkr_forex_what_to_show
            if market_key == "forex"
            else settings.ibkr_futures_what_to_show
        )
        window_timeout_seconds = max(float(self._timeout_seconds) * 6.0, 30.0)
        bars = None
        for attempt in range(1, 4):
            try:
                bars = await asyncio.wait_for(
                    self._ib.reqHistoricalDataAsync(
                        contract=contract,
                        endDateTime=end.astimezone(UTC),
                        durationStr=duration,
                        barSizeSetting="1 min",
                        whatToShow=what_to_show,
                        useRTH=False,
                        formatDate=2,
                        keepUpToDate=False,
                    ),
                    timeout=window_timeout_seconds,
                )
                break
            except TimeoutError:
                if attempt >= 3:
                    raise TimeoutError(
                        "IBKR historical request timed out "
                        f"market={market_key} symbol={symbol} "
                        f"window={start.isoformat()}..{end.isoformat()} "
                        f"timeout={window_timeout_seconds}s"
                    )
                await asyncio.sleep(0.25 * attempt)
        output: list[OhlcvBar] = []
        for item in bars or []:
            timestamp = _parse_bar_timestamp(getattr(item, "date", None))
            if timestamp is None:
                continue
            if timestamp < start or timestamp > end:
                continue
            output.append(
                OhlcvBar(
                    timestamp=timestamp.replace(second=0, microsecond=0),
                    open=Decimal(str(getattr(item, "open", 0))),
                    high=Decimal(str(getattr(item, "high", 0))),
                    low=Decimal(str(getattr(item, "low", 0))),
                    close=Decimal(str(getattr(item, "close", 0))),
                    volume=Decimal(str(getattr(item, "volume", 0))),
                )
            )
        output = sorted(output, key=lambda row: row.timestamp)
        if len(output) <= limit:
            return output
        return output[-limit:]

    async def fetch_ohlcv(
        self,
        *,
        symbol: str,
        market: str,
        timeframe: str,
        since: datetime | None,
        until: datetime | None = None,
        limit: int = 5000,
    ) -> list[OhlcvBar]:
        timeframe_key = str(timeframe).strip().lower()
        if timeframe_key not in {"1m", "1min"}:
            raise ValueError("IBKR incremental provider only supports 1m timeframe.")
        if since is None:
            raise ValueError("since is required for IBKR incremental fetch.")

        await self._ensure_connected()
        assert self._ib is not None

        start = since.astimezone(UTC).replace(second=0, microsecond=0)
        end = (until or datetime.now(UTC)).astimezone(UTC).replace(second=0, microsecond=0)
        if end < start:
            return []

        bars_per_window = max(1, min(int(limit), 10_080))
        max_span = timedelta(minutes=bars_per_window - 1)
        cursor = start
        merged: dict[datetime, OhlcvBar] = {}
        while cursor <= end:
            window_end = min(end, cursor + max_span)
            window_rows = await self._fetch_window(
                symbol=symbol,
                market=market,
                start=cursor,
                end=window_end,
                limit=bars_per_window,
            )
            for row in window_rows:
                merged[row.timestamp] = row
            cursor = window_end + timedelta(minutes=1)
            # Gentle pacing to avoid local gateway bursts.
            await asyncio.sleep(0)
        return [merged[key] for key in sorted(merged)]
