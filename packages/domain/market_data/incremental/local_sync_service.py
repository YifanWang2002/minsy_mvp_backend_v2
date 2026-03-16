"""Local-only incremental collector service."""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from packages.domain.market_data.incremental.hmac_auth import (
    build_signature_payload,
    sign_payload,
)
from packages.domain.market_data.incremental.provider_router import (
    normalize_incremental_market,
    resolve_provider_for_market,
)
from packages.domain.market_data.incremental.session_gate import (
    market_is_open_for_incremental,
)
from packages.infra.observability.logger import logger
from packages.infra.providers.market_data.alpaca_client import AlpacaMarketDataClient
from packages.infra.providers.market_data.ibkr_async import IbkrAsyncMarketDataProvider
from packages.infra.providers.trading.adapters.base import OhlcvBar
from packages.infra.storage.gcs_client import GcsClient
from packages.shared_settings.schema.settings import settings


@dataclass(frozen=True, slots=True)
class LocalIncrementalSyncResult:
    status: str
    run_id: str
    symbols_seen: int
    symbols_synced: int
    files_uploaded: int
    rows_uploaded: int
    skipped_closed: int
    skipped_uptodate: int
    import_job_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "run_id": self.run_id,
            "symbols_seen": self.symbols_seen,
            "symbols_synced": self.symbols_synced,
            "files_uploaded": self.files_uploaded,
            "rows_uploaded": self.rows_uploaded,
            "skipped_closed": self.skipped_closed,
            "skipped_uptodate": self.skipped_uptodate,
            "import_job_id": self.import_job_id,
        }


class _RemoteIncrementalApiClient:
    def __init__(self) -> None:
        base_url = settings.market_data_incremental_remote_base_url.strip()
        if not base_url:
            raise ValueError(
                "MARKET_DATA_INCREMENTAL_REMOTE_BASE_URL is required for local collector."
            )
        self._base_url = base_url.rstrip("/")
        api_prefix = settings.api_v1_prefix.strip() or "/api/v1"
        self._api_prefix = "/" + api_prefix.strip("/")
        self._key_id = settings.market_data_incremental_hmac_key_id.strip()
        self._secret = settings.market_data_incremental_hmac_secret.strip()
        if not self._secret:
            raise ValueError(
                "MARKET_DATA_INCREMENTAL_HMAC_SECRET is required for local collector."
            )
        self._client = httpx.AsyncClient(
            timeout=max(float(settings.market_data_incremental_remote_timeout_seconds), 1.0),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    def _endpoint(self, path: str) -> str:
        normalized = "/" + path.lstrip("/")
        return f"{self._base_url}{normalized}"

    def _api_path(self, suffix: str) -> str:
        suffix_path = "/" + suffix.lstrip("/")
        return f"{self._api_prefix}{suffix_path}"

    def _headers(self, *, method: str, path: str, body: bytes) -> dict[str, str]:
        now_epoch = int(datetime.now(UTC).timestamp())
        payload = build_signature_payload(
            timestamp_epoch_seconds=now_epoch,
            method=method,
            path=path,
            body=body,
        )
        signature = sign_payload(secret=self._secret, payload=payload)
        return {
            "X-Minsy-Service-Key-Id": self._key_id,
            "X-Minsy-Service-Timestamp": str(now_epoch),
            "X-Minsy-Service-Signature": signature,
        }

    async def fetch_inventory(self) -> dict[str, Any]:
        path = self._api_path("/market-data/inventory")
        headers = self._headers(method="GET", path=path, body=b"")
        response = await self._client.get(self._endpoint(path), headers=headers)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Unexpected inventory payload")
        return payload

    async def notify_import(
        self,
        *,
        run_id: str,
        bucket: str,
        prefix: str,
        manifest_object: str,
        file_count: int,
        total_rows: int,
    ) -> dict[str, Any]:
        path = self._api_path("/market-data/incremental-imports")
        body_json = {
            "run_id": run_id,
            "bucket": bucket,
            "prefix": prefix,
            "manifest_object": manifest_object,
            "file_count": int(file_count),
            "total_rows": int(total_rows),
        }
        body = json.dumps(body_json, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        max_attempts = 5
        for attempt in range(1, max_attempts + 1):
            headers = self._headers(method="POST", path=path, body=body)
            headers["Content-Type"] = "application/json"
            try:
                response = await self._client.post(
                    self._endpoint(path),
                    headers=headers,
                    content=body,
                )
                if response.status_code >= 500 and attempt < max_attempts:
                    await asyncio.sleep(float(attempt))
                    continue
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise ValueError("Unexpected import notify payload")
                return payload
            except Exception:
                if attempt >= max_attempts:
                    raise
                await asyncio.sleep(float(attempt))
        raise RuntimeError("unreachable")


def _upload_file_with_fresh_client(
    *,
    local_path: Path,
    bucket_name: str,
    object_name: str,
    content_type: str | None = None,
) -> None:
    client = GcsClient()
    client.upload_file(
        local_path=local_path,
        bucket_name=bucket_name,
        object_name=object_name,
        content_type=content_type,
    )


def _upload_bytes_with_fresh_client(
    *,
    payload: bytes,
    bucket_name: str,
    object_name: str,
    content_type: str | None = None,
) -> None:
    client = GcsClient()
    client.upload_bytes(
        payload=payload,
        bucket_name=bucket_name,
        object_name=object_name,
        content_type=content_type,
    )


def _parse_iso_datetime(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
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


def _bars_to_frame(rows: list[OhlcvBar]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    payload = {
        "timestamp": [item.timestamp.astimezone(UTC).replace(second=0, microsecond=0) for item in rows],
        "open": [float(item.open) for item in rows],
        "high": [float(item.high) for item in rows],
        "low": [float(item.low) for item in rows],
        "close": [float(item.close) for item in rows],
        "volume": [float(item.volume) for item in rows],
    }
    frame = pd.DataFrame(payload)
    frame = frame.drop_duplicates(subset=["timestamp"], keep="last")
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    return frame


def _aggregate_5m(frame_1m: pd.DataFrame) -> pd.DataFrame:
    if frame_1m.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    data = frame_1m.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    data = data.set_index("timestamp").sort_index()
    agg = data.resample("5min", label="left", closed="left").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    agg = agg.dropna(subset=["open", "high", "low", "close"])
    agg = agg.reset_index()
    return agg


def _split_frame_by_month(frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    if frame.empty:
        return []
    data = frame.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    data["_month_key"] = data["timestamp"].dt.strftime("%Y-%m")
    chunks: list[tuple[str, pd.DataFrame]] = []
    for month_key, month_frame in data.groupby("_month_key", sort=True):
        normalized = month_frame.drop(columns=["_month_key"]).reset_index(drop=True)
        chunks.append((str(month_key), normalized))
    return chunks


def _split_windows(
    *,
    start: datetime,
    end: datetime,
    max_minutes: int,
) -> list[tuple[datetime, datetime]]:
    if end < start:
        return []
    windows: list[tuple[datetime, datetime]] = []
    cursor = start
    step = timedelta(minutes=max(1, int(max_minutes)) - 1)
    while cursor <= end:
        window_end = min(end, cursor + step)
        windows.append((cursor, window_end))
        cursor = window_end + timedelta(minutes=1)
    return windows


async def _fetch_alpaca_1m(
    *,
    symbol: str,
    market: str,
    start: datetime,
    end: datetime,
) -> list[OhlcvBar]:
    client = AlpacaMarketDataClient()
    try:
        merged: dict[datetime, OhlcvBar] = {}
        for window_start, window_end in _split_windows(
            start=start,
            end=end,
            max_minutes=1000,
        ):
            rows = await client.fetch_ohlcv(
                symbol=symbol,
                market=market,
                timeframe="1Min",
                since=window_start,
                until=window_end,
                limit=1000,
            )
            for item in rows:
                ts = item.timestamp.astimezone(UTC).replace(second=0, microsecond=0)
                if ts < start or ts > end:
                    continue
                merged[ts] = OhlcvBar(
                    timestamp=ts,
                    open=item.open,
                    high=item.high,
                    low=item.low,
                    close=item.close,
                    volume=item.volume,
                )
        return [merged[key] for key in sorted(merged)]
    finally:
        await client.aclose()


async def _fetch_ibkr_1m(
    *,
    symbol: str,
    market: str,
    start: datetime,
    end: datetime,
) -> list[OhlcvBar]:
    provider = IbkrAsyncMarketDataProvider()
    try:
        return await provider.fetch_ohlcv(
            symbol=symbol,
            market=market,
            timeframe="1m",
            since=start,
            until=end,
            # Keep request windows conservative to reduce IBKR historical timeout risk.
            limit=1_500,
        )
    finally:
        await provider.aclose()


def _coverage_end_for_symbol(item: dict[str, Any]) -> datetime | None:
    tf_coverage = item.get("timeframe_coverage")
    if isinstance(tf_coverage, dict):
        one_min = tf_coverage.get("1m")
        if isinstance(one_min, dict):
            parsed = _parse_iso_datetime(one_min.get("end"))
            if parsed is not None:
                return parsed
    coverage = item.get("coverage")
    if isinstance(coverage, dict):
        return _parse_iso_datetime(coverage.get("end"))
    return None


async def run_local_incremental_sync(
    *,
    window_start: datetime | None = None,
    window_end: datetime | None = None,
    respect_session_gate: bool = True,
    include_markets: set[str] | None = None,
) -> LocalIncrementalSyncResult:
    now_utc = datetime.now(UTC)
    run_id = now_utc.strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    if settings.market_data_incremental_execution_mode != "local_collector":
        return LocalIncrementalSyncResult(
            status="skipped_not_local_collector",
            run_id=run_id,
            symbols_seen=0,
            symbols_synced=0,
            files_uploaded=0,
            rows_uploaded=0,
            skipped_closed=0,
            skipped_uptodate=0,
        )

    bucket = settings.market_data_incremental_gcs_bucket.strip()
    if not bucket:
        raise ValueError("MARKET_DATA_INCREMENTAL_GCS_BUCKET is required for local collector.")

    api_client = _RemoteIncrementalApiClient()
    safety_lag = timedelta(minutes=max(int(settings.market_data_incremental_safety_lag_minutes), 0))
    default_end = (now_utc - safety_lag).replace(second=0, microsecond=0)
    if window_end is None:
        effective_end = default_end
    else:
        effective_end = window_end.astimezone(UTC).replace(second=0, microsecond=0)
        if effective_end > default_end:
            effective_end = default_end
    forced_start = (
        window_start.astimezone(UTC).replace(second=0, microsecond=0)
        if window_start is not None
        else None
    )
    symbols_seen = 0
    symbols_synced = 0
    files_uploaded = 0
    rows_uploaded = 0
    skipped_closed = 0
    skipped_uptodate = 0
    prefix_root = settings.market_data_incremental_gcs_prefix.strip().strip("/")
    date_folder = now_utc.strftime("%Y-%m-%d")
    prefix = "/".join(part for part in (prefix_root, date_folder, run_id) if part)
    stage_dir = Path("runtime/incremental") / date_folder / run_id
    stage_dir.mkdir(parents=True, exist_ok=True)
    manifest_entries: list[dict[str, Any]] = []

    try:
        inventory = await api_client.fetch_inventory()
        markets = inventory.get("markets")
        if not isinstance(markets, list):
            markets = []

        for market_item in markets:
            if not isinstance(market_item, dict):
                continue
            market = str(market_item.get("market") or "").strip().lower()
            symbols = market_item.get("symbols")
            if not market or not isinstance(symbols, list):
                continue
            if include_markets:
                try:
                    normalized_market = normalize_incremental_market(market)
                except ValueError:
                    continue
                if normalized_market not in include_markets:
                    continue

            for symbol_item in symbols:
                if not isinstance(symbol_item, dict):
                    continue
                symbols_seen += 1
                symbol = str(symbol_item.get("symbol") or "").strip().upper()
                if not symbol:
                    skipped_uptodate += 1
                    continue
                session = str(symbol_item.get("session") or "eth").strip().lower()
                if session not in {"rth", "eth"}:
                    session = "eth"
                try:
                    is_open = market_is_open_for_incremental(market=market, now=now_utc)
                    provider = resolve_provider_for_market(market)
                except ValueError:
                    logger.info(
                        "[market-data-incremental] skip unsupported market=%s symbol=%s",
                        market,
                        symbol,
                    )
                    skipped_uptodate += 1
                    continue

                if respect_session_gate and not is_open:
                    skipped_closed += 1
                    continue

                coverage_end = _coverage_end_for_symbol(symbol_item)
                if coverage_end is None:
                    # Conservative bootstrap range for unknown coverage.
                    coverage_end = effective_end - timedelta(days=1)
                start = (coverage_end + timedelta(minutes=1)).replace(second=0, microsecond=0)
                if forced_start is not None:
                    start = max(start, forced_start)
                if start >= effective_end:
                    skipped_uptodate += 1
                    continue

                try:
                    if provider == "alpaca":
                        fetch_coro = _fetch_alpaca_1m(
                            symbol=symbol,
                            market=market,
                            start=start,
                            end=effective_end,
                        )
                        bars_1m = await asyncio.wait_for(fetch_coro, timeout=180.0)
                    else:
                        fetch_coro = _fetch_ibkr_1m(
                            symbol=symbol,
                            market=market,
                            start=start,
                            end=effective_end,
                        )
                        bars_1m = await asyncio.wait_for(fetch_coro, timeout=600.0)
                except Exception:
                    logger.exception(
                        "[market-data-incremental] skip symbol after provider fetch failure "
                        "market=%s symbol=%s start=%s end=%s",
                        market,
                        symbol,
                        start.isoformat(),
                        effective_end.isoformat(),
                    )
                    skipped_uptodate += 1
                    continue
                frame_1m = _bars_to_frame(bars_1m)
                if frame_1m.empty:
                    skipped_uptodate += 1
                    continue

                frame_5m = _aggregate_5m(frame_1m)
                symbol_dir = stage_dir / market / symbol / session
                symbol_dir.mkdir(parents=True, exist_ok=True)
                chunked_1m = _split_frame_by_month(frame_1m)
                if not chunked_1m:
                    skipped_uptodate += 1
                    continue

                chunked_5m = _split_frame_by_month(frame_5m) if not frame_5m.empty else []

                for month_key, frame_1m_chunk in chunked_1m:
                    file_1m = symbol_dir / "1m" / f"{month_key}.parquet"
                    file_1m.parent.mkdir(parents=True, exist_ok=True)
                    frame_1m_chunk.to_parquet(file_1m, index=False)
                    object_1m = f"{prefix}/{market}/{symbol}/{session}/1m/{month_key}.parquet"
                    await asyncio.to_thread(
                        _upload_file_with_fresh_client,
                        local_path=file_1m,
                        bucket_name=bucket,
                        object_name=object_1m,
                        content_type="application/octet-stream",
                    )
                    files_uploaded += 1
                    rows_uploaded += int(len(frame_1m_chunk))
                    manifest_entries.append(
                        {
                            "market": market,
                            "symbol": symbol,
                            "timeframe": "1m",
                            "session": session,
                            "gcs_object": object_1m,
                            "row_count": int(len(frame_1m_chunk)),
                            "start": frame_1m_chunk["timestamp"].min().isoformat(),
                            "end": frame_1m_chunk["timestamp"].max().isoformat(),
                        }
                    )

                for month_key, frame_5m_chunk in chunked_5m:
                    file_5m = symbol_dir / "5m" / f"{month_key}.parquet"
                    file_5m.parent.mkdir(parents=True, exist_ok=True)
                    frame_5m_chunk.to_parquet(file_5m, index=False)
                    object_5m = f"{prefix}/{market}/{symbol}/{session}/5m/{month_key}.parquet"
                    await asyncio.to_thread(
                        _upload_file_with_fresh_client,
                        local_path=file_5m,
                        bucket_name=bucket,
                        object_name=object_5m,
                        content_type="application/octet-stream",
                    )
                    files_uploaded += 1
                    rows_uploaded += int(len(frame_5m_chunk))
                    manifest_entries.append(
                        {
                            "market": market,
                            "symbol": symbol,
                            "timeframe": "5m",
                            "session": session,
                            "gcs_object": object_5m,
                            "row_count": int(len(frame_5m_chunk)),
                            "start": frame_5m_chunk["timestamp"].min().isoformat(),
                            "end": frame_5m_chunk["timestamp"].max().isoformat(),
                        }
                    )
                symbols_synced += 1

        if not manifest_entries:
            return LocalIncrementalSyncResult(
                status="no_data",
                run_id=run_id,
                symbols_seen=symbols_seen,
                symbols_synced=symbols_synced,
                files_uploaded=files_uploaded,
                rows_uploaded=rows_uploaded,
                skipped_closed=skipped_closed,
                skipped_uptodate=skipped_uptodate,
            )

        manifest = {
            "run_id": run_id,
            "generated_at": now_utc.isoformat(),
            "bucket": bucket,
            "prefix": prefix,
            "files": manifest_entries,
        }
        manifest_bytes = json.dumps(manifest, ensure_ascii=False).encode("utf-8")
        manifest_object = f"{prefix}/manifest.json"
        await asyncio.to_thread(
            _upload_bytes_with_fresh_client,
            payload=manifest_bytes,
            bucket_name=bucket,
            object_name=manifest_object,
            content_type="application/json",
        )

        notify_response = await api_client.notify_import(
            run_id=run_id,
            bucket=bucket,
            prefix=prefix,
            manifest_object=manifest_object,
            file_count=files_uploaded,
            total_rows=rows_uploaded,
        )
        import_job_id = str(notify_response.get("import_job_id") or "").strip() or None
        return LocalIncrementalSyncResult(
            status="ok",
            run_id=run_id,
            symbols_seen=symbols_seen,
            symbols_synced=symbols_synced,
            files_uploaded=files_uploaded,
            rows_uploaded=rows_uploaded,
            skipped_closed=skipped_closed,
            skipped_uptodate=skipped_uptodate,
            import_job_id=import_job_id,
        )
    except Exception:
        logger.exception("[market-data-incremental] local incremental sync failed run_id=%s", run_id)
        raise
    finally:
        await api_client.aclose()
