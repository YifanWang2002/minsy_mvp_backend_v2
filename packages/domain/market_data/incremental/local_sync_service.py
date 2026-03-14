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
        headers = self._headers(method="POST", path=path, body=body)
        headers["Content-Type"] = "application/json"
        response = await self._client.post(
            self._endpoint(path),
            headers=headers,
            content=body,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Unexpected import notify payload")
        return payload


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
            limit=1500,
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


async def run_local_incremental_sync() -> LocalIncrementalSyncResult:
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

    gcs_client = GcsClient()
    api_client = _RemoteIncrementalApiClient()
    safety_lag = timedelta(minutes=max(int(settings.market_data_incremental_safety_lag_minutes), 0))
    effective_end = (now_utc - safety_lag).replace(second=0, microsecond=0)
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

            for symbol_item in symbols:
                if not isinstance(symbol_item, dict):
                    continue
                symbols_seen += 1
                symbol = str(symbol_item.get("symbol") or "").strip().upper()
                if not symbol:
                    skipped_uptodate += 1
                    continue
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

                if not is_open:
                    skipped_closed += 1
                    continue

                coverage_end = _coverage_end_for_symbol(symbol_item)
                if coverage_end is None:
                    # Conservative bootstrap range for unknown coverage.
                    coverage_end = effective_end - timedelta(days=1)
                start = (coverage_end + timedelta(minutes=1)).replace(second=0, microsecond=0)
                if start >= effective_end:
                    skipped_uptodate += 1
                    continue

                if provider == "alpaca":
                    bars_1m = await _fetch_alpaca_1m(
                        symbol=symbol,
                        market=market,
                        start=start,
                        end=effective_end,
                    )
                else:
                    bars_1m = await _fetch_ibkr_1m(
                        symbol=symbol,
                        market=market,
                        start=start,
                        end=effective_end,
                    )
                frame_1m = _bars_to_frame(bars_1m)
                if frame_1m.empty:
                    skipped_uptodate += 1
                    continue

                frame_5m = _aggregate_5m(frame_1m)
                symbol_dir = stage_dir / market / symbol
                symbol_dir.mkdir(parents=True, exist_ok=True)
                file_1m = symbol_dir / "1m.parquet"
                frame_1m.to_parquet(file_1m, index=False)

                object_1m = f"{prefix}/{market}/{symbol}/1m.parquet"
                await asyncio.to_thread(
                    gcs_client.upload_file,
                    local_path=file_1m,
                    bucket_name=bucket,
                    object_name=object_1m,
                    content_type="application/octet-stream",
                )

                files_uploaded += 1
                rows_uploaded += int(len(frame_1m))
                symbols_synced += 1
                start_iso = frame_1m["timestamp"].min().isoformat()
                end_iso = frame_1m["timestamp"].max().isoformat()
                manifest_entries.append(
                    {
                        "market": market,
                        "symbol": symbol,
                        "timeframe": "1m",
                        "session": symbol_item.get("session", "eth"),
                        "gcs_object": object_1m,
                        "row_count": int(len(frame_1m)),
                        "start": start_iso,
                        "end": end_iso,
                    }
                )
                if not frame_5m.empty:
                    file_5m = symbol_dir / "5m.parquet"
                    frame_5m.to_parquet(file_5m, index=False)
                    object_5m = f"{prefix}/{market}/{symbol}/5m.parquet"
                    await asyncio.to_thread(
                        gcs_client.upload_file,
                        local_path=file_5m,
                        bucket_name=bucket,
                        object_name=object_5m,
                        content_type="application/octet-stream",
                    )
                    files_uploaded += 1
                    rows_uploaded += int(len(frame_5m))
                    manifest_entries.append(
                        {
                            "market": market,
                            "symbol": symbol,
                            "timeframe": "5m",
                            "session": symbol_item.get("session", "eth"),
                            "gcs_object": object_5m,
                            "row_count": int(len(frame_5m)),
                            "start": frame_5m["timestamp"].min().isoformat(),
                            "end": frame_5m["timestamp"].max().isoformat(),
                        }
                    )

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
            gcs_client.upload_bytes,
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
