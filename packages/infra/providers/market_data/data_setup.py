"""Download and extract market-data parquet files on first startup."""

from __future__ import annotations

import subprocess
from pathlib import Path

import httpx

from packages.infra.observability.logger import log_success, logger

_URLS: dict[str, str] = {
    "crypto": "https://storage.googleapis.com/minsy-mvp-data/crypto.7z",
    "futures": "https://storage.googleapis.com/minsy-mvp-data/futures.7z",
    "forex": "https://storage.googleapis.com/minsy-mvp-data/forex.7z",
    "us_stocks": "https://storage.googleapis.com/minsy-mvp-data/us_stocks.7z",
}

# Resolve relative to the *backend* package root (cwd when uvicorn starts).
_DATA_DIR = Path("data").resolve()


def _has_parquet_files() -> bool:
    """Return True if at least one .parquet file exists under _DATA_DIR."""
    if not _DATA_DIR.exists():
        return False
    return any(_DATA_DIR.rglob("*.parquet"))


def _download_file(url: str, output_path: Path) -> None:
    """Stream-download a large file with httpx."""
    with httpx.stream("GET", url, timeout=300, follow_redirects=True) as r:
        r.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in r.iter_bytes(chunk_size=8 * 1024 * 1024):
                f.write(chunk)


def _extract_7z(archive_path: Path, extract_to: Path) -> None:
    """Extract a .7z archive using the system 7z binary."""
    extract_to.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["7z", "x", str(archive_path), f"-o{extract_to}", "-y"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"7z extraction failed for {archive_path.name}:\n{result.stderr}"
        )


def _log_data_report() -> None:
    """Scan _DATA_DIR and log a summary of markets and tickers found."""
    if not _DATA_DIR.exists():
        return

    # Collect tickers per market sub-directory.
    # Filename pattern: {TICKER}_{timeframe}_{session}_{year}.parquet
    market_tickers: dict[str, set[str]] = {}
    total_files = 0

    for parquet_file in _DATA_DIR.rglob("*.parquet"):
        total_files += 1
        market = parquet_file.parent.name          # e.g. "crypto", "futures"
        ticker = parquet_file.stem.split("_")[0]   # e.g. "BTCUSD", "GC"
        market_tickers.setdefault(market, set()).add(ticker)

    if not market_tickers:
        logger.warning("No parquet files found in %s — report skipped.", _DATA_DIR)
        return

    # Build a readable report.
    lines = [
        "",
        "╔══════════════════════════════════════════════════╗",
        "║           Market Data Inventory Report           ║",
        "╚══════════════════════════════════════════════════╝",
        f"  Data directory : {_DATA_DIR}",
        f"  Total files    : {total_files}",
        f"  Markets        : {len(market_tickers)}",
        "",
    ]

    for market in sorted(market_tickers):
        tickers = sorted(market_tickers[market])
        lines.append(f"  [{market}] ({len(tickers)} tickers)")
        lines.append(f"    {', '.join(tickers)}")
        lines.append("")

    lines.append("══════════════════════════════════════════════════")

    logger.info("\n".join(lines))


def ensure_market_data() -> None:
    """Check for parquet data; download & extract if missing.

    Intended to be called once during application startup.
    """
    if _has_parquet_files():
        log_success(
            f"Market data already present in {_DATA_DIR} — skipping download."
        )
        _log_data_report()
        return

    logger.info("No parquet files found in %s — downloading market data …", _DATA_DIR)
    _DATA_DIR.mkdir(parents=True, exist_ok=True)

    failed: list[str] = []

    for name, url in _URLS.items():
        archive_path = _DATA_DIR / f"{name}.7z"
        extract_dir = _DATA_DIR / name

        try:
            logger.info("[%s] Downloading from %s …", name, url)
            _download_file(url, archive_path)

            logger.info("[%s] Extracting …", name)
            _extract_7z(archive_path, extract_dir)

            # Basic sanity check — directory should not be empty.
            if not any(extract_dir.iterdir()):
                raise RuntimeError(f"Extraction appears empty for {name}")

            # Clean up the archive to save disk space.
            archive_path.unlink()
            logger.info("[%s] Done — archive removed.", name)

        except Exception:
            failed.append(name)
            logger.exception("[%s] Failed to download or extract — skipping.", name)
            # Clean up partial archive if it was left behind.
            if archive_path.exists():
                archive_path.unlink(missing_ok=True)

    if failed:
        logger.warning(
            "Market data setup finished with errors. "
            "Failed datasets: %s. The app will start, but some data may be missing.",
            ", ".join(failed),
        )
    else:
        log_success("All market datasets downloaded and ready.")

    _log_data_report()
