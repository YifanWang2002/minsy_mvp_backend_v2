from __future__ import annotations

import json
import re
import selectors
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

BACKEND_DIR = Path(__file__).resolve().parents[2]
COMPOSE_FILE = BACKEND_DIR / "compose.dev.yml"
_EXPECTED_COMPOSE_SERVICES = (
    "postgres",
    "redis",
    "mcp",
    "api",
    "worker-cpu",
    "worker-io",
    "beat",
)


@dataclass(frozen=True, slots=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


def run_command(
    args: list[str],
    *,
    cwd: Path | None = None,
    timeout: int = 300,
    check: bool = True,
) -> CommandResult:
    completed = subprocess.run(
        args,
        cwd=str(cwd or BACKEND_DIR),
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    result = CommandResult(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
    if check and result.returncode != 0:
        output = (result.stdout + "\n" + result.stderr).strip()
        raise AssertionError(
            f"Command failed ({result.returncode}): {' '.join(args)}\n{output}"
        )
    return result


def parse_sse_payloads(raw_text: str) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for block in [item.strip() for item in raw_text.split("\n\n") if item.strip()]:
        for line in block.splitlines():
            if not line.startswith("data: "):
                continue
            payloads.append(json.loads(line.removeprefix("data: ")))
    return payloads


def compose_ps() -> list[dict[str, Any]]:
    result = run_command(
        [
            "docker",
            "compose",
            "-f",
            str(COMPOSE_FILE),
            "ps",
            "--format",
            "json",
        ],
        cwd=BACKEND_DIR,
    )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return [json.loads(line) for line in lines]


def _compose_service_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("Service", "")).strip(): row
        for row in rows
        if isinstance(row, dict)
    }


def ensure_compose_stack_up(*, timeout_seconds: int = 420) -> list[dict[str, Any]]:
    run_command(
        [
            "docker",
            "compose",
            "-f",
            str(COMPOSE_FILE),
            "up",
            "-d",
            "--build",
        ],
        cwd=BACKEND_DIR,
        timeout=1200,
    )

    deadline = time.monotonic() + float(timeout_seconds)
    last_rows: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        rows = compose_ps()
        last_rows = rows
        service_map = _compose_service_map(rows)

        if not all(name in service_map for name in _EXPECTED_COMPOSE_SERVICES):
            time.sleep(3)
            continue

        all_running = all(
            str(service_map[name].get("State", "")).strip().lower() == "running"
            for name in _EXPECTED_COMPOSE_SERVICES
        )
        if not all_running:
            time.sleep(3)
            continue

        healthy_services = ("postgres", "redis", "mcp", "api")
        all_healthy = all(
            str(service_map[name].get("Health", "")).strip().lower() in {"healthy", ""}
            for name in healthy_services
        )
        if all_healthy:
            return rows
        time.sleep(3)

    raise AssertionError(f"compose stack not ready: {last_rows}")


def wait_http_ok(url: str, *, timeout_seconds: int = 90, min_status: int = 200, max_status: int = 499) -> int:
    deadline = time.monotonic() + float(timeout_seconds)
    last_error: str = ""
    while time.monotonic() < deadline:
        try:
            with urlopen(url, timeout=8) as response:  # noqa: S310
                status_code = int(response.status)
            if min_status <= status_code <= max_status:
                return status_code
            last_error = f"status={status_code}"
        except HTTPError as exc:
            status_code = int(exc.code)
            if min_status <= status_code <= max_status:
                return status_code
            last_error = f"status={status_code}"
        except URLError as exc:
            last_error = str(exc)
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(2)
    raise AssertionError(f"HTTP probe failed for {url}: {last_error}")


def start_cloudflared_tunnel(*, target_url: str = "http://127.0.0.1:8110") -> tuple[subprocess.Popen[str], str]:
    process = subprocess.Popen(
        [
            "cloudflared",
            "tunnel",
            "--url",
            target_url,
            "--no-autoupdate",
        ],
        cwd=str(BACKEND_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    pattern = re.compile(r"https://[-a-z0-9]+\\.trycloudflare\\.com")
    if process.stdout is None:
        stop_process(process)
        raise AssertionError("cloudflared stdout unavailable.")

    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    deadline = time.monotonic() + 90.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            break
        events = selector.select(timeout=1.0)
        if not events:
            continue
        for key, _ in events:
            line = key.fileobj.readline()
            if not line:
                continue
            match = pattern.search(line)
            if match is not None:
                selector.unregister(process.stdout)
                selector.close()
                return process, match.group(0)

    try:
        remaining = process.stdout.read() if process.stdout is not None else ""
    except Exception:  # noqa: BLE001
        remaining = ""
    stop_process(process)
    raise AssertionError(f"Failed to parse cloudflared public URL. output={remaining[:500]}")


def stop_process(process: subprocess.Popen[Any] | None) -> None:
    if process is None:
        return
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
