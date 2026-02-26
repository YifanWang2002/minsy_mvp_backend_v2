"""Launch all MCP domain servers and one domain router in a single process tree."""

from __future__ import annotations

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from types import FrameType


@dataclass(frozen=True, slots=True)
class DomainProcessSpec:
    domain: str
    port: int


DOMAIN_SPECS: tuple[DomainProcessSpec, ...] = (
    DomainProcessSpec("strategy", 8111),
    DomainProcessSpec("backtest", 8112),
    DomainProcessSpec("market", 8113),
    DomainProcessSpec("stress", 8114),
    DomainProcessSpec("trading", 8115),
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run all MCP domain servers and one router process."
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable path.")
    parser.add_argument(
        "--transport",
        choices=("streamable-http", "sse"),
        default="streamable-http",
        help="Transport for domain MCP servers.",
    )
    parser.add_argument(
        "--mcp-host",
        default="127.0.0.1",
        help="Host for internal MCP domain servers.",
    )
    parser.add_argument(
        "--mount-path",
        default="/",
        help="Mount path for MCP domain servers.",
    )
    parser.add_argument(
        "--stateful-http",
        action="store_true",
        help="Use stateful streamable-http mode for domain servers.",
    )
    parser.add_argument("--router-host", default="0.0.0.0", help="Host for MCP router.")
    parser.add_argument("--router-port", type=int, default=8110, help="Port for MCP router.")
    parser.add_argument(
        "--router-log-level",
        default="info",
        help="Log level for router process.",
    )
    parser.add_argument(
        "--startup-wait-seconds",
        type=float,
        default=0.2,
        help="Delay between child process starts.",
    )
    return parser


def _domain_command(
    *,
    python_executable: str,
    domain: str,
    port: int,
    transport: str,
    host: str,
    mount_path: str,
    stateful_http: bool,
) -> list[str]:
    command = [
        python_executable,
        "-m",
        "apps.mcp.gateway",
        "--domain",
        domain,
        "--transport",
        transport,
        "--host",
        host,
        "--port",
        str(port),
        "--mount-path",
        mount_path,
    ]
    if stateful_http:
        command.append("--stateful-http")
    return command


def _router_command(
    *,
    python_executable: str,
    host: str,
    port: int,
    log_level: str,
) -> list[str]:
    return [
        python_executable,
        "-m",
        "apps.mcp.proxy.router",
        "--host",
        host,
        "--port",
        str(port),
        "--log-level",
        log_level,
    ]


def _build_child_env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("MINSY_SERVICE", "mcp")
    env.setdefault("MCP_PROXY_UPSTREAM_STRATEGY", "http://127.0.0.1:8111")
    env.setdefault("MCP_PROXY_UPSTREAM_BACKTEST", "http://127.0.0.1:8112")
    env.setdefault("MCP_PROXY_UPSTREAM_MARKET", "http://127.0.0.1:8113")
    env.setdefault("MCP_PROXY_UPSTREAM_STRESS", "http://127.0.0.1:8114")
    env.setdefault("MCP_PROXY_UPSTREAM_TRADING", "http://127.0.0.1:8115")
    env.setdefault("MCP_PROXY_UPSTREAM_LEGACY", "http://127.0.0.1:8111")
    return env


def _terminate_children(
    children: list[tuple[str, subprocess.Popen[bytes]]],
    logger: logging.Logger,
    *,
    grace_seconds: float = 8.0,
) -> None:
    if not children:
        return

    for name, proc in children:
        if proc.poll() is None:
            logger.info("stopping child=%s pid=%s", name, proc.pid)
            proc.terminate()

    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        alive = [proc for _, proc in children if proc.poll() is None]
        if not alive:
            return
        time.sleep(0.2)

    for name, proc in children:
        if proc.poll() is None:
            logger.warning("forcing kill child=%s pid=%s", name, proc.pid)
            proc.kill()


def main() -> int:
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logger = logging.getLogger("apps.mcp.launch_all")
    env = _build_child_env()

    stop_requested = {"value": False}

    def _signal_handler(signum: int, _frame: FrameType | None) -> None:
        if stop_requested["value"]:
            return
        stop_requested["value"] = True
        logger.info("signal received signum=%s, shutting down", signum)

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    children: list[tuple[str, subprocess.Popen[bytes]]] = []
    try:
        for spec in DOMAIN_SPECS:
            command = _domain_command(
                python_executable=args.python,
                domain=spec.domain,
                port=spec.port,
                transport=args.transport,
                host=args.mcp_host,
                mount_path=args.mount_path,
                stateful_http=bool(args.stateful_http),
            )
            logger.info(
                "starting mcp domain=%s port=%s transport=%s",
                spec.domain,
                spec.port,
                args.transport,
            )
            process = subprocess.Popen(command, env=env)  # noqa: S603
            children.append((f"mcp:{spec.domain}", process))
            time.sleep(max(args.startup_wait_seconds, 0.0))

        router_command = _router_command(
            python_executable=args.python,
            host=args.router_host,
            port=args.router_port,
            log_level=args.router_log_level,
        )
        logger.info("starting mcp router host=%s port=%s", args.router_host, args.router_port)
        router_process = subprocess.Popen(router_command, env=env)  # noqa: S603
        children.append(("router", router_process))

        logger.info("all mcp child processes started count=%s", len(children))
        while not stop_requested["value"]:
            failed: tuple[str, subprocess.Popen[bytes]] | None = None
            for name, process in children:
                code = process.poll()
                if code is None:
                    continue
                failed = (name, process)
                logger.error("child exited unexpectedly name=%s code=%s", name, code)
                stop_requested["value"] = True
                break

            if failed is not None:
                break
            time.sleep(0.5)
    finally:
        _terminate_children(children, logger)

    non_zero_codes = [proc.returncode for _, proc in children if (proc.returncode or 0) != 0]
    if non_zero_codes:
        return int(non_zero_codes[0] or 1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
