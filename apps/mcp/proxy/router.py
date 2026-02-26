"""Reverse proxy for domain-prefixed MCP routes."""

from __future__ import annotations

import argparse
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, StreamingResponse
from starlette.background import BackgroundTask

logger = logging.getLogger(__name__)

_REQUEST_HOP_BY_HOP_HEADERS: frozenset[str] = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "host",
    }
)

_RESPONSE_HOP_BY_HOP_HEADERS: frozenset[str] = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)

_RESPONSE_STRIP_HEADERS: frozenset[str] = (
    _RESPONSE_HOP_BY_HOP_HEADERS | frozenset({"date", "server"})
)


@dataclass(frozen=True, slots=True)
class ProxyRoute:
    prefix: str
    upstream_env: str
    default_upstream: str
    strip_prefix: bool


@dataclass(frozen=True, slots=True)
class ProxyTarget:
    upstream_base_url: str
    rewritten_path: str


_ROUTES: tuple[ProxyRoute, ...] = (
    ProxyRoute(
        prefix="/strategy",
        upstream_env="MCP_PROXY_UPSTREAM_STRATEGY",
        default_upstream="http://127.0.0.1:8111",
        strip_prefix=True,
    ),
    ProxyRoute(
        prefix="/backtest",
        upstream_env="MCP_PROXY_UPSTREAM_BACKTEST",
        default_upstream="http://127.0.0.1:8112",
        strip_prefix=True,
    ),
    ProxyRoute(
        prefix="/market",
        upstream_env="MCP_PROXY_UPSTREAM_MARKET",
        default_upstream="http://127.0.0.1:8113",
        strip_prefix=True,
    ),
    ProxyRoute(
        prefix="/stress",
        upstream_env="MCP_PROXY_UPSTREAM_STRESS",
        default_upstream="http://127.0.0.1:8114",
        strip_prefix=True,
    ),
    ProxyRoute(
        prefix="/trading",
        upstream_env="MCP_PROXY_UPSTREAM_TRADING",
        default_upstream="http://127.0.0.1:8115",
        strip_prefix=True,
    ),
    ProxyRoute(
        prefix="/mcp",
        upstream_env="MCP_PROXY_UPSTREAM_LEGACY",
        default_upstream="http://127.0.0.1:8111",
        strip_prefix=False,
    ),
)


def _path_matches_prefix(path: str, prefix: str) -> bool:
    return path == prefix or path.startswith(prefix + "/")


def _normalize_path(path: str) -> str:
    if not path:
        return "/"
    return path if path.startswith("/") else "/" + path


def _resolve_upstream_base_url(route: ProxyRoute) -> str:
    candidate = os.getenv(route.upstream_env, "").strip()
    if candidate:
        return candidate.rstrip("/")
    return route.default_upstream.rstrip("/")


def resolve_proxy_target(path: str) -> ProxyTarget | None:
    """Map incoming path to target upstream base URL and rewritten path."""
    normalized_path = _normalize_path(path)
    for route in _ROUTES:
        if not _path_matches_prefix(normalized_path, route.prefix):
            continue
        if route.strip_prefix:
            rewritten = normalized_path[len(route.prefix) :]
            rewritten = _normalize_path(rewritten)
        else:
            rewritten = normalized_path
        return ProxyTarget(
            upstream_base_url=_resolve_upstream_base_url(route),
            rewritten_path=rewritten,
        )
    return None


def build_upstream_url(
    *,
    upstream_base_url: str,
    rewritten_path: str,
    query: str,
) -> str:
    base = upstream_base_url.rstrip("/")
    path = _normalize_path(rewritten_path)
    if query:
        return f"{base}{path}?{query}"
    return f"{base}{path}"


@asynccontextmanager
async def _lifespan(app: FastAPI):
    app.state.http = httpx.AsyncClient(
        follow_redirects=False,
        timeout=None,
        trust_env=False,
    )
    try:
        yield
    finally:
        await app.state.http.aclose()


app = FastAPI(title="MCP Reverse Proxy", lifespan=_lifespan)


@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
)
async def proxy_request(path: str, request: Request):
    incoming_path = "/" + path
    target = resolve_proxy_target(incoming_path)
    if target is None:
        return PlainTextResponse("MCP route not found", status_code=404)

    upstream_url = build_upstream_url(
        upstream_base_url=target.upstream_base_url,
        rewritten_path=target.rewritten_path,
        query=request.url.query,
    )

    request_headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in _REQUEST_HOP_BY_HOP_HEADERS
    }
    body = await request.body()

    upstream_request = request.app.state.http.build_request(
        method=request.method,
        url=upstream_url,
        headers=request_headers,
        content=body,
    )
    try:
        upstream_response = await request.app.state.http.send(upstream_request, stream=True)
    except httpx.HTTPError as exc:
        logger.warning(
            "mcp proxy upstream request failed method=%s path=%s upstream=%s error=%s",
            request.method,
            incoming_path,
            upstream_url,
            exc,
        )
        return PlainTextResponse(
            f"MCP upstream unavailable: {upstream_url}",
            status_code=502,
        )

    response_headers = {
        key: value
        for key, value in upstream_response.headers.items()
        if key.lower() not in _RESPONSE_STRIP_HEADERS
    }
    return StreamingResponse(
        upstream_response.aiter_raw(),
        status_code=upstream_response.status_code,
        headers=response_headers,
        background=BackgroundTask(upstream_response.aclose),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run reverse proxy for domain-prefixed MCP routes."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8110)
    parser.add_argument("--log-level", default="info")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        access_log=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
