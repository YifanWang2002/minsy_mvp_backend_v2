"""Service settings loaders built on top of the legacy config module."""

from __future__ import annotations

from functools import lru_cache

from packages.shared_settings.schema import (
    ApiSettings,
    BeatSettings,
    CommonSettings,
    McpSettings,
    WorkerCpuSettings,
    WorkerIoSettings,
)
from packages.shared_settings.schema.settings import Settings, get_settings


@lru_cache
def _load_legacy_settings(
    *,
    env_file: str | None = None,
    service: str | None = None,
) -> Settings:
    return get_settings(env_file=env_file, service=service)


@lru_cache
def get_common_settings(env_file: str | None = None) -> CommonSettings:
    return CommonSettings(_load_legacy_settings(env_file=env_file, service="common"))


@lru_cache
def get_api_settings(env_file: str | None = None) -> ApiSettings:
    return ApiSettings(_load_legacy_settings(env_file=env_file, service="api"))


@lru_cache
def get_mcp_settings(env_file: str | None = None) -> McpSettings:
    return McpSettings(_load_legacy_settings(env_file=env_file, service="mcp"))


@lru_cache
def get_worker_cpu_settings(env_file: str | None = None) -> WorkerCpuSettings:
    return WorkerCpuSettings(_load_legacy_settings(env_file=env_file, service="worker_cpu"))


@lru_cache
def get_worker_io_settings(env_file: str | None = None) -> WorkerIoSettings:
    return WorkerIoSettings(_load_legacy_settings(env_file=env_file, service="worker_io"))


@lru_cache
def get_beat_settings(env_file: str | None = None) -> BeatSettings:
    return BeatSettings(_load_legacy_settings(env_file=env_file, service="beat"))
