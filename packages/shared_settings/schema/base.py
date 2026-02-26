"""Service-level settings proxies (compatibility wrappers)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from packages.shared_settings.schema.settings import Settings


class _SettingsProxy:
    """Forward all attribute access to legacy settings during migration."""

    __slots__ = ("_legacy",)

    def __init__(self, legacy: "Settings") -> None:
        object.__setattr__(self, "_legacy", legacy)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._legacy, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_legacy":
            object.__setattr__(self, name, value)
            return
        setattr(self._legacy, name, value)

    @property
    def legacy(self) -> "Settings":
        return self._legacy


class CommonSettings(_SettingsProxy):
    pass


class ApiSettings(_SettingsProxy):
    pass


class McpSettings(_SettingsProxy):
    pass


class WorkerCpuSettings(_SettingsProxy):
    pass


class WorkerIoSettings(_SettingsProxy):
    pass


class BeatSettings(_SettingsProxy):
    pass
