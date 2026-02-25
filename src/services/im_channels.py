"""IM channel provider abstraction for notification dispatch."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from src.models.social_connector import SocialConnectorBinding


@dataclass(frozen=True, slots=True)
class IMDeliveryResult:
    """Unified channel-delivery result payload."""

    success: bool
    provider: str
    provider_message_id: str | None = None
    request_payload: dict[str, Any] = field(default_factory=dict)
    response_payload: dict[str, Any] = field(default_factory=dict)
    error_code: str | None = None
    error_message: str | None = None


class IMChannelProvider(Protocol):
    """Protocol for pluggable IM delivery providers."""

    channel: str

    async def send_event(
        self,
        *,
        binding: SocialConnectorBinding,
        event_type: str,
        payload: dict[str, Any],
        locale: str,
    ) -> IMDeliveryResult:
        """Send one notification event via IM channel."""


class IMProviderRegistry:
    """Channel -> provider registry to support multi-IM expansion."""

    def __init__(self) -> None:
        self._providers: dict[str, IMChannelProvider] = {}

    def register(self, provider: IMChannelProvider) -> None:
        channel = str(provider.channel).strip().lower()
        if not channel:
            raise ValueError("Provider channel cannot be empty.")
        self._providers[channel] = provider

    def get(self, channel: str) -> IMChannelProvider | None:
        return self._providers.get(str(channel).strip().lower())

    def channels(self) -> tuple[str, ...]:
        return tuple(sorted(self._providers))
