"""Domain queue publisher ports.

Engine/domain services publish async jobs through these ports instead of
importing Celery worker modules directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
from uuid import UUID

PublishFn = Callable[[UUID], str]


@dataclass(frozen=True, slots=True)
class JobQueuePorts:
    """Queue publishing contract used by domain services."""

    enqueue_backtest_job: PublishFn
    enqueue_stress_job: PublishFn
    enqueue_market_data_sync_job: PublishFn


_ports: JobQueuePorts | None = None


def configure_job_queue_ports(ports: JobQueuePorts) -> None:
    """Register runtime queue publisher implementations."""

    global _ports
    _ports = ports


def reset_job_queue_ports() -> None:
    """Test helper to clear configured publishers."""

    global _ports
    _ports = None


def get_job_queue_ports() -> JobQueuePorts:
    """Return configured publishers; lazily bootstrap default infra adapter."""

    global _ports
    if _ports is None:
        from packages.infra.queue.publishers import configure_default_job_queue_ports

        configure_default_job_queue_ports()
    assert _ports is not None
    return _ports
