"""Celery CPU worker entrypoint (backtest + stress queues)."""

from __future__ import annotations

import os

# Entry-point scoped profile selection for Celery worker startup.
os.environ["MINSY_CELERY_PROFILE"] = "cpu"
os.environ.setdefault("MINSY_SERVICE", "worker_cpu")

from apps.worker.common.celery_base import celery_app

__all__ = ["celery_app"]
