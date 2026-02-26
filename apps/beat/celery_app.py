"""Celery beat scheduler entrypoint."""

from __future__ import annotations

import os

# Entry-point scoped profile selection for scheduler startup.
os.environ["MINSY_CELERY_PROFILE"] = "beat"
os.environ.setdefault("MINSY_SERVICE", "beat")

from packages.infra.queue.celery_app import celery_app

__all__ = ["celery_app"]
