"""Celery worker package."""

from src.workers.celery_app import celery_app

__all__ = ["celery_app"]
