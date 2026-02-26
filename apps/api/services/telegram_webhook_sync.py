"""Compatibility module forwarding to packages domain telegram webhook sync."""

from importlib import import_module as _import_module
import sys as _sys

_sys.modules[__name__] = _import_module("packages.domain.user.services.telegram_webhook_sync")
