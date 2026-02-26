"""Compatibility module forwarding to packages domain openai stream service."""

from importlib import import_module as _import_module
import sys as _sys

_sys.modules[__name__] = _import_module("packages.domain.session.services.openai_stream_service")
