"""Compatibility module forwarding to packages.infra.auth.mcp_context."""

from importlib import import_module as _import_module
import sys as _sys

_sys.modules[__name__] = _import_module("packages.infra.auth.mcp_context")

