from __future__ import annotations

import argparse
import importlib
from unittest.mock import patch

import src.main as main_module
import src.mcp.server as mcp_server_module


def test_fastapi_create_app_bootstraps_sentry() -> None:
    with patch("src.main.init_backend_sentry") as mocked_init:
        main_module.create_app()

    mocked_init.assert_called_once_with(source="fastapi")


def test_celery_module_bootstraps_sentry_on_import(monkeypatch) -> None:
    import src.observability.sentry_setup as sentry_setup_module
    celery_module = importlib.import_module("src.workers.celery_app")

    calls: list[str] = []
    monkeypatch.setattr(
        sentry_setup_module,
        "init_backend_sentry",
        lambda *, source: calls.append(source) or False,
    )

    importlib.reload(celery_module)

    assert "celery" in calls


def test_mcp_main_bootstraps_sentry() -> None:
    class _FakeMcpServer:
        def run(self, transport: str) -> None:
            self.transport = transport

    fake_args = argparse.Namespace(
        domain="strategy",
        transport="stdio",
        host="127.0.0.1",
        port=8111,
        mount_path="/",
        stateful_http=False,
    )
    fake_server = _FakeMcpServer()

    with (
        patch("src.mcp.server.init_backend_sentry") as mocked_init,
        patch("src.mcp.server._build_parser") as mocked_parser,
        patch("src.mcp.server.create_mcp_server", return_value=fake_server),
        patch("src.mcp.server.registered_tool_names", return_value=("tool_a",)),
    ):
        mocked_parser.return_value.parse_args.return_value = fake_args
        exit_code = mcp_server_module.main()

    assert exit_code == 0
    mocked_init.assert_called_once_with(source="mcp")
