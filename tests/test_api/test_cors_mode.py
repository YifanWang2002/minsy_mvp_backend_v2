from fastapi.middleware.cors import CORSMiddleware

from src.main import create_app, settings


def _cors_kwargs() -> dict:
    app = create_app()
    for middleware in app.user_middleware:
        if middleware.cls is CORSMiddleware:
            return middleware.kwargs
    raise AssertionError("CORSMiddleware should always be configured.")


def test_dev_mode_relaxes_cors(monkeypatch) -> None:
    monkeypatch.setattr(settings, "app_env", "dev")
    kwargs = _cors_kwargs()

    assert kwargs.get("allow_origin_regex") == ".*"
    assert "allow_origins" not in kwargs


def test_prod_mode_uses_configured_cors_origins(monkeypatch) -> None:
    monkeypatch.setattr(settings, "app_env", "prod")
    monkeypatch.setattr(settings, "cors_origins", ["https://app.minsyai.com"])
    kwargs = _cors_kwargs()

    assert kwargs.get("allow_origins") == ["https://app.minsyai.com"]
    assert "allow_origin_regex" not in kwargs
