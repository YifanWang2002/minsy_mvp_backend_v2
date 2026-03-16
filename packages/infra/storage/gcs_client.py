"""Google Cloud Storage client wrapper."""

from __future__ import annotations

from pathlib import Path

from packages.shared_settings.schema.settings import settings


class GcsClient:
    """Thin adapter around google-cloud-storage."""

    _DEFAULT_TIMEOUT_SECONDS = 300

    def __init__(self) -> None:
        try:
            from google.cloud import storage  # type: ignore[import-not-found]
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("google-cloud-storage package is not installed") from exc
        credentials_path = settings.google_application_credentials.strip()
        project = settings.google_cloud_project.strip() or settings.gcp_project.strip() or None
        if credentials_path:
            resolved = Path(credentials_path).expanduser().resolve()
            if not resolved.is_file():
                raise RuntimeError(
                    "GOOGLE_APPLICATION_CREDENTIALS file does not exist: "
                    f"{resolved}"
                )
            self._client = storage.Client.from_service_account_json(
                str(resolved),
                project=project,
            )
            return
        self._client = storage.Client(project=project)

    def upload_file(
        self,
        *,
        local_path: str | Path,
        bucket_name: str,
        object_name: str,
        content_type: str | None = None,
    ) -> None:
        path = Path(local_path).expanduser().resolve()
        blob = self._client.bucket(bucket_name).blob(object_name)
        blob.upload_from_filename(
            str(path),
            content_type=content_type,
            timeout=self._DEFAULT_TIMEOUT_SECONDS,
        )

    def upload_bytes(
        self,
        *,
        payload: bytes,
        bucket_name: str,
        object_name: str,
        content_type: str | None = None,
    ) -> None:
        blob = self._client.bucket(bucket_name).blob(object_name)
        blob.upload_from_string(
            payload,
            content_type=content_type,
            timeout=self._DEFAULT_TIMEOUT_SECONDS,
        )

    def download_bytes(
        self,
        *,
        bucket_name: str,
        object_name: str,
    ) -> bytes:
        blob = self._client.bucket(bucket_name).blob(object_name)
        return bytes(blob.download_as_bytes(timeout=self._DEFAULT_TIMEOUT_SECONDS))
