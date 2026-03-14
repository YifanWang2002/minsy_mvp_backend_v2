"""Google Cloud Storage client wrapper."""

from __future__ import annotations

from pathlib import Path


class GcsClient:
    """Thin adapter around google-cloud-storage."""

    def __init__(self) -> None:
        try:
            from google.cloud import storage  # type: ignore[import-not-found]
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("google-cloud-storage package is not installed") from exc
        self._client = storage.Client()

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
        blob.upload_from_string(payload, content_type=content_type)

    def download_bytes(
        self,
        *,
        bucket_name: str,
        object_name: str,
    ) -> bytes:
        blob = self._client.bucket(bucket_name).blob(object_name)
        return bytes(blob.download_as_bytes())
