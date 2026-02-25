#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

cd "${BACKEND_DIR}"
exec uv run celery -A src.workers.celery_app.celery_app beat -l info "$@"
