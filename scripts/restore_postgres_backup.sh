#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/restore_postgres_backup.sh /path/to/backup.dump

Environment variables:
  POSTGRES_HOST      default: localhost
  POSTGRES_PORT      default: 5432
  POSTGRES_USER      default: postgres
  POSTGRES_PASSWORD  default: (empty)
  POSTGRES_DB        default: minsy_pgsql
  POSTGRES_PG_RESTORE_BIN default: pg_restore
EOF
}

BACKUP_FILE="${1:-}"
if [[ -z "${BACKUP_FILE}" ]]; then
  usage
  exit 1
fi

if [[ ! -f "${BACKUP_FILE}" ]]; then
  echo "Backup file not found: ${BACKUP_FILE}" >&2
  exit 1
fi

POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_USER="${POSTGRES_USER:-postgres}"
POSTGRES_DB="${POSTGRES_DB:-minsy_pgsql}"
POSTGRES_PG_RESTORE_BIN="${POSTGRES_PG_RESTORE_BIN:-pg_restore}"

if ! command -v "${POSTGRES_PG_RESTORE_BIN}" >/dev/null 2>&1; then
  echo "Command not found: ${POSTGRES_PG_RESTORE_BIN}" >&2
  exit 1
fi

if [[ -n "${POSTGRES_PASSWORD:-}" ]]; then
  export PGPASSWORD="${POSTGRES_PASSWORD}"
fi

echo "Restoring '${BACKUP_FILE}' into ${POSTGRES_DB}@${POSTGRES_HOST}:${POSTGRES_PORT} ..."
"${POSTGRES_PG_RESTORE_BIN}" \
  --clean \
  --if-exists \
  --no-owner \
  --no-privileges \
  --host "${POSTGRES_HOST}" \
  --port "${POSTGRES_PORT}" \
  --username "${POSTGRES_USER}" \
  --dbname "${POSTGRES_DB}" \
  "${BACKUP_FILE}"

echo "Restore completed."
