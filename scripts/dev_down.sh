#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
RUN_DIR="${BACKEND_DIR}/.run"
COMPOSE_FILE="${BACKEND_DIR}/compose.dev.yml"

PROFILE="${MINSY_PROFILE:-dev}"
ENV_FILE="${MINSY_ENV_FILE:-${BACKEND_DIR}/.env.${PROFILE}}"
KEEP_INFRA=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --keep-infra)
      KEEP_INFRA=true
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -f "${ENV_FILE}" ]]; then
  export MINSY_ENV_FILE="${ENV_FILE}"
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

stop_process() {
  local name="$1"
  local pid_file="${RUN_DIR}/${name}.pid"

  if [[ ! -f "${pid_file}" ]]; then
    return
  fi

  local pid
  pid="$(cat "${pid_file}")"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" >/dev/null 2>&1; then
    kill "${pid}" >/dev/null 2>&1 || true
    sleep 1
    if kill -0 "${pid}" >/dev/null 2>&1; then
      kill -9 "${pid}" >/dev/null 2>&1 || true
    fi
    echo "[stop] ${name} (pid=${pid})"
  fi

  rm -f "${pid_file}"
}

services=(
  api
  mcp_proxy
  mcp_strategy
  mcp_backtest
  mcp_market
  mcp_stress
  mcp_trading
  worker_backtest
  worker_runtime
  worker_notifications
  worker_maintenance
  beat
  flower
)

for service in "${services[@]}"; do
  stop_process "${service}"
done

if [[ "${KEEP_INFRA}" == "false" ]]; then
  if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
    docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" down --remove-orphans >/dev/null || true
    echo "[stop] docker infra (postgres + redis)"
  fi
fi

echo "Done."
