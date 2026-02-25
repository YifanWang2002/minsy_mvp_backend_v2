#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
RUN_DIR="${BACKEND_DIR}/.run"
LOG_DIR="${BACKEND_DIR}/logs/dev"
COMPOSE_FILE="${BACKEND_DIR}/compose.dev.yml"

PROFILE="${MINSY_PROFILE:-dev}"
ENV_FILE="${MINSY_ENV_FILE:-${BACKEND_DIR}/.env.${PROFILE}}"

if [[ ! -f "${ENV_FILE}" ]]; then
  "${SCRIPT_DIR}/render_env.sh" --profile "${PROFILE}" --output "${ENV_FILE}"
fi

export MINSY_ENV_FILE="${ENV_FILE}"
# Export profile variables so direct `os.getenv(...)` paths also see them.
set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a
mkdir -p "${RUN_DIR}" "${LOG_DIR}"

require_cmd() {
  local cmd="$1"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "Missing command: ${cmd}" >&2
    exit 1
  fi
}

require_cmd uv
require_cmd docker
require_cmd curl

if ! docker compose version >/dev/null 2>&1; then
  echo "docker compose is required." >&2
  exit 1
fi

start_process() {
  local name="$1"
  shift

  local pid_file="${RUN_DIR}/${name}.pid"
  local log_file="${LOG_DIR}/${name}.log"

  if [[ -f "${pid_file}" ]]; then
    local pid
    pid="$(cat "${pid_file}")"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" >/dev/null 2>&1; then
      echo "[skip] ${name} already running (pid=${pid})"
      return
    fi
    rm -f "${pid_file}"
  fi

  (
    cd "${BACKEND_DIR}"
    nohup env MINSY_ENV_FILE="${MINSY_ENV_FILE}" "$@" >"${log_file}" 2>&1 &
    echo "$!" > "${pid_file}"
  )
  echo "[start] ${name} (pid=$(cat "${pid_file}"))"
}

wait_http() {
  local name="$1"
  local url="$2"
  local timeout_seconds="$3"
  local elapsed=0

  while [[ "${elapsed}" -lt "${timeout_seconds}" ]]; do
    if curl -sS -o /dev/null "${url}"; then
      echo "[ok] ${name} reachable: ${url}"
      return 0
    fi
    sleep 1
    elapsed=$((elapsed + 1))
  done

  echo "[warn] ${name} not reachable after ${timeout_seconds}s: ${url}" >&2
  return 1
}

read_env_bool() {
  local key="$1"
  local value
  value="$(awk -F= -v k="${key}" '$1 == k {print tolower($2)}' "${ENV_FILE}" | tail -n 1 | tr -d '[:space:]')"
  [[ "${value}" == "1" || "${value}" == "true" || "${value}" == "yes" || "${value}" == "on" ]]
}

echo "[step] sync python dependencies"
(
  cd "${BACKEND_DIR}"
  uv sync --frozen >/dev/null
)

echo "[step] start infra (postgres + redis)"
docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" up -d postgres redis >/dev/null

# MCP domain servers
start_process mcp_strategy uv run python -m src.mcp.server --domain strategy --transport streamable-http --host 127.0.0.1 --port 8111
start_process mcp_backtest uv run python -m src.mcp.server --domain backtest --transport streamable-http --host 127.0.0.1 --port 8112
start_process mcp_market uv run python -m src.mcp.server --domain market --transport streamable-http --host 127.0.0.1 --port 8113
start_process mcp_stress uv run python -m src.mcp.server --domain stress --transport streamable-http --host 127.0.0.1 --port 8114
start_process mcp_trading uv run python -m src.mcp.server --domain trading --transport streamable-http --host 127.0.0.1 --port 8115

# Optional dev reverse proxy for domain-prefixed MCP URLs
start_process mcp_proxy uv run python -m src.mcp.dev_proxy --host 127.0.0.1 --port 8110

# API
start_process api uv run uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

# Celery workers (split by workload)
start_process worker_backtest uv run celery -A src.workers.celery_app.celery_app worker -l info -Q backtest --hostname=backtest@%h --concurrency="${BACKTEST_WORKER_CONCURRENCY:-1}"
start_process worker_runtime uv run celery -A src.workers.celery_app.celery_app worker -l info -Q paper_trading,market_data,trade_approval --hostname=runtime@%h --concurrency="${RUNTIME_WORKER_CONCURRENCY:-4}"
start_process worker_notifications uv run celery -A src.workers.celery_app.celery_app worker -l info -Q notifications --hostname=notifications@%h --concurrency="${NOTIFICATIONS_WORKER_CONCURRENCY:-1}"
start_process worker_maintenance uv run celery -A src.workers.celery_app.celery_app worker -l info -Q maintenance --hostname=maintenance@%h --concurrency="${MAINTENANCE_WORKER_CONCURRENCY:-1}"
start_process beat uv run celery -A src.workers.celery_app.celery_app beat -l info

if read_env_bool FLOWER_ENABLED; then
  start_process flower uv run celery -A src.workers.celery_app.celery_app flower --address="${FLOWER_HOST:-127.0.0.1}" --port="${FLOWER_PORT:-5555}"
fi

wait_http api "http://127.0.0.1:8000/api/v1/health" 60 || true
wait_http mcp_proxy "http://127.0.0.1:8110/strategy/mcp" 30 || true

echo "\nAll dev services requested."
echo "Env profile: ${ENV_FILE}"
echo "Logs: ${LOG_DIR}"
echo "Stop command: ./scripts/dev_down.sh"
