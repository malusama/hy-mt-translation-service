#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# launchd's default PATH does not include Homebrew.
export PATH="/opt/homebrew/bin:/usr/local/bin:${PATH}"

NO_BUILD=0
for arg in "$@"; do
  case "$arg" in
    --no-build) NO_BUILD=1 ;;
  esac
done

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

MODEL_PATH="${MODEL_GGUF:-${MODEL_PATH:-${MODEL_ID:-}}}"
if [[ -z "${MODEL_PATH}" ]]; then
  echo "Missing model path. Set MODEL_GGUF to a local .gguf file path (recommended)." >&2
  echo "Example: export MODEL_GGUF=\"$ROOT/model/HY-MT1.5-1.8B-Q4_K_M.gguf\"" >&2
  exit 1
fi
if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "Model file not found: ${MODEL_PATH}" >&2
  exit 1
fi

if ! command -v llama-server >/dev/null 2>&1; then
  echo "llama-server not found. Install with: brew install llama.cpp" >&2
  exit 1
fi

LLAMA_PORT="${LLAMA_SERVER_PORT:-18080}"
LLAMA_HOST="${LLAMA_SERVER_HOST:-127.0.0.1}"
export LLAMA_SERVER_URL="${LLAMA_SERVER_URL:-http://${LLAMA_HOST}:${LLAMA_PORT}}"

start_llama_server() {
  mkdir -p "$ROOT/.logs"
  local log="$ROOT/.logs/llama-server.log"
  local pidfile="$ROOT/.logs/llama-server.pid"

  # If already healthy, do nothing.
  if curl -fsS --max-time 2 "${LLAMA_SERVER_URL}/health" >/dev/null 2>&1; then
    return 0
  fi

  # Best-effort kill stale pid.
  if [[ -f "$pidfile" ]]; then
    local pid
    pid="$(cat "$pidfile" 2>/dev/null || true)"
    if [[ -n "$pid" ]]; then
      kill "$pid" >/dev/null 2>&1 || true
    fi
    rm -f "$pidfile"
  fi

  local ngl="${LLAMA_N_GPU_LAYERS:-999}"
  nohup llama-server --host "$LLAMA_HOST" --port "$LLAMA_PORT" -m "$MODEL_PATH" -ngl "$ngl" >>"$log" 2>&1 &
  echo $! >"$pidfile"

  # Wait for readiness (model load can take a while).
  local deadline=$((SECONDS + 600))
  while (( SECONDS < deadline )); do
    if curl -fsS --max-time 2 "${LLAMA_SERVER_URL}/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done

  echo "llama-server failed to become ready (see $log)" >&2
  return 1
}

start_llama_server

BIN="$ROOT/rust-web/target/release/hy-mt-rust-web"
if [[ "$NO_BUILD" -ne 1 ]]; then
  cargo build --release --manifest-path "$ROOT/rust-web/Cargo.toml"
fi

if [[ ! -x "$BIN" ]]; then
  echo "Rust binary not found at: $BIN" >&2
  echo "Run: cargo build --release --manifest-path rust-web/Cargo.toml" >&2
  exit 1
fi

exec "$BIN"
