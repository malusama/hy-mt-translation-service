#!/usr/bin/env bash
set -euo pipefail

export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH}"

MODEL_PATH="${MODEL_GGUF:-}"
MODEL_URL="${MODEL_GGUF_URL:-https://huggingface.co/tencent/HY-MT1.5-1.8B-GGUF/resolve/main/HY-MT1.5-1.8B-Q4_K_M.gguf}"
MODEL_DIR="${MODEL_DIR:-/runpod-volume/models}"

if [[ -z "${MODEL_PATH}" ]]; then
  mkdir -p "${MODEL_DIR}"
  filename="$(basename "${MODEL_URL}")"
  MODEL_PATH="${MODEL_DIR}/${filename}"
fi

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "Downloading GGUF model to: ${MODEL_PATH}" >&2
  mkdir -p "$(dirname "${MODEL_PATH}")"
  auth=()
  if [[ -n "${HF_TOKEN:-}" ]]; then
    auth=(-H "Authorization: Bearer ${HF_TOKEN}")
  elif [[ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
    auth=(-H "Authorization: Bearer ${HUGGING_FACE_HUB_TOKEN}")
  fi
  curl -fL --retry 5 --retry-delay 2 -o "${MODEL_PATH}.tmp" "${auth[@]}" "${MODEL_URL}"
  mv "${MODEL_PATH}.tmp" "${MODEL_PATH}"
fi

LLAMA_HOST="${LLAMA_SERVER_HOST:-127.0.0.1}"
LLAMA_PORT="${LLAMA_SERVER_PORT:-18080}"
export LLAMA_SERVER_URL="${LLAMA_SERVER_URL:-http://${LLAMA_HOST}:${LLAMA_PORT}}"

start_llama() {
  local ngl="${LLAMA_N_GPU_LAYERS:-999}"
  local threads="${LLAMA_THREADS:-0}"
  local ctx="${LLAMA_N_CTX:-0}"
  local batch="${LLAMA_N_BATCH:-0}"

  args=(--host "${LLAMA_HOST}" --port "${LLAMA_PORT}" -m "${MODEL_PATH}" -ngl "${ngl}")
  if [[ "${threads}" != "0" ]]; then args+=(--threads "${threads}"); fi
  if [[ "${ctx}" != "0" ]]; then args+=(--ctx-size "${ctx}"); fi
  if [[ "${batch}" != "0" ]]; then args+=(--batch-size "${batch}"); fi

  echo "Starting llama-server: llama-server ${args[*]}" >&2
  llama-server "${args[@]}" >/var/log/llama-server.log 2>&1 &
  echo $! >/tmp/llama-server.pid
}

wait_http_ok() {
  local url="$1"
  local deadline=$((SECONDS + 600))
  while (( SECONDS < deadline )); do
    if curl -fsS --max-time 2 "${url}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

if ! curl -fsS --max-time 2 "${LLAMA_SERVER_URL}/health" >/dev/null 2>&1; then
  start_llama
  if ! wait_http_ok "${LLAMA_SERVER_URL}/health"; then
    echo "llama-server failed to become ready; tailing logs:" >&2
    tail -n 200 /var/log/llama-server.log >&2 || true
    exit 1
  fi
fi

export HOST="${HOST:-127.0.0.1}"
export PORT="${PORT:-3000}"
export OPENAI_MODEL_NAME="${OPENAI_MODEL_NAME:-$(basename "${MODEL_PATH}")}"

echo "Starting Rust web: /usr/local/bin/hy-mt-rust-web (${HOST}:${PORT})" >&2
/usr/local/bin/hy-mt-rust-web >/var/log/hy-mt-rust-web.log 2>&1 &
echo $! >/tmp/hy-mt-rust-web.pid

if ! wait_http_ok "http://${HOST}:${PORT}/health"; then
  echo "Rust web failed to become ready; tailing logs:" >&2
  tail -n 200 /var/log/hy-mt-rust-web.log >&2 || true
  exit 1
fi

cleanup() {
  if [[ -f /tmp/hy-mt-rust-web.pid ]]; then
    kill "$(cat /tmp/hy-mt-rust-web.pid)" >/dev/null 2>&1 || true
  fi
  if [[ -f /tmp/llama-server.pid ]]; then
    kill "$(cat /tmp/llama-server.pid)" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

exec python3.11 -u /src/handler.py
