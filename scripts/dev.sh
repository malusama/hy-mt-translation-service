#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

backend="${BACKEND:-}"
backend="$(echo "${backend}" | tr '[:upper:]' '[:lower:]' | xargs || true)"

# Local dev on macOS defaults to MLX unless explicitly opting into GGUF/llama.cpp.
if [[ "${backend}" == "mlx" || "${backend}" == "transformers" || "${backend}" == "auto" || -z "${backend}" ]]; then
  if [[ "$(uname -s)" == "Darwin" ]]; then
    exec bash "$ROOT/scripts/run-mlx.sh"
  fi
fi

exec bash "$ROOT/scripts/run-rust.sh"
