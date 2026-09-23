#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Load env (API_KEY, PORT, MODEL_ID, etc.)
if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "MLX backend is macOS-only. Use scripts/run-rust.sh (GGUF/llama.cpp) or BACKEND=transformers." >&2
  exit 1
fi

if [[ "$(uname -m)" != "arm64" ]]; then
  echo "MLX backend requires Apple Silicon (arm64). Current arch: $(uname -m)" >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install uv first, or run via your own venv." >&2
  echo "macOS (Homebrew): brew install uv" >&2
  exit 1
fi

export BACKEND="${BACKEND:-mlx}"
export MODEL_ID="${MODEL_ID:-mlx-community/Hy-MT2-1.8B-4bit}"
export HOST="${HOST:-127.0.0.1}"
export PORT="${PORT:-3000}"
export UVICORN_WORKERS="${UVICORN_WORKERS:-1}"

# Avoid a long startup / download unless explicitly enabled.
export PRELOAD_MODEL="${PRELOAD_MODEL:-0}"

# uv may be pinned to an incompatible interpreter globally; default to a compatible one.
UV_PYTHON_BIN="${UV_PYTHON:-3.12}"

exec uv run -p "${UV_PYTHON_BIN}" --extra mlx hy-mt-server
