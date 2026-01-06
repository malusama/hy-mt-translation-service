#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="com.hy-mt.mlx.translation-service"
PLIST_PATH="$HOME/Library/LaunchAgents/${LABEL}.plist"
LOG_OUT="$HOME/Library/Logs/hy-mt-mlx-translation-service.out.log"
LOG_ERR="$HOME/Library/Logs/hy-mt-mlx-translation-service.err.log"

mkdir -p "$HOME/Library/LaunchAgents" "$HOME/Library/Logs"

# Load .env if present (so LLAMA_FEATURES can be set there).
if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

MODEL_PATH="${MODEL_GGUF:-${MODEL_PATH:-${MODEL_ID:-}}}"
if [[ -z "${MODEL_PATH}" ]]; then
  echo "Missing MODEL_GGUF in .env (or MODEL_PATH/MODEL_ID) pointing to a local .gguf file." >&2
  exit 1
fi
if [[ "${MODEL_PATH}" != /* ]]; then
  MODEL_PATH="${ROOT}/${MODEL_PATH}"
fi
if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "Model file not found: ${MODEL_PATH}" >&2
  echo "Set MODEL_GGUF to a real .gguf path, then re-run: bash scripts/launchd-install.sh" >&2
  exit 1
fi

# Build once so launchd doesn't need Cargo at startup.
FEATURES="${LLAMA_FEATURES:-}"
if [[ -z "${FEATURES}" ]]; then
  if [[ "$(uname -s)" == "Darwin" ]]; then
    FEATURES="metal"
  fi
fi
FEATURE_ARGS=()
if [[ -n "${FEATURES}" ]]; then
  FEATURE_ARGS=(--features "${FEATURES}")
fi
cargo build --release --manifest-path "$ROOT/rust-web/Cargo.toml"

cat >"$PLIST_PATH" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>${LABEL}</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key>
  <dict>
    <!-- Avoid restart loops when model path is missing; keepalive when it exists. -->
    <key>SuccessfulExit</key><false/>
    <key>PathState</key>
    <dict>
      <key>${MODEL_PATH}</key><true/>
    </dict>
  </dict>
  <key>WorkingDirectory</key><string>${ROOT}</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>${ROOT}/scripts/run-rust.sh</string>
    <string>--no-build</string>
  </array>
  <key>StandardOutPath</key><string>${LOG_OUT}</string>
  <key>StandardErrorPath</key><string>${LOG_ERR}</string>
</dict>
</plist>
EOF

# macOS 13+ recommended style (bootstrap into GUI domain); fall back to legacy load.
launchctl bootout "gui/$UID" "$PLIST_PATH" >/dev/null 2>&1 || true
if launchctl bootstrap "gui/$UID" "$PLIST_PATH" >/dev/null 2>&1; then
  launchctl enable "gui/$UID/${LABEL}" >/dev/null 2>&1 || true
  launchctl kickstart -k "gui/$UID/${LABEL}" >/dev/null 2>&1 || true
else
  launchctl unload "$PLIST_PATH" >/dev/null 2>&1 || true
  launchctl load "$PLIST_PATH"
fi

echo "Installed launchd agent: $PLIST_PATH"
echo "Logs: $LOG_OUT / $LOG_ERR"
