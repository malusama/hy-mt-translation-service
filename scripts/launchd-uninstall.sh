#!/usr/bin/env bash
set -euo pipefail

remove_one() {
  local label="$1"
  local plist="$HOME/Library/LaunchAgents/${label}.plist"
  if [[ -f "$plist" ]]; then
    launchctl bootout "gui/$UID" "$plist" >/dev/null 2>&1 || true
    launchctl unload "$plist" >/dev/null 2>&1 || true
    rm -f "$plist"
    echo "Removed: $plist"
  fi
}

# Legacy label(s)
remove_one "com.hy-mt.rust-web"
remove_one "com.hy-mt.mlx.translation-service"

echo "Done."
