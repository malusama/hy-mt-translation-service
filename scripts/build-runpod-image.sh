#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/build-runpod-image.sh job <image-ref> [--push]
  bash scripts/build-runpod-image.sh lb  <image-ref> [--push]

Examples:
  bash scripts/build-runpod-image.sh job ghcr.io/ORG/hy-mt-gguf-llama-rust:runpod-job --push
  bash scripts/build-runpod-image.sh lb  docker.io/USER/hy-mt-gguf-llama-rust:runpod-lb  --push

Notes:
  - `job` builds Traditional Serverless (RunPod Job API): Dockerfile.runpod.serverless
  - `lb`  builds Load Balancing / HTTP Workers (direct HTTP): Dockerfile.runpod.loadbalancing
  - Uses linux/amd64 by default (RunPod).
EOF
}

KIND="${1:-}"
IMAGE_REF="${2:-}"
PUSH="${3:-}"

if [[ -z "${KIND}" || -z "${IMAGE_REF}" ]]; then
  usage
  exit 2
fi

FILE=""
case "${KIND}" in
  job) FILE="Dockerfile.runpod.serverless" ;;
  lb)  FILE="Dockerfile.runpod.loadbalancing" ;;
  *) usage; exit 2 ;;
esac

PLATFORM="${PLATFORM:-linux/amd64}"

ARGS=(docker buildx build --platform "${PLATFORM}" -f "${FILE}" -t "${IMAGE_REF}")
if [[ "${PUSH}" == "--push" ]]; then
  ARGS+=(--push)
else
  # Default to local load for quick verification.
  ARGS+=(--load)
fi
ARGS+=(.)

echo "Building ${KIND} image: ${IMAGE_REF}" >&2
echo "Dockerfile: ${FILE} (platform=${PLATFORM})" >&2
exec "${ARGS[@]}"

