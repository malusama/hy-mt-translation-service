from __future__ import annotations

import os
from typing import Optional


RUNPOD_HF_CACHE_DIR = "/runpod-volume/huggingface-cache/hub"


def _find_cached_snapshot_dir(model_name: str) -> Optional[str]:
    """
    Locate a Hugging Face cached model directory on Runpod.

    Runpod cached models are stored under:
      /runpod-volume/huggingface-cache/hub/models--ORG--NAME/snapshots/HASH/
    """
    cache_name = model_name.replace("/", "--")
    snapshots_dir = os.path.join(RUNPOD_HF_CACHE_DIR, f"models--{cache_name}", "snapshots")
    if not os.path.isdir(snapshots_dir):
        return None

    snapshots = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
    if not snapshots:
        return None

    # Use the latest-ish snapshot deterministically.
    snapshots.sort()
    return os.path.join(snapshots_dir, snapshots[-1])


def maybe_resolve_runpod_cached_model_path(model_id_or_path: str) -> str:
    """
    If running on Runpod with cached models enabled, resolve HF repo ID to the cached snapshot path.
    Otherwise, return the input unchanged.
    """
    if os.path.exists(model_id_or_path):
        return model_id_or_path

    if "/" not in model_id_or_path:
        return model_id_or_path

    if not os.path.isdir(RUNPOD_HF_CACHE_DIR):
        return model_id_or_path

    resolved = _find_cached_snapshot_dir(model_id_or_path)
    return resolved or model_id_or_path

