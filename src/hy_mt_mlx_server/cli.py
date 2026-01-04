from __future__ import annotations

import os

import uvicorn

from .api import get_settings, make_app
from .model import (
    HyMtMlxModel,
    HyMtTransformersModel,
    TranslationModel,
    has_mlx_backend,
    has_transformers_backend,
)
from .settings import Settings


def _build_model(settings: Settings) -> TranslationModel:
    backend = settings.backend.strip().lower()
    if backend == "auto":
        backend = "mlx" if has_mlx_backend() else "transformers"

    if backend == "mlx":
        return HyMtMlxModel(settings.model_id, max_concurrency=settings.model_max_concurrency)
    if backend == "transformers":
        if not has_transformers_backend():
            raise RuntimeError("transformers backend selected but 'torch'/'transformers' not installed")
        return HyMtTransformersModel(
            settings.model_id,
            device=settings.device,
            dtype=settings.dtype,
            max_concurrency=settings.model_max_concurrency,
        )
    raise RuntimeError("BACKEND must be one of: auto | mlx | transformers")


def create_app():
    settings = get_settings()
    model = _build_model(settings)
    return make_app(model, settings)


def main() -> None:
    settings = get_settings()
    workers = max(1, int(settings.uvicorn_workers))
    if workers > 1:
        uvicorn.run(
            "hy_mt_mlx_server.cli:create_app",
            factory=True,
            host=settings.host,
            port=settings.port,
            workers=workers,
            log_level=os.getenv("LOG_LEVEL", "info").lower(),
        )
        return

    model = _build_model(settings)
    app = make_app(model, settings)
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )
