from __future__ import annotations

import os

import uvicorn

from .api import get_settings, make_app, router_config_from_settings
from .http_backend import HyMtHttpModel, LinguaSparkFastBackend, parse_pairs
from .model import (
    HyMtMlxModel,
    HyMtTransformersModel,
    TranslationModel,
    has_mlx_backend,
    has_transformers_backend,
)
from .router import Router
from .settings import Settings
from .shortlist import Glossary, ShortlistStore


def _build_model(settings: Settings) -> TranslationModel:
    backend = settings.backend.strip().lower()
    if backend == "auto":
        backend = "mlx" if has_mlx_backend() else "transformers"

    if backend == "mlx":
        return HyMtMlxModel(
            settings.model_id,
            max_concurrency=settings.model_max_concurrency,
            prompt_style=settings.prompt_style,
        )
    if backend == "transformers":
        if not has_transformers_backend():
            raise RuntimeError("transformers backend selected but 'torch'/'transformers' not installed")
        return HyMtTransformersModel(
            settings.model_id,
            device=settings.device,
            dtype=settings.dtype,
            max_concurrency=settings.model_max_concurrency,
            prompt_style=settings.prompt_style,
        )
    if backend == "http":
        if not settings.http_base_url:
            raise RuntimeError("BACKEND=http requires MODEL_HTTP_BASE_URL")
        return HyMtHttpModel(
            settings.http_model or settings.model_id,
            base_url=settings.http_base_url,
            api_key=settings.http_api_key,
            timeout_s=settings.http_timeout_s,
            max_concurrency=max(1, settings.model_max_concurrency),
            prompt_style=settings.prompt_style,
        )
    raise RuntimeError("BACKEND must be one of: auto | mlx | transformers | http")


def build_router(settings: Settings, model: TranslationModel) -> Router:
    """Wire the decision layer: shortlist/glossary files + optional fast path."""
    config = router_config_from_settings(settings)

    # The stores always exist, even without a backing file: the memory and
    # glossary are then simply populated at runtime through /shortlist and
    # /glossary. Configuring a path only adds persistence.
    shortlist = (
        ShortlistStore.from_jsonl(settings.shortlist_path) if settings.shortlist_path else ShortlistStore()
    )
    glossary = Glossary.from_jsonl(settings.glossary_path) if settings.glossary_path else Glossary()

    fast_backend = None
    if settings.router_fast_url:
        fast_backend = LinguaSparkFastBackend(
            settings.router_fast_url,
            pairs=parse_pairs(settings.router_fast_pairs),
            timeout_s=settings.router_fast_timeout_s,
            api_key=settings.router_fast_api_key,
        )

    return Router(
        model,
        config=config,
        fast_backend=fast_backend,
        shortlist=shortlist,
        glossary=glossary,
    )


def create_app():
    settings = get_settings()
    model = _build_model(settings)
    router = build_router(settings, model)
    return make_app(model, settings, router)


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
    router = build_router(settings, model)
    app = make_app(model, settings, router)
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )
