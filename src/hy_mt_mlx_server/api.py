from __future__ import annotations

import os
import time
from typing import Any, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from .imme import translate_imme
from .lang import detect_language_code, normalize_lang
from .model import GenerationParams, TranslationModel, has_mlx_backend
from .router import Router, RouterConfig, SegmentResult
from .schemas import (
    DeeplxRequest,
    DeeplxResponse,
    DetectRequest,
    DetectResponse,
    GlossaryAddRequest,
    HcfyRequest,
    HcfyResponse,
    ImmersiveRequest,
    ImmersiveResponse,
    MemoryAddResponse,
    RouteRequest,
    RouteResponse,
    ShortlistAddRequest,
    TranslateRequest,
    TranslateResponse,
)
from .settings import Settings
from .shortlist import (
    GlossaryEntry,
    TranslationMemoryEntry,
    append_jsonl,
)

DEFAULT_MLX_MODEL_ID = "mlx-community/Hy-MT2-1.8B-4bit"
DEFAULT_TRANSFORMERS_MODEL_ID = "tencent/Hy-MT2-1.8B"


def _truthy(value: str) -> bool:
    return value.strip() not in {"0", "false", "False", "no", "NO", ""}


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _imme_batch_mode(value: str) -> str:
    s = value.strip().lower()
    if s in {"0", "false", "no", "off"}:
        return "off"
    if s in {"1", "true", "yes", "on"}:
        return "on"
    return "auto"


def get_settings() -> Settings:
    """Read configuration from the environment.

    pydantic does not auto-load env here, so the mapping stays explicit: one
    line per variable, which is also the documentation.
    """
    backend_raw = (os.getenv("BACKEND", "auto") or "auto").strip().lower()
    model_id_env = (os.getenv("MODEL_ID") or "").strip()
    if model_id_env:
        model_id = model_id_env
    elif backend_raw == "mlx":
        model_id = DEFAULT_MLX_MODEL_ID
    elif backend_raw == "transformers":
        model_id = DEFAULT_TRANSFORMERS_MODEL_ID
    elif backend_raw == "http":
        model_id = (os.getenv("MODEL_HTTP_MODEL") or "").strip()
    else:
        model_id = DEFAULT_MLX_MODEL_ID if has_mlx_backend() else DEFAULT_TRANSFORMERS_MODEL_ID

    max_new_tokens = _int_env("MAX_NEW_TOKENS", 1024)
    max_input_chars_default = min(2000, max_new_tokens)

    return Settings(
        HOST=os.getenv("HOST", "127.0.0.1"),
        PORT=_int_env("PORT", 3000),
        API_KEY=os.getenv("API_KEY", ""),
        MODEL_ID=model_id,
        MAX_NEW_TOKENS=max_new_tokens,
        TEMPERATURE=_float_env("TEMPERATURE", 0.0),
        TOP_P=_float_env("TOP_P", 0.6),
        TOP_K=_int_env("TOP_K", 20),
        REPETITION_PENALTY=_float_env("REPETITION_PENALTY", 1.05),
        PRELOAD_MODEL=_truthy(os.getenv("PRELOAD_MODEL", "1")),
        BACKEND=backend_raw,
        DEVICE=os.getenv("DEVICE", "auto"),
        DTYPE=os.getenv("DTYPE", "auto"),
        MODEL_MAX_CONCURRENCY=_int_env("MODEL_MAX_CONCURRENCY", 1),
        UVICORN_WORKERS=_int_env("UVICORN_WORKERS", _int_env("WORKERS", 1)),
        IMME_BATCH=_imme_batch_mode(os.getenv("IMME_BATCH", "auto")),
        MAX_INPUT_CHARS=_int_env("MAX_INPUT_CHARS", max_input_chars_default),
        IMME_BATCH_SIZE=_int_env("IMME_BATCH_SIZE", 32),
        IMME_MAX_TEXTS=_int_env("IMME_MAX_TEXTS", 1024),
        PROMPT_STYLE=os.getenv("PROMPT_STYLE", "legacy"),
        MODEL_HTTP_BASE_URL=os.getenv("MODEL_HTTP_BASE_URL", ""),
        MODEL_HTTP_MODEL=os.getenv("MODEL_HTTP_MODEL", ""),
        MODEL_HTTP_API_KEY=os.getenv("MODEL_HTTP_API_KEY", ""),
        MODEL_HTTP_TIMEOUT=_float_env("MODEL_HTTP_TIMEOUT", 120.0),
        ROUTER_ENABLED=_truthy(os.getenv("ROUTER_ENABLED", "1")),
        ROUTER_SKIP_SYMBOL_ONLY=_truthy(os.getenv("ROUTER_SKIP_SYMBOL_ONLY", "1")),
        ROUTER_SKIP_SAME_LANGUAGE=_truthy(os.getenv("ROUTER_SKIP_SAME_LANGUAGE", "1")),
        ROUTER_CACHE_SIZE=_int_env("ROUTER_CACHE_SIZE", 4096),
        ROUTER_RETRY=_truthy(os.getenv("ROUTER_RETRY", "1")),
        ROUTER_FAST_URL=os.getenv("ROUTER_FAST_URL", ""),
        ROUTER_FAST_PAIRS=os.getenv("ROUTER_FAST_PAIRS", "en:zh"),
        ROUTER_FAST_MAX_CHARS=_int_env("ROUTER_FAST_MAX_CHARS", 240),
        ROUTER_FAST_TIMEOUT=_float_env("ROUTER_FAST_TIMEOUT", 30.0),
        ROUTER_FAST_API_KEY=os.getenv("ROUTER_FAST_API_KEY", ""),
        SHORTLIST_PATH=os.getenv("SHORTLIST_PATH", ""),
        GLOSSARY_PATH=os.getenv("GLOSSARY_PATH", ""),
        SHORTLIST_TOP_K=_int_env("SHORTLIST_TOP_K", 5),
        SHORTLIST_REUSE=_float_env("SHORTLIST_REUSE", 0.97),
        SHORTLIST_REFERENCE=_float_env("SHORTLIST_REFERENCE", 0.80),
        SHORTLIST_REFERENCE_LIMIT=_int_env("SHORTLIST_REFERENCE_LIMIT", 1),
        SHORTLIST_RERANK=os.getenv("SHORTLIST_RERANK", "off"),
        GLOSSARY_LIMIT=_int_env("GLOSSARY_LIMIT", 8),
    )


def router_config_from_settings(settings: Settings) -> RouterConfig:
    return RouterConfig(
        enabled=settings.router_enabled,
        skip_symbol_only=settings.router_skip_symbol_only,
        skip_same_language=settings.router_skip_same_language,
        cache_size=settings.router_cache_size,
        retry_on_hard_issues=settings.router_retry,
        prompt_style=settings.prompt_style,
        max_input_chars=settings.max_input_chars,
        fast_enabled=bool(settings.router_fast_url),
        fast_max_chars=settings.router_fast_max_chars,
        shortlist_top_k=settings.shortlist_top_k,
        reuse_threshold=settings.shortlist_reuse,
        reference_threshold=settings.shortlist_reference,
        reference_limit=settings.shortlist_reference_limit,
        glossary_limit=settings.glossary_limit,
        rerank=settings.shortlist_rerank,
    )


def make_app(model: TranslationModel, settings: Settings, router: Optional[Router] = None) -> FastAPI:
    app = FastAPI(title="HY-MT2 Translation Service")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if router is None:
        router = Router(model, config=router_config_from_settings(settings))

    async def _auth(
        request: Request,
        authorization: Optional[str] = Header(default=None),
    ) -> None:
        expected = settings.api_key
        if not expected:
            return

        header_key: Optional[str] = None
        if authorization and authorization.startswith("Bearer "):
            header_key = authorization.removeprefix("Bearer ").strip()

        query_key = request.query_params.get("token")

        if header_key != expected and query_key != expected:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    def _params() -> GenerationParams:
        return GenerationParams(
            max_new_tokens=settings.max_new_tokens,
            temperature=settings.temperature,
            top_p=settings.top_p,
            top_k=settings.top_k,
            repetition_penalty=settings.repetition_penalty,
        )

    def _item_payload(result: SegmentResult) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "detected_source_lang": result.source_lang,
            "text": result.text,
        }
        if result.engine:
            payload["engine"] = result.engine
        if result.reason:
            payload["reason"] = result.reason
        if result.issues:
            payload["issues"] = result.issues
        if result.similarity is not None:
            payload["similarity"] = round(result.similarity, 4)
        return payload

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/stats")
    async def stats() -> dict[str, Any]:
        return {
            "backend": settings.backend,
            "model": model.model_id,
            "prompt_style": settings.prompt_style,
            "router": router.stats,
            "fast_path": settings.router_fast_url or None,
        }

    @app.post("/detect", dependencies=[Depends(_auth)], response_model=DetectResponse)
    async def detect(req: DetectRequest) -> DetectResponse:
        return DetectResponse(language=detect_language_code(req.text))

    async def _translate_one(text: str, from_lang: Optional[str], to_lang: str) -> tuple[SegmentResult, str]:
        """Translate one segment through the router.

        ``from_lang`` is the caller's declared source (may be None = auto).
        """
        results = await router.translate_many(
            [text],
            source_lang=from_lang,
            target_lang=to_lang,
            params=_params(),
        )
        result = results[0]
        detected = from_lang if from_lang else detect_language_code(text)
        return result, detected

    @app.post("/translate", dependencies=[Depends(_auth)], response_model=TranslateResponse)
    async def translate(req: TranslateRequest) -> Any:
        from_lang = normalize_lang(req.from_lang)
        to_lang = normalize_lang(req.to) or "en"
        try:
            result, detected = await _translate_one(req.text, from_lang, to_lang)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"translation backend error: {type(e).__name__}: {e}")
        return {
            "text": result.text,
            "from": detected,
            "to": to_lang,
            "engine": result.engine,
            "reason": result.reason or None,
            "issues": result.issues or None,
        }

    @app.post("/kiss", dependencies=[Depends(_auth)], response_model=TranslateResponse)
    async def translate_kiss(req: TranslateRequest) -> Any:
        return await translate(req)

    @app.post("/route", dependencies=[Depends(_auth)], response_model=RouteResponse)
    async def route(req: RouteRequest) -> Any:
        texts = req.text_list if req.text_list else ([req.text] if req.text else [])
        target_lang = normalize_lang(req.target_lang) or "zh"
        source_lang = normalize_lang(req.source_lang)
        return {"routes": router.route_only(texts, source_lang=source_lang, target_lang=target_lang)}

    @app.post("/imme", dependencies=[Depends(_auth)], response_model=ImmersiveResponse)
    async def translate_immersive(req: ImmersiveRequest) -> Any:
        source_lang = normalize_lang(req.source_lang)
        target_lang = normalize_lang(req.target_lang) or "en"

        if not req.text_list:
            return {"translations": []}

        if settings.imme_max_texts > 0 and len(req.text_list) > settings.imme_max_texts:
            raise HTTPException(
                status_code=413,
                detail=f"text_list too large ({len(req.text_list)}); limit is IMME_MAX_TEXTS={settings.imme_max_texts}",
            )

        if not router.config.enabled:
            return await _legacy_imme(req, source_lang, target_lang)

        try:
            results = await router.translate_many(
                [str(t) for t in req.text_list],
                source_lang=source_lang,
                target_lang=target_lang,
                params=_params(),
            )
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"translation backend error: {type(e).__name__}: {e}")
        return {"translations": [_item_payload(r) for r in results]}

    async def _legacy_imme(req: ImmersiveRequest, source_lang: Optional[str], target_lang: str) -> Any:
        try:
            out = await translate_imme(
                model,
                text_list=req.text_list,
                target_lang=target_lang,
                source_lang=source_lang,
                params=_params(),
                max_input_chars=settings.max_input_chars,
                batch_mode=settings.imme_batch,
                batch_size=settings.imme_batch_size,
            )
            return {"translations": out}
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"translation backend error: {type(e).__name__}: {e}")

    # HCFY compatibility (very small mapping, same as LinguaSpark/server)
    HCFY_LANGUAGE_CODE_MAP = [("中文(简体)", "zh"), ("英语", "en"), ("日语", "jp")]

    def _hcfy_to_code(name: str) -> str:
        for n, c in HCFY_LANGUAGE_CODE_MAP:
            if n == name:
                return c
        return name

    def _hcfy_to_name(code: str) -> str:
        for n, c in HCFY_LANGUAGE_CODE_MAP:
            if c == code:
                return n
        return code

    @app.post("/hcfy", dependencies=[Depends(_auth)], response_model=HcfyResponse)
    async def translate_hcfy(req: HcfyRequest) -> Any:
        source_code = _hcfy_to_code(req.source) if req.source else None
        source_code = normalize_lang(source_code)

        if not req.destination:
            target_code = "en"
        elif source_code and len(req.destination) > 1 and normalize_lang(_hcfy_to_code(req.destination[0])) == source_code:
            target_code = _hcfy_to_code(req.destination[1])
        else:
            target_code = _hcfy_to_code(req.destination[0])

        target_code = normalize_lang(target_code) or "en"
        result, detected = await _translate_one(req.text, source_code, target_code)
        return {
            "text": req.text,
            "from": _hcfy_to_name(detected),
            "to": _hcfy_to_name(target_code),
            "result": [result.text],
        }

    @app.post("/deeplx", dependencies=[Depends(_auth)], response_model=DeeplxResponse)
    async def translate_deeplx(req: DeeplxRequest) -> Any:
        from_lang = normalize_lang(req.source_lang) or "en"
        to_lang = normalize_lang(req.target_lang) or "zh"
        result, detected = await _translate_one(req.text, from_lang, to_lang)
        return {
            "code": 200,
            "id": int(time.time() * 1000),
            "data": result.text,
            "alternatives": [],
            "source_lang": detected.upper(),
            "target_lang": to_lang.upper(),
            "method": "Free",
        }

    @app.post("/shortlist", dependencies=[Depends(_auth)], response_model=MemoryAddResponse)
    async def add_shortlist(req: ShortlistAddRequest) -> Any:
        entries = [
            TranslationMemoryEntry(
                source=e.source.strip(),
                target=e.target.strip(),
                source_lang=(e.source_lang or "").strip(),
                target_lang=(e.target_lang or "").strip(),
            )
            for e in req.entries
        ]
        added = 0
        if router.shortlist is not None:
            before = len(router.shortlist)
            router.shortlist.add_many(entries)
            added = len(router.shortlist) - before
        persisted = bool(settings.shortlist_path)
        if persisted:
            for entry in entries:
                append_jsonl(
                    settings.shortlist_path,
                    {
                        "source": entry.source,
                        "target": entry.target,
                        "source_lang": entry.source_lang,
                        "target_lang": entry.target_lang,
                    },
                )
        return {
            "added": added,
            "shortlist_entries": len(router.shortlist) if router.shortlist else 0,
            "glossary_entries": len(router.glossary) if router.glossary else 0,
            "persisted": persisted,
        }

    @app.post("/glossary", dependencies=[Depends(_auth)], response_model=MemoryAddResponse)
    async def add_glossary(req: GlossaryAddRequest) -> Any:
        terms = [
            GlossaryEntry(
                source_term=t.source_term.strip(),
                target_term=t.target_term.strip(),
                source_lang=(t.source_lang or "").strip(),
                target_lang=(t.target_lang or "").strip(),
            )
            for t in req.terms
        ]
        added = 0
        if router.glossary is not None:
            before = len(router.glossary)
            router.glossary.add_many(terms)
            added = len(router.glossary) - before
        persisted = bool(settings.glossary_path)
        if persisted:
            for term in terms:
                append_jsonl(
                    settings.glossary_path,
                    {
                        "source_term": term.source_term,
                        "target_term": term.target_term,
                        "source_lang": term.source_lang,
                        "target_lang": term.target_lang,
                    },
                )
        return {
            "added": added,
            "shortlist_entries": len(router.shortlist) if router.shortlist else 0,
            "glossary_entries": len(router.glossary) if router.glossary else 0,
            "persisted": persisted,
        }

    @app.on_event("startup")
    async def _startup() -> None:
        if settings.preload_model:
            await model.ensure_loaded()

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        model.close()
        router.close()

    return app
