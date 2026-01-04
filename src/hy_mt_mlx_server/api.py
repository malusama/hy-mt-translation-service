from __future__ import annotations

import os
import time
from typing import Any, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from .schemas import (
    DeeplxRequest,
    DeeplxResponse,
    DetectRequest,
    DetectResponse,
    HcfyRequest,
    HcfyResponse,
    ImmersiveRequest,
    ImmersiveResponse,
    TranslateRequest,
    TranslateResponse,
)

from .lang import detect_language_code, normalize_lang
from .model import GenerationParams, TranslationModel
from .settings import Settings


def get_settings() -> Settings:
    # pydantic BaseModel doesn't auto-load env; do it manually for explicitness
    import os

    from .model import has_mlx_backend

    def _truthy(v: str) -> bool:
        return v.strip() not in {"0", "false", "False", "no", "NO", ""}

    def _float(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, str(default)))
        except Exception:
            return default

    def _int(name: str, default: int) -> int:
        try:
            return int(os.getenv(name, str(default)))
        except Exception:
            return default

    def _imme_batch(v: str) -> str:
        s = v.strip().lower()
        if s in {"0", "false", "no", "off"}:
            return "off"
        if s in {"1", "true", "yes", "on"}:
            return "on"
        return "auto"

    backend_raw = (os.getenv("BACKEND", "auto") or "auto").strip().lower()
    model_id_env = (os.getenv("MODEL_ID") or "").strip()
    default_mlx_model_id = "m-i/HY-MT1.5-1.8B-mlx-8Bit"
    default_transformers_model_id = "tencent/HY-MT1.5-1.8B"
    if model_id_env:
        model_id = model_id_env
    elif backend_raw == "mlx":
        model_id = default_mlx_model_id
    elif backend_raw == "transformers":
        model_id = default_transformers_model_id
    else:
        model_id = default_mlx_model_id if has_mlx_backend() else default_transformers_model_id

    return Settings(
        HOST=os.getenv("HOST", "127.0.0.1"),
        PORT=_int("PORT", 3000),
        API_KEY=os.getenv("API_KEY", ""),
        MODEL_ID=model_id,
        MAX_NEW_TOKENS=_int("MAX_NEW_TOKENS", 1024),
        TEMPERATURE=_float("TEMPERATURE", 0.0),
        TOP_P=_float("TOP_P", 0.6),
        TOP_K=_int("TOP_K", 20),
        REPETITION_PENALTY=_float("REPETITION_PENALTY", 1.05),
        PRELOAD_MODEL=_truthy(os.getenv("PRELOAD_MODEL", "1")),
        BACKEND=os.getenv("BACKEND", "auto"),
        DEVICE=os.getenv("DEVICE", "auto"),
        DTYPE=os.getenv("DTYPE", "auto"),
        MODEL_MAX_CONCURRENCY=_int("MODEL_MAX_CONCURRENCY", 1),
        UVICORN_WORKERS=_int("UVICORN_WORKERS", _int("WORKERS", 1)),
        IMME_BATCH=_imme_batch(os.getenv("IMME_BATCH", "auto")),
    )


def make_app(model: TranslationModel, settings: Settings) -> FastAPI:
    app = FastAPI(title="HY-MT MLX Translation Service")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

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

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/detect", dependencies=[Depends(_auth)], response_model=DetectResponse)
    async def detect(req: DetectRequest) -> DetectResponse:
        return DetectResponse(language=detect_language_code(req.text))

    async def _safe_translate(text: str, to_lang: str, params: GenerationParams) -> str:
        try:
            return await model.translate(text=text, to_lang=to_lang, params=params)
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"translation backend error: {type(e).__name__}: {e}")

    async def _safe_translate_many(texts: list[str], to_lang: str, params: GenerationParams) -> list[str]:
        try:
            return await model.translate_many(texts, to_lang, params)
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"translation backend error: {type(e).__name__}: {e}")

    async def _translate_one(text: str, from_lang: Optional[str], to_lang: str) -> tuple[str, str, str]:
        detected = detect_language_code(text) if from_lang is None else from_lang
        params = GenerationParams(
            max_new_tokens=settings.max_new_tokens,
            temperature=settings.temperature,
            top_p=settings.top_p,
            top_k=settings.top_k,
            repetition_penalty=settings.repetition_penalty,
        )
        translated = await _safe_translate(
            text=text,
            to_lang=to_lang,
            params=params,
        )
        return translated, detected, to_lang

    @app.post("/translate", dependencies=[Depends(_auth)], response_model=TranslateResponse)
    async def translate(req: TranslateRequest) -> Any:
        from_lang = normalize_lang(req.from_lang)
        to_lang = normalize_lang(req.to) or "en"
        text, detected, to_final = await _translate_one(req.text, from_lang, to_lang)
        return {"text": text, "from": detected, "to": to_final}

    @app.post("/kiss", dependencies=[Depends(_auth)], response_model=TranslateResponse)
    async def translate_kiss(req: TranslateRequest) -> Any:
        from_lang = normalize_lang(req.from_lang)
        to_lang = normalize_lang(req.to) or "en"
        text, detected, to_final = await _translate_one(req.text, from_lang, to_lang)
        return {"text": text, "from": detected, "to": to_final}

    @app.post("/imme", dependencies=[Depends(_auth)], response_model=ImmersiveResponse)
    async def translate_immersive(req: ImmersiveRequest) -> Any:
        source_lang = normalize_lang(req.source_lang)
        target_lang = normalize_lang(req.target_lang) or "en"

        if not req.text_list:
            return {"translations": []}

        detected_list = [source_lang if source_lang is not None else detect_language_code(t) for t in req.text_list]
        params = GenerationParams(
            max_new_tokens=settings.max_new_tokens,
            temperature=settings.temperature,
            top_p=settings.top_p,
            top_k=settings.top_k,
            repetition_penalty=settings.repetition_penalty,
        )

        translated_list: list[str]
        if settings.imme_batch != "off" and len(req.text_list) > 1:
            try:
                translated_list = await _safe_translate_many(req.text_list, target_lang, params)
            except HTTPException:
                if settings.imme_batch == "on":
                    raise
                translated_list = []
                for t in req.text_list:
                    translated_list.append(await _safe_translate(text=t, to_lang=target_lang, params=params))
            except Exception:
                if settings.imme_batch == "on":
                    raise
                translated_list = []
                for t in req.text_list:
                    translated_list.append(await _safe_translate(text=t, to_lang=target_lang, params=params))
        else:
            translated_list = []
            for t in req.text_list:
                translated_list.append(await _safe_translate(text=t, to_lang=target_lang, params=params))

        if len(translated_list) != len(detected_list):
            # Shouldn't happen; keep correctness over speed.
            if settings.imme_batch == "on":
                raise RuntimeError("translate_many returned unexpected length")
            translated_list = []
            for t in req.text_list:
                translated_list.append(await _safe_translate(text=t, to_lang=target_lang, params=params))

        out = [{"detected_source_lang": detected_list[i], "text": translated_list[i]} for i in range(len(detected_list))]
        return {"translations": out}

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
        translated, detected, _ = await _translate_one(req.text, source_code, target_code)
        return {
            "text": req.text,
            "from": _hcfy_to_name(detected),
            "to": _hcfy_to_name(target_code),
            "result": [translated],
        }

    @app.post("/deeplx", dependencies=[Depends(_auth)], response_model=DeeplxResponse)
    async def translate_deeplx(req: DeeplxRequest) -> Any:
        from_lang = normalize_lang(req.source_lang) or "en"
        to_lang = normalize_lang(req.target_lang) or "zh"
        translated, detected, to_final = await _translate_one(req.text, from_lang, to_lang)
        return {
            "code": 200,
            "id": int(time.time() * 1000),
            "data": translated,
            "alternatives": [],
            "source_lang": detected.upper(),
            "target_lang": to_final.upper(),
            "method": "Free",
        }

    @app.on_event("startup")
    async def _startup() -> None:
        if settings.preload_model:
            await model.ensure_loaded()

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        model.close()

    return app
