from __future__ import annotations

from pydantic import BaseModel, Field


class Settings(BaseModel):
    host: str = Field(default="127.0.0.1", alias="HOST")
    port: int = Field(default=3000, alias="PORT")
    api_key: str = Field(default="", alias="API_KEY")
    model_id: str = Field(default="mlx-community/Hy-MT2-1.8B-4bit", alias="MODEL_ID")
    max_new_tokens: int = Field(default=1024, alias="MAX_NEW_TOKENS")
    temperature: float = Field(default=0.0, alias="TEMPERATURE")
    top_p: float = Field(default=0.6, alias="TOP_P")
    top_k: int = Field(default=20, alias="TOP_K")
    repetition_penalty: float = Field(default=1.05, alias="REPETITION_PENALTY")
    preload_model: bool = Field(default=True, alias="PRELOAD_MODEL")
    backend: str = Field(default="auto", alias="BACKEND")  # auto | mlx | transformers | http
    device: str = Field(default="auto", alias="DEVICE")  # transformers only: auto | cpu | cuda
    dtype: str = Field(default="auto", alias="DTYPE")  # transformers only: auto | float16 | bfloat16 | float32
    model_max_concurrency: int = Field(default=1, alias="MODEL_MAX_CONCURRENCY")
    uvicorn_workers: int = Field(default=1, alias="UVICORN_WORKERS")
    imme_batch: str = Field(default="auto", alias="IMME_BATCH")  # auto | on | off
    max_input_chars: int = Field(default=2000, alias="MAX_INPUT_CHARS")
    imme_batch_size: int = Field(default=32, alias="IMME_BATCH_SIZE")
    imme_max_texts: int = Field(default=1024, alias="IMME_MAX_TEXTS")
    prompt_style: str = Field(default="legacy", alias="PROMPT_STYLE")  # legacy | official

    # HTTP engine (BACKEND=http): any OpenAI-compatible chat-completions server
    http_base_url: str = Field(default="", alias="MODEL_HTTP_BASE_URL")
    http_model: str = Field(default="", alias="MODEL_HTTP_MODEL")
    http_api_key: str = Field(default="", alias="MODEL_HTTP_API_KEY")
    http_timeout_s: float = Field(default=120.0, alias="MODEL_HTTP_TIMEOUT")

    # Router / shortlist
    router_enabled: bool = Field(default=True, alias="ROUTER_ENABLED")
    router_skip_symbol_only: bool = Field(default=True, alias="ROUTER_SKIP_SYMBOL_ONLY")
    router_skip_untranslatable: bool = Field(default=True, alias="ROUTER_SKIP_UNTRANSLATABLE")
    router_skip_same_language: bool = Field(default=True, alias="ROUTER_SKIP_SAME_LANGUAGE")
    router_cache_size: int = Field(default=4096, alias="ROUTER_CACHE_SIZE")
    router_retry: bool = Field(default=True, alias="ROUTER_RETRY")
    router_fast_url: str = Field(default="", alias="ROUTER_FAST_URL")
    router_fast_pairs: str = Field(default="en:zh", alias="ROUTER_FAST_PAIRS")
    router_fast_max_chars: int = Field(default=240, alias="ROUTER_FAST_MAX_CHARS")
    router_fast_timeout_s: float = Field(default=30.0, alias="ROUTER_FAST_TIMEOUT")
    router_fast_api_key: str = Field(default="", alias="ROUTER_FAST_API_KEY")
    shortlist_path: str = Field(default="", alias="SHORTLIST_PATH")
    glossary_path: str = Field(default="", alias="GLOSSARY_PATH")
    shortlist_top_k: int = Field(default=5, alias="SHORTLIST_TOP_K")
    shortlist_reuse: float = Field(default=0.97, alias="SHORTLIST_REUSE")
    shortlist_reference: float = Field(default=0.80, alias="SHORTLIST_REFERENCE")
    shortlist_reference_limit: int = Field(default=1, alias="SHORTLIST_REFERENCE_LIMIT")
    shortlist_rerank: str = Field(default="off", alias="SHORTLIST_RERANK")  # off | model
    glossary_limit: int = Field(default=8, alias="GLOSSARY_LIMIT")
