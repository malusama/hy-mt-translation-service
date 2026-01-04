from __future__ import annotations

from pydantic import BaseModel, Field


class Settings(BaseModel):
    host: str = Field(default="127.0.0.1", alias="HOST")
    port: int = Field(default=3000, alias="PORT")
    api_key: str = Field(default="", alias="API_KEY")
    model_id: str = Field(default="m-i/HY-MT1.5-1.8B-mlx-8Bit", alias="MODEL_ID")
    max_new_tokens: int = Field(default=1024, alias="MAX_NEW_TOKENS")
    temperature: float = Field(default=0.0, alias="TEMPERATURE")
    top_p: float = Field(default=0.6, alias="TOP_P")
    top_k: int = Field(default=20, alias="TOP_K")
    repetition_penalty: float = Field(default=1.05, alias="REPETITION_PENALTY")
    preload_model: bool = Field(default=True, alias="PRELOAD_MODEL")
    backend: str = Field(default="auto", alias="BACKEND")  # auto | mlx | transformers
    device: str = Field(default="auto", alias="DEVICE")  # transformers only: auto | cpu | cuda
    dtype: str = Field(default="auto", alias="DTYPE")  # transformers only: auto | float16 | bfloat16 | float32
    model_max_concurrency: int = Field(default=1, alias="MODEL_MAX_CONCURRENCY")
    uvicorn_workers: int = Field(default=1, alias="UVICORN_WORKERS")
    imme_batch: str = Field(default="auto", alias="IMME_BATCH")  # auto | on | off
    max_input_chars: int = Field(default=2000, alias="MAX_INPUT_CHARS")
    imme_batch_size: int = Field(default=32, alias="IMME_BATCH_SIZE")
    imme_max_texts: int = Field(default=1024, alias="IMME_MAX_TEXTS")
