# Note:
# - MLX/Metal only works on macOS, and cannot run inside a Linux Docker image.
# - This Docker image runs the same API using the Transformers backend.

FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app

# Install app (Transformers backend)
COPY pyproject.toml README.md /app/
COPY src /app/src
RUN pip install --no-cache-dir ".[transformers]"

ENV HOST=0.0.0.0 \
    PORT=3000 \
    BACKEND=transformers \
    DEVICE=cpu \
    DTYPE=float32 \
    MODEL_ID=tencent/HY-MT1.5-1.8B \
    PRELOAD_MODEL=1

EXPOSE 3000

CMD ["hy-mt-server"]
