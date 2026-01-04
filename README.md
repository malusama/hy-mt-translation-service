# HY-MT (MLX) Translation Service

用 `mlx`/`mlx-lm` 在 Apple Silicon 上运行 `tencent/HY-MT1.5-1.8B`（推荐直接用已转换的 MLX 权重），并提供与 `LinguaSpark/server` 类似的 API，给沉浸式翻译等插件调用。

## 部署方式一览

- **macOS + Apple Silicon**：`mlx/mlx-lm`（最低延迟/本机自用）
- **Linux / 无 GPU**：Docker + `transformers`（CPU，较慢）
- **Linux + NVIDIA GPU（RunPod 等）**：`transformers + CUDA`（推荐，见 `RUNPOD.md`）
- **RunPod Serverless（Worker 网关）**：scale-to-zero 更省钱；用 `cloudflare-worker/` 把 Job API 变成 HTTP（见 `RUNPOD.md` / `cloudflare-worker/README.md`）

## 运行环境

- macOS + Apple Silicon
- Python 3.11（建议用 `uv` 创建 venv）

## 快速开始

```bash
cd hy-mt-mlx-translation-service
uv venv --python python3.11
source .venv/bin/activate
uv pip install -e '.[mlx]'

# 启动
export HOST=127.0.0.1
export PORT=3000
export MODEL_ID=m-i/HY-MT1.5-1.8B-mlx-8Bit
hy-mt-server
```

也可以用环境文件：

```bash
cp .env.example .env
set -a && source .env && set +a
hy-mt-server
```

健康检查：

```bash
curl http://127.0.0.1:3000/health
```

## Docker（注意）

`mlx`/`mlx-lm` 依赖 macOS + Metal，无法在 Linux Docker 容器里安装/运行。仓库里的 `Dockerfile` 会使用 `transformers` 后端提供同样的 API（更通用，但在纯 CPU 上会更慢）。

构建镜像：

```bash
docker build -t hy-mt-service .
```

启动：

```bash
docker run --rm -p 3000:3000 hy-mt-service
```

推荐：运行时挂载 **HF 缓存目录**（避免每次重启重复下载）：

```bash
docker run --rm -p 3000:3000 \
  -v hy-mt-hf-cache:/root/.cache/huggingface \
  hy-mt-service
```

如果你已经把模型下载到本地目录，也可以直接挂载（完全离线）：

```bash
docker run --rm -p 3000:3000 \
  -v "$PWD/model:/opt/model:ro" \
  -e MODEL_ID=/opt/model \
  -e TRANSFORMERS_OFFLINE=1 \
  hy-mt-service
```

## 环境变量

- `HOST`：监听地址，默认 `127.0.0.1`
- `PORT`：端口，默认 `3000`
- `MODEL_ID`：HF 模型 ID（`mlx` 默认 `m-i/HY-MT1.5-1.8B-mlx-8Bit`；`transformers` 默认 `tencent/HY-MT1.5-1.8B`）
- `API_KEY`：可选，设置后要求 `Authorization: Bearer <key>` 或 `?token=<key>`
- `MAX_NEW_TOKENS`：默认 `1024`
- `TEMPERATURE`：默认 `0`（翻译推荐用确定性输出）
- `TOP_P`：默认 `0.6`（仅在 `TEMPERATURE>0` 时生效）
- `TOP_K`：默认 `20`（仅在 `TEMPERATURE>0` 时生效）
- `REPETITION_PENALTY`：默认 `1.05`
- `MAX_INPUT_CHARS`：单段文本超过该长度会自动分块翻译再拼接（用于避免长文在上下文/输出上限下“看似成功但被截断”）
- `PRELOAD_MODEL`：默认 `1`，设为 `0` 可跳过启动时预加载（首次请求再加载）
- `BACKEND`：`auto|mlx|transformers`（默认 `auto`；Docker 默认用 `transformers`）
- `DEVICE`：`auto|cpu|cuda`（transformers 用）
- `DTYPE`：`auto|float16|bfloat16|float32`（transformers 用）
- `MODEL_MAX_CONCURRENCY`：每个进程允许同时进行的生成次数（默认 `1`；过大可能导致卡顿/内存飙升）
- `UVICORN_WORKERS`：Uvicorn 多进程 worker 数（默认 `1`；每个 worker 会各自加载一份模型）
- `IMME_BATCH`：`/imme` 是否尝试批量翻译：`auto|on|off`（默认 `auto`；失败会回退为逐条翻译）
- `IMME_BATCH_SIZE`：`/imme` 批量翻译时的分批大小（防止超大 `text_list` 触发 OOM）
- `IMME_MAX_TEXTS`：`/imme` 允许的 `text_list` 最大段数（超过返回 413）

## 并发/性能建议

- **沉浸式翻译**：尽量走 `POST /imme` 一次带多段（本服务会尝试把 `text_list` 合并成一次生成；失败自动回退）。
- **Apple Silicon + MLX**：优先保持 `UVICORN_WORKERS=1`，再按需尝试 `MODEL_MAX_CONCURRENCY=2`（过大通常只会更慢）。
- **Docker/CPU（transformers）**：更优先调大 `UVICORN_WORKERS`，再按需调 `MODEL_MAX_CONCURRENCY`。

## API（兼容 LinguaSpark/server）

- `POST /translate`：`{ "text": "...", "from": "en"(可选/auto), "to": "zh" }`
- `POST /detect`：`{ "text": "..." }`
- `POST /imme`：沉浸式翻译：`{ "source_lang": "auto"(可选), "target_lang": "zh", "text_list": ["..."] }`
- `POST /kiss`：同 `/translate`
- `POST /hcfy`：划词翻译兼容
- `POST /deeplx`：DeepLX 兼容
- `GET /health`

## 给沉浸式翻译插件用

把服务启动在本机后，一般填：

- Base URL：`http://127.0.0.1:3000`
- 端点：使用 `POST /imme`
- 如开启 `API_KEY`：在插件里配置请求头 `Authorization: Bearer <API_KEY>`（或用 `?token=` 方式）
