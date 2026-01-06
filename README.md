# HY-MT Translation Service (MLX + GGUF/llama.cpp)

提供与 `LinguaSpark/server` 类似的 API，给沉浸式翻译等插件调用，支持两种后端：

- **MLX（macOS Apple Silicon）**：本机自用最低延迟（`/imme`、`/translate` 等）
- **GGUF + llama.cpp（跨平台）**：配合 Rust Web 提供接口，并支持 **OpenAI 风格 SSE 流式输出**

## 部署方式一览

- **macOS + Apple Silicon（默认）**：MLX（推荐本机自用）
- **macOS + Apple Silicon（可选）**：GGUF + llama.cpp（可选 Metal 加速；需要本机 `llama-server`）
- **Linux / 无 GPU**：GGUF + llama.cpp（CPU）
- **Linux + NVIDIA GPU（RunPod 等）**：GGUF + llama.cpp（可选 CUDA）
- **RunPod Serverless（Worker 网关）**：可用（但 Job API 轮询不适合逐 token SSE）

## 运行环境

- **MLX**：Python + `uv`（macOS arm64）
- **GGUF/llama.cpp**：Rust toolchain + 本机 `llama-server`（或自行编译 llama.cpp）

## 快速开始

一键启动（推荐）：

```bash
bash scripts/dev.sh
```

说明：
- **macOS Apple Silicon**：`scripts/dev.sh` 默认启动 **MLX** 后端（`BACKEND=mlx`）。
- **其它平台 / 或想用 GGUF/llama.cpp**：直接运行 `bash scripts/run-rust.sh`（需要 `MODEL_GGUF` + 本机 `llama-server`），或在 `.env` 里设置 `BACKEND=rust` 后再跑 `scripts/dev.sh`。

准备环境文件：

```bash
cp .env.example .env
set -a && source .env && set +a
bash scripts/dev.sh
```

健康检查：

```bash
curl http://127.0.0.1:3000/health
```

## 模型（GGUF）

仅在 GGUF/llama.cpp 路线需要：下载 `tencent/HY-MT1.5-1.8B-GGUF` 的某个量化文件到本地，然后在 `.env` 里设置 `MODEL_GGUF=/path/to/model.gguf`。

## OpenAI 风格流式输出（SSE）

- `POST /v1/chat/completions`（支持 `stream=true`；额外支持自定义字段 `to`/`from`）
- 兼容接口也可用 `?stream=1`：`POST /translate?stream=1` / `POST /imme?stream=1`

说明：RunPod Serverless（Job API + 轮询）这条链路天然不适合做逐 token SSE 流式；如需流式，建议使用 Pods（长驻 HTTP）或自建常驻服务。

## macOS 自启动（launchd）

安装并立即启动：

```bash
bash scripts/launchd-install.sh
```

卸载：

```bash
bash scripts/launchd-uninstall.sh
```

说明：自启动默认运行 Rust Web（`scripts/run-rust.sh --no-build`），并在脚本内读取 `.env`（包括 `MODEL_GGUF`）。

## Docker（注意）

当前主路径是 Rust + llama.cpp；如需容器化建议自行构建镜像并挂载 `.gguf` 模型文件（本仓库旧的 Python/Docker 方案仅作参考）。

## 环境变量

通用：
- `HOST`：监听地址，默认 `127.0.0.1`
- `PORT`：端口，默认 `3000`
- `API_KEY`：可选，设置后要求 `Authorization: Bearer <key>` 或 `?token=<key>`
- `MAX_NEW_TOKENS`：默认 `1024`
- `TEMPERATURE`：默认 `0`（翻译推荐用确定性输出）
- `TOP_P`：默认 `0.6`（仅在 `TEMPERATURE>0` 时生效）
- `TOP_K`：默认 `20`（仅在 `TEMPERATURE>0` 时生效）
- `REPETITION_PENALTY`：默认 `1.05`
- `MAX_INPUT_CHARS`：单段文本超过该长度会自动分块翻译再拼接（用于避免长文在上下文/输出上限下“看似成功但被截断”）
- `MODEL_MAX_CONCURRENCY`：每个进程允许同时进行的生成次数（默认 `1`；过大可能导致卡顿/内存飙升）
- `IMME_MAX_TEXTS`：`/imme` 允许的 `text_list` 最大段数（超过返回 413）

MLX：
- `BACKEND=mlx`
- `MODEL_ID`：默认 `m-i/HY-MT1.5-1.8B-mlx-8Bit`
- `PRELOAD_MODEL`：默认 `0`（启动时不强制预加载；首次请求再加载）
- `UVICORN_WORKERS`：默认 `1`

GGUF/llama.cpp（Rust Web）：
- `MODEL_GGUF`：本地 GGUF 模型路径（必填）
- `LLAMA_FEATURES`：编译 features（macOS 建议 `metal`；Linux NVIDIA 建议 `cuda`）
- `LLAMA_N_GPU_LAYERS`：加载到 GPU 的层数（默认 `0`）
- `LLAMA_N_CTX`：上下文长度（默认 `0` 表示使用模型默认）

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
