# 在 RunPod 上运行（Linux + NVIDIA GPU）

本项目的 `mlx/mlx-lm` 后端只能在 macOS + Apple Silicon 上跑；RunPod 属于 Linux + NVIDIA GPU。

`tencent/HY-MT1.5-1.8B` 的架构在 **vLLM 里可能不受支持**，但可以用 **GGUF + llama.cpp** 在 Linux + NVIDIA GPU 上直接跑（并且支持 OpenAI 风格 SSE 流式输出）。

## 你应该用 Pods（长驻服务）

如果你的目标是让沉浸式翻译等插件“直接请求一个稳定的 HTTP 地址”，建议使用 **RunPod Pods**（长驻 Web 服务）。  
`https://console.runpod.io/deploy` 通常是 **Serverless**（按请求计费/冷启动），不太适合直接挂一个长期对外的 FastAPI 服务。

## Pods：transformers + CUDA（推荐）

仓库提供 `Dockerfile.runpod.transformers`，在容器里直接启动本项目 FastAPI 服务（对外 API）：`0.0.0.0:3000`。

### 1) 构建并推送镜像

RunPod 侧需要能拉取你的镜像（DockerHub/GHCR/私有仓库均可）。

在本机（推荐 buildx 直接构建 `linux/amd64` 并推送）：

```bash
docker buildx build --platform linux/amd64 \
  -f Dockerfile.runpod.transformers \
  -t <your-registry>/hy-mt-transformers-gpu:runpod \
  --push .
```

### 2) 创建 Pod（GPU）

在 RunPod 创建一个 GPU Pod，关键配置：

- **Image**：`<your-registry>/hy-mt-transformers-gpu:runpod`
- **Expose 端口**：`3000`
- **Volume（强烈建议）**：挂载持久盘到 `/runpod-volume`（用于缓存 HF 模型，避免每次重启重新下载）

推荐环境变量（Pod 的 Environment Variables）：

- `MODEL_ID=tencent/HY-MT1.5-1.8B`
- `BACKEND=transformers`
- `DEVICE=cuda`
- `DTYPE=float16`
- `HF_HOME=/runpod-volume/hf`（推荐；`Dockerfile.runpod.transformers` 默认也是这个路径）
- 如模型需要登录：`HF_TOKEN=...`（也可同时设置 `HUGGING_FACE_HUB_TOKEN`）
- 可选：`API_KEY=...`（开启后所有请求需带 `Authorization: Bearer <API_KEY>` 或 `?token=`）

### 3) 验证

拿到 Pod 对外映射的 `3000` 端口后：

```bash
curl http://<POD_HOST>:<PUBLIC_PORT>/health
```

以及沉浸式翻译接口示例：

```bash
curl -X POST "http://<POD_HOST>:<PUBLIC_PORT>/imme" \
  -H "Content-Type: application/json" \
  -d '{"source_lang":"auto","target_lang":"zh","text_list":["Hello world","How are you?"]}'
```

## Serverless：scale-to-zero（配合网关提供 HTTP）

Serverless 适合“绝大多数时间没请求”的场景：`active_workers=0` 时可 scale-to-zero，基本不收空闲 GPU 费用；代价是有 **冷启动**（尤其第一次需要加载模型/权重）。

RunPod Serverless 有两种形态：

- **Load Balancing（HTTP Workers）**：对外就是 HTTP（推荐；可直接支持 SSE 流式输出）。
- **Traditional（Job API）**：对外是 `/run` + `/status` 的 Job API（如需给浏览器/插件提供稳定 HTTP 地址，建议加一层 Cloudflare Worker 网关，本仓库提供 `cloudflare-worker/`）。

### 1) Load Balancing（推荐：直接 HTTP + 支持 SSE）

构建并推送镜像：

```bash
docker buildx build --platform linux/amd64 \
  -f Dockerfile.runpod.loadbalancing \
  -t <your-registry>/hy-mt-gguf-llama-rust:runpod-lb \
  --push .
```

在 RunPod 创建 **Serverless Endpoint（Load Balancing）**，并配置：

- **Image**：`<your-registry>/hy-mt-gguf-llama-rust:runpod-lb`
- **Expose 端口**：`3000`
- **Volume（建议）**：挂载到 `/runpod-volume`（用于缓存 GGUF 文件，避免每次冷启动重复下载）

推荐环境变量（Endpoint Environment Variables）：

- `MODEL_GGUF_URL=https://huggingface.co/tencent/HY-MT1.5-1.8B-GGUF/resolve/main/HY-MT1.5-1.8B-Q4_K_M.gguf`
- `MODEL_DIR=/runpod-volume/models`
- `LLAMA_N_GPU_LAYERS=999`（尽量全上 GPU；OOM 再调小）
- `MAX_NEW_TOKENS=1024`、`TEMPERATURE=0`、`TOP_P=0.6`
- （可选）`API_KEY=...`（开启后所有请求需带 `Authorization: Bearer <API_KEY>` 或 `?token=`）

验证（拿到对外映射的 `3000` 端口后）：

```bash
curl http://<POD_HOST>:<PUBLIC_PORT>/health
```

### 2) Traditional（Job API：兼容 Cloudflare Worker 网关）

构建并推送镜像：

```bash
docker buildx build --platform linux/amd64 \
  -f Dockerfile.runpod.serverless \
  -t <your-registry>/hy-mt-gguf-llama-rust:runpod-job \
  --push .
```

在 RunPod 创建 **Serverless Endpoint（Traditional）**，并使用 `cloudflare-worker/` 将 `/imme|/translate|/detect` 转成 Job API。

### 3) 部署 Cloudflare Worker 网关（Traditional 才需要）

见 `cloudflare-worker/README.md`。
