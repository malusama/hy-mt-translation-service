# 在 RunPod 上运行（Linux + NVIDIA GPU）

本项目的 `mlx/mlx-lm` 后端只能在 macOS + Apple Silicon 上跑；RunPod 属于 Linux + NVIDIA GPU。

注意：`tencent/HY-MT1.5-1.8B` 当前的模型架构在 **vLLM 里还不受支持**（即使升级 Transformers 也会报 *architectures are not supported*），因此在 RunPod 上建议直接用本项目的 **transformers + CUDA** 后端跑推理。

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

RunPod Serverless 对外是 Job API，不是 `POST /imme` 这种原生 HTTP 路由。若要给浏览器/插件提供稳定的 HTTP 地址，推荐加一层 Cloudflare Worker 网关（本仓库提供 `cloudflare-worker/`）。

### 1) 部署 Serverless（runpodctl）

1) 把 `runpod.toml` 里的 `[project].uuid` 替换成你自己的 RunPod Project UUID。

2) 确保 RunPod 项目里创建了 Network Volume（用于 `HF_HOME=/runpod-volume/hf` 缓存模型）。

3) 部署：

```bash
runpodctl project deploy
```

推荐环境变量（Endpoint Environment Variables）：

- `MODEL_ID=tencent/HY-MT1.5-1.8B`
- `DEVICE=cuda`
- `DTYPE=float16`
- `HF_HOME=/runpod-volume/hf`
- `MAX_NEW_TOKENS=1024`
- `MAX_INPUT_CHARS=2000`（长文本自动分块拼接，避免“看似成功但被截断”）
- `IMME_BATCH_SIZE=32`、`IMME_MAX_TEXTS=1024`（防止超大 `text_list` OOM）

### 2) 部署 Cloudflare Worker 网关

见 `cloudflare-worker/README.md`。
