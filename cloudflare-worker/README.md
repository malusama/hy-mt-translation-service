# Cloudflare Worker: RunPod Serverless HTTP Gateway

这个 Worker 支持两种后端形态，并**默认优先直连 Load Balancing（HTTP Workers）**以获得 **SSE 流式输出**：

1) **Upstream 模式（推荐）**：直接反代你的 RunPod Serverless **Load Balancing / HTTP Workers** 的公网地址（可流式）。
2) **Job 模式（兼容）**：把 `/imme|/translate|/detect` 转换为 RunPod Traditional Serverless 的 Job API（`/run` + `/status`），并在能力允许的情况下同步等待一段时间；超时则返回 `202` + `status_url` 供轮询。

## 解决 `OPTIONS` 500（CORS 预检）

浏览器跨域 `fetch` 常触发 preflight（`OPTIONS`）。Worker 必须对所有路径优先返回 `204` 并带 CORS 头，否则会出现 Cloudflare `1101` 这类错误导致浏览器直接失败。

本目录的 `src/index.ts` 已在最前面处理 `OPTIONS`。

## 配置

1) 编辑 `wrangler.toml`：

- **Upstream（推荐，支持 SSE）**：设置 `UPSTREAM_BASE_URL`
- **Job（传统）**：设置 `RUNPOD_ENDPOINT_ID`

2) 设置 RunPod API key（Wrangler secret）：

```bash
wrangler secret put RUNPOD_API_KEY
```

> 只有 **Job 模式**需要 `RUNPOD_API_KEY`。

3)（可选）给网关加一层简单鉴权：

在 `wrangler.toml` 的 `[vars]` 里设置 `CLIENT_API_KEY`，然后客户端请求带 `Authorization: Bearer <CLIENT_API_KEY>`。

4)（可选）上游鉴权（Upstream 模式）：

- 如果你的上游 Rust 服务开启了 `API_KEY`，可以在 `wrangler.toml` 设置 `UPSTREAM_API_KEY`，Worker 会自动给上游加 `Authorization: Bearer ...`。
- 或者设置 `FORWARD_AUTH=1`，把客户端的 `Authorization` 原样转发给上游（仅当你希望两边用同一套 token）。

## 部署

```bash
cd cloudflare-worker
npm i
npx wrangler deploy
```

## 使用

- `GET /health`：健康检查（Upstream 模式下会直接探测上游 `/health`）
- **Upstream 模式**：默认把所有路径直接反代到上游（包括 `/v1/chat/completions` 的 `stream=true` SSE）
- **Job 模式**：
  - `POST /imme`：透传为 `action=imme`
  - `POST /translate`：透传为 `action=translate`
  - `POST /detect`：透传为 `action=detect`
  - `GET /job/<id>`：查询 job 状态

### 强制切换模式

- 默认：如果配置了 `UPSTREAM_BASE_URL`，优先使用 Upstream 模式。
- 强制 Job 模式：在请求 URL 上加 `?mode=job`（例如 `/translate?mode=job`）。
