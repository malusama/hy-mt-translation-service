# Cloudflare Worker: RunPod Serverless HTTP Gateway

RunPod Serverless 对外是 Job API（`/run` + `/status`）。这个 Worker 把常见的 HTTP 路由（`/imme`、`/translate`、`/detect`）转换为 RunPod job，并在能力允许的情况下同步等待一段时间；超时则返回 `202` 并给出 `status_url` 供客户端轮询。

## 解决 `OPTIONS` 500（CORS 预检）

浏览器跨域 `fetch` 常触发 preflight（`OPTIONS`）。Worker 必须对所有路径优先返回 `204` 并带 CORS 头，否则会出现 Cloudflare `1101` 这类错误导致浏览器直接失败。

本目录的 `src/index.ts` 已在最前面处理 `OPTIONS`。

## 配置

1) 编辑 `wrangler.toml`：

- `RUNPOD_ENDPOINT_ID`：你的 RunPod Serverless endpoint id

2) 设置 RunPod API key（Wrangler secret）：

```bash
wrangler secret put RUNPOD_API_KEY
```

3)（可选）给网关加一层简单鉴权：

在 `wrangler.toml` 的 `[vars]` 里设置 `CLIENT_API_KEY`，然后客户端请求带 `Authorization: Bearer <CLIENT_API_KEY>`。

## 部署

```bash
cd cloudflare-worker
npm i
npx wrangler deploy
```

## 使用

- `GET /health`：健康检查
- `POST /imme`：透传为 `action=imme`
- `POST /translate`：透传为 `action=translate`
- `POST /detect`：透传为 `action=detect`
- `GET /job/<id>`：查询 job 状态

