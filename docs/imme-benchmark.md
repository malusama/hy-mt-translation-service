# `/imme` 网关响应时间 & 译文质量抽样（2026-01-04）

测试对象：

- URL：`https://<your-gateway-host>/imme`
- 方法：`POST` JSON
- 请求示例：

```json
{
  "target_lang": "zh",
  "text_list": ["Hello world"]
}
```

响应示例：

```json
{
  "translations": [
    { "detected_source_lang": "en", "text": "你好，世界。" }
  ]
}
```

说明：

- 本报告结果来自 2026-01-04 的**顺序（非并发）**请求；不同时间/地区/负载会有波动。
- 网关链路为 Cloudflare Worker → RunPod Serverless（同步等待一段时间，超时则可能返回 `202` 并要求轮询）。
- 观测到响应头 `CF-Ray` 后缀为 `-SIN`（当次请求走 Cloudflare 新加坡 PoP），延迟会受 PoP 与 RunPod region 影响。

---

## 1) 不同长度（单段 `text_list[0]`）响应时间

配置：

- `target_lang=zh`
- 每个长度 `warmup=1`，`trials=4`（大尺寸部分 `trials=2`）
- 单请求超时 `20s`（大尺寸 `40s`），均未触发 `202` 轮询（全部直接 `200` 返回）
- 测试文本为英文 seed 句子重复拼接并裁剪到指定字符数

结果（单位：毫秒）：

| 输入长度（chars） | OK/Trials | Async(202) | p50 | p95 | mean |
|---:|:---:|---:|---:|---:|---:|
| 16 | 4/4 | 0 | 1964.8 | 2020.8 | 1974.3 |
| 64 | 4/4 | 0 | 1939.3 | 1976.4 | 1947.1 |
| 256 | 4/4 | 0 | 2993.9 | 3002.5 | 2990.9 |
| 1024 | 4/4 | 0 | 3029.3 | 3093.3 | 3026.0 |
| 4096 | 4/4 | 0 | 3000.5 | 3033.2 | 3006.0 |
| 8192 | 4/4 | 0 | 3024.1 | 3079.9 | 3029.7 |
| 16384 | 2/2 | 0 | 3137.0 | 3223.7 | 3137.0 |
| 32768 | 2/2 | 0 | 3076.4 | 3096.3 | 3076.4 |

### 关键结论（单段长文本）

- 延迟随输入长度增长并不明显，原因见「3) 输出疑似被截断」：输出长度几乎固定在 ~130 字符左右，推理计算量被“输出上限”主导。

---

## 2) 不同段数（多段 `text_list`）响应时间与上限

### 2.1 1–32 段（每段约 96 chars）

配置：

- `segment_len≈96 chars`
- `trials=4`，`warmup=1`，超时 `30s`

结果（单位：毫秒）：

| 段数（segs） | OK/Trials | p50 | p95 | mean |
|---:|:---:|---:|---:|---:|
| 1 | 4/4 | 1960.1 | 1965.5 | 1949.5 |
| 2 | 4/4 | 1959.4 | 2150.2 | 1999.7 |
| 4 | 4/4 | 2034.6 | 2202.5 | 2050.7 |
| 8 | 4/4 | 1980.5 | 1988.3 | 1979.1 |
| 16 | 4/4 | 1941.4 | 1975.1 | 1942.7 |
| 32 | 4/4 | 2108.7 | 2250.2 | 2101.8 |

补充：`32` 段请求可稳定返回 `translations=32`。

### 2.2 更大段数（单次测量）

同一短句重复，测得：

| 段数 | 结果 | 总耗时 |
|---:|:---:|---:|
| 128 | 200 | ~3.229s |
| 512 | 200 | ~8.054s |
| 1024 | 200 | ~12.485s |
| 2048 | 500 | ~12.868s |

`2048` 段失败原因（RunPod job failed）：Transformers `generate` 走批量推理触发 **CUDA OOM**（报错中显示尝试分配 2.77 GiB，GPU 空闲仅 1.60 GiB）。

### 关键结论（多段批量）

- 1–32 段延迟几乎不变，说明服务端可能在做合批/批量推理或 pipeline 固定开销占主导。
- 超大段数（本次 2048 段）会直接触发 OOM：需要服务端做**分批**（chunk）、限制 `text_list` 上限或在 `/imme` 上关闭 batch/降低 batch size。

---

## 3) 输出疑似被截断（影响“长文本准确率”评估）

观测：输入字符数从 `256 → 32768` 增长，但 `translations[0].text` 输出字符数基本固定在 `~126–135`。

示例（单段输入 `32768 chars`，返回 `out_chars=128`，输出开头预览）：

> `机器翻译是一件非常困难的事情。它必须能够保留文本的含义、语气以及格式，包括像3.14159这样的数字、2026-01-04这样的日期，还有像\`foo(bar)\`这样的代码...`

这说明当前 `/imme` 对“超长单段文本”并不适合作为完整翻译接口：更像是用于“切段后的短句/段落翻译”。

---

## 4) 译文准确率抽样（主观、小样本）

### 4.1 英→中

- `Hello world.` → `你好，世界。`（OK）
- `It's raining cats and dogs.` → `下着倾盆大雨呢。`（意译 OK）
- `Break a leg!` → `祝你好运！`（译文 OK；但 `detected_source_lang` 误判为 `es`）
- `The API returns HTTP 429 when rate limited.` → `当遇到速率限制时，该 API 会返回 HTTP 429 状态码。`（OK）
- `Use \`git reset --hard\` with caution.` → `使用 \`git reset --hard\` 时，请务必谨慎操作。`（OK）
- `I saw her duck.` → `我看见她突然蹲了下来。`（歧义句，选了“duck=躲闪/蹲下”含义，可接受但非唯一）

### 4.2 中→英

- `你好，世界。` → `Hello, world.`（OK）
- `塞翁失马，焉知非福。` → `What’s lost by Saiwang might actually be a blessing in disguise.`（不够地道，“塞翁”被人名化 Saiwang）
- `这个接口在高并发下会触发 429，需要指数退避重试。` → `... exponential backoff retries ...`（OK）
- `我看到她的鸭子了。` → `I saw her ducks.`（把单数译成复数；轻微问题）

结论：短句/技术句整体可用；典故、歧义、细粒度数量一致性仍有一定错误率。

---

## 5) 额外问题：`OPTIONS /imme` 返回 500（可能影响浏览器调用）

多次尝试 `OPTIONS /imme` 返回：

- HTTP `500`
- body：`error code: 1101`（Cloudflare Worker 异常）

如果前端用 `fetch` 跨域 JSON 请求，通常会触发 CORS preflight（`OPTIONS`），这会导致浏览器端直接失败（即使 `POST` 本身是 OK 的）。

---

## 6) 复现命令

单段翻译：

```bash
curl -sS -X POST 'https://<your-gateway-host>/imme' \
  -H 'content-type: application/json' \
  -d '{"target_lang":"zh","text_list":["Hello world."]}'
```

检查 `OPTIONS`：

```bash
curl -i -sS -X OPTIONS 'https://<your-gateway-host>/imme'
```

---

## 7) 解决方案（对应本仓库的改动/配置）

### 7.1 长文本“截断”

根因通常是：**上下文长度有限 + `MAX_NEW_TOKENS`/输出上限** 导致模型只返回开头一小段（看起来像“成功”，但其实被截断）。

本仓库已在服务端加入“按字符分块翻译再拼接”的兜底机制：

- `MAX_INPUT_CHARS`：单段文本超过该长度会自动拆分为多个 chunk 翻译再拼接
- 如果你仍观察到输出被截断：要么 **调大 `MAX_NEW_TOKENS`**，要么 **调小 `MAX_INPUT_CHARS`**

### 7.2 超大 `text_list` OOM（2048 段）

根因：`generate()` 批量推理时显存随 batch/序列长度增长，超大段数会直接 OOM。

本仓库已在 `/imme` 内做分批：

- `IMME_BATCH_SIZE`：每批最多处理多少段
- `IMME_MAX_TEXTS`：保护上限，超出直接 413

### 7.3 `OPTIONS /imme` 500（Cloudflare Worker 1101）

这是 **网关 Worker** 的运行时异常，不是模型服务本身的问题。

- 若你能改 Worker：确保 `OPTIONS` 在所有逻辑之前直接返回 `204`，并带上 `Access-Control-Allow-*` 头（尤其 `content-type,authorization`）
- 若你不想处理网关：用 RunPod **Pods** 直接暴露 `:3000`，浏览器预检会由 FastAPI 的 CORS 中间件正常响应
