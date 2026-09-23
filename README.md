# Hy-MT2 Translation Service (MLX + llama.cpp/HTTP)

提供与 `LinguaSpark/server` 类似的 API，给沉浸式翻译等插件调用。默认模型是
**`mlx-community/Hy-MT2-1.8B-4bit`**（腾讯 Hy-MT2 系列），三条推理路线：

- **MLX（macOS Apple Silicon）**：本机自用最低延迟（`/imme`、`/translate` 等）
- **GGUF + llama.cpp（跨平台）**：配合 Rust Web 提供接口，并支持 **OpenAI 风格 SSE 流式输出**
- **HTTP（`BACKEND=http`）**：把 Python 这层当网关，推理交给任意 OpenAI 兼容的
  `llama-server` / RunPod / 另一个实例（本机实测 llama.cpp Metal 比 MLX 快约 2.5×）

另外内置一层**路由（router）**与**shortlist / 术语表**：能不下模型的段直接跳过、
命中缓存/翻译记忆直接复用、短句走快通道、只对真正需要的段落调用 1.8B 模型。
见 [路由](#路由router) 与 [Shortlist / 术语表](#shortlist--术语表)。

## 部署方式一览

- **macOS + Apple Silicon（默认）**：MLX（推荐本机自用）
- **macOS + Apple Silicon（可选）**：GGUF + llama.cpp（可选 Metal 加速；需要本机 `llama-server`）
- **Linux / 无 GPU**：GGUF + llama.cpp（CPU）
- **Linux + NVIDIA GPU（RunPod 等）**：GGUF + llama.cpp（可选 CUDA）
- **RunPod Serverless（Worker 网关）**：可用（但 Job API 轮询不适合逐 token SSE）

试过但**暂时不要用**的：vLLM 官方的 Apple Silicon 插件
[vllm-metal](https://github.com/vllm-project/vllm-metal)。它能服务
`mlx-community/Hy-MT2-1.8B-4bit`（Hunyuan dense 在它的支持矩阵里）并返回正确译文，
但实测单条 8–20 秒，比同一台机器上的 `mlx-lm` 慢 60–150 倍，原因未查清——
过程和待查假设见 [docs/vllm-metal-20260923.md](docs/vllm-metal-20260923.md)，
压测脚本 `tools/bench_vllm_metal.py`。

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

腾讯 Hy-MT2 系列（2026-05 发布）包含 1.8B / 7B / 30B-A3B，本服务默认用 1.8B：

| 用途 | 仓库 |
|---|---|
| MLX（默认） | `mlx-community/Hy-MT2-1.8B-4bit` |
| transformers / GGUF 参考 | `tencent/Hy-MT2-1.8B` |
| llama.cpp | `tencent/Hy-MT2-1.8B-GGUF`（`Q4_K_M` / `Q6_K` / `Q8_0`） |
| 极限量化 | `tencent/Hy-MT2-1.8B-1.25Bit-GGUF`、`...-2bit-GGUF`、`tencent/Hy-MT2-1.8B-FP8` |

GGUF/llama.cpp 路线：下载一个量化文件到本地，然后在 `.env` 里设置
`MODEL_GGUF=/path/to/Hy-MT2-1.8B-Q4_K_M.gguf`。

> 注意：官方 1.25-bit GGUF 需要 AngelSlim 提供的定制 llama.cpp，Homebrew 的
> `llama.cpp` 无法载入（实测报 `failed to load model`）。Q4_K_M 等常规量化没有这个问题。

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
- `PROMPT_STYLE`：`legacy`（默认，短指令，prefill 更便宜）或 `official`（Hy-MT2 模型卡措辞）

MLX：
- `BACKEND=mlx`
- `MODEL_ID`：默认 `mlx-community/Hy-MT2-1.8B-4bit`
- `PRELOAD_MODEL`：默认 `0`（启动时不强制预加载；首次请求再加载）
- `UVICORN_WORKERS`：默认 `1`

HTTP（`BACKEND=http`）：
- `MODEL_HTTP_BASE_URL`：OpenAI 兼容服务地址（必填），如 `http://127.0.0.1:18080`
- `MODEL_HTTP_MODEL`：可选，请求里带上的 model 名
- `MODEL_HTTP_API_KEY`：可选，`Authorization: Bearer ...`
- `MODEL_HTTP_TIMEOUT`：单次请求超时，默认 `120`（秒）

GGUF/llama.cpp（Rust Web）：
- `MODEL_GGUF`：本地 GGUF 模型路径（必填）
- `LLAMA_FEATURES`：编译 features（macOS 建议 `metal`；Linux NVIDIA 建议 `cuda`）
- `LLAMA_N_GPU_LAYERS`：加载到 GPU 的层数（默认 `0`）
- `LLAMA_N_CTX`：上下文长度（默认 `0` 表示使用模型默认）

路由（router）：
- `ROUTER_ENABLED`：默认 `1`；设为 `0` 完全回到旧行为（不做跳过/缓存/快通道，也丢弃 `/route`、`/stats` 的判定信息）
- `ROUTER_SKIP_SYMBOL_ONLY`：默认 `1`，纯数字/符号/分隔符段直接原样返回
- `ROUTER_SKIP_UNTRANSLATABLE`：默认 `1`，标识符/代码/URL/邮箱/版本号等“没有可翻译内容”的段直接原样返回（不调模型）
- `ROUTER_SKIP_SAME_LANGUAGE`：默认 `1`，只在**脚本可判定**时判定“已是目标语言”（日文假名→ja、谚文→ko、汉字→zh；拉丁文不做判定，避免把法语当作英文目标跳过）
- `ROUTER_CACHE_SIZE`：精确命中缓存条数，默认 `4096`（LRU）
- `ROUTER_RETRY`：默认 `1`，主模型输出触发硬性问题（空/回声/目标文字缺失/复读）时用另一种 prompt 措辞重试一次
- `ROUTER_FAST_URL`：快通道地址（LinguaSpark/hy-mt 兼容的 `/imme`），例如本机的 `marian-edge`：`http://127.0.0.1:3000`
- `ROUTER_FAST_PAIRS`：快通道支持的语向，默认 `en:zh`（逗号分隔，如 `en:zh,en:ja`）
- `ROUTER_FAST_MAX_CHARS`：超过该长度的段不走快通道，默认 `240`
- `ROUTER_FAST_TIMEOUT` / `ROUTER_FAST_API_KEY`：快通道超时与鉴权

Shortlist / 术语表：
- `SHORTLIST_PATH`：翻译记忆 JSONL（`{"source","target","source_lang","target_lang"}`），可热加载
- `GLOSSARY_PATH`：术语表 JSONL（`{"source_term","target_term","source_lang","target_lang"}`）
- `SHORTLIST_TOP_K`：召回条数，默认 `5`
- `SHORTLIST_REUSE`：直接复用阈值，默认 `0.97`（命中即零延迟，不再调用模型）
- `SHORTLIST_REFERENCE`：作为参考译文注入 prompt 的阈值，默认 `0.80`
- `SHORTLIST_REFERENCE_LIMIT`：注入条数上限，默认 `1`
- `SHORTLIST_RERANK`：`off`（默认）或 `model`（对候选做一次前向打分，取对数概率最高者）
- `GLOSSARY_LIMIT`：单段注入术语条数上限，默认 `8`

术语表与快通道的关系：**命中术语的段落不会走快通道**。术语是硬要求，只有主模型会收到
reference block；31M 学生模型会直接忽略它。代价是这些段从 ~20ms 变成 ~150ms。

## 路由（Router）

每个 segment 按顺序判定，命中即停（前 5 步都不需要前向）：

| 判定 | 触发条件 | 结果 |
|---|---|---|
| `symbol` | 没有字母（数字/标点/emoji/分隔符） | 原样返回 |
| `passthrough` | 有字母但没有可翻译内容（`TODO`/`SHA256`/`v1.2.3`/URL/邮箱/`</div>`/纯命令） | 原样返回 |
| `identity` | 显式 source==target，或脚本可判定已是目标语言 | 原样返回 |
| `cache` | 规范化后的 (source, target, text) 精确命中 | 返回缓存 |
| `memory` | shortlist 近似命中且相似度 ≥ `SHORTLIST_REUSE`，并通过质量门 | 直接复用 TM 译文 |
| `fast` | 快通道支持该语向且段长 ≤ `ROUTER_FAST_MAX_CHARS` | 快通道译文（**要过质量门**） |
| `main` | 其余 | Hy-MT2 主模型（可带术语/参考注入） |

两处质量门（`quality.py`）：快通道输出不合格会**升级**到主模型而不是直接返回；主模型输出
出现**硬性**问题（空/回声/目标文字缺失/复读）会重试一次——这是兜底，不是常规路径：实测
25 段正常语料在修复后是 0 次硬性问题（见下）。只有过门的译文才写入缓存。质量门是启发式的
（脚本、长度比、占位符/URL/format token 是否保留、复读、截断），不是 COMET 分数，**也检测
不出语义错译**（例如把 `TODO` 译成“全部完成”），那需要术语表/上下文或质量模型。

“回声/目标文字缺失”只在**确有可翻译内容**时才判：标识符、命令、URL 原样返回是正确答案，
不会触发重试；单字品牌名（`GitHub`）也不强制要求变成中文。

不看模型也能看判定：

```bash
curl -s http://127.0.0.1:3000/route -H 'content-type: application/json' \
  -d '{"text_list":["12:30","The cache is cold.","这是中文。"],"target_lang":"zh"}'
```

`GET /stats` 返回累计计数（各引擎命中数、无前向比例、缓存/记忆条目数）。

## Shortlist / 术语表

- 默认 embedder 是**确定性哈希字符 n-gram 向量**（无需下模型、无网络、跨进程稳定）。
- `memory` 复用只在这两条都通过质量门时发生：存下来的 (source, target) 对本身合格，
  且该 target 对本次 source 也合格（数字/URL 不同会挡住复用）。
- 近似但不够像的命中会作为**参考译文**注入 prompt（Hy-MT2 的 terminology 模板），由模型自己判断。
- `SHORTLIST_RERANK=model` 时，候选按**一次前向的对数概率**重排（只打分、不生成）。
- 运行期可直接追加（会同时写回 JSONL）：

```bash
curl -s http://127.0.0.1:3000/shortlist -H 'content-type: application/json' \
  -d '{"entries":[{"source":"The cache is cold.","target":"缓存已失效。","source_lang":"en","target_lang":"zh"}]}'
curl -s http://127.0.0.1:3000/glossary -H 'content-type: application/json' \
  -d '{"terms":[{"source_term":"service mesh","target_term":"服务网格","source_lang":"en","target_lang":"zh"}]}'
```

仓库里带了一份起步用的 `data/glossary.example.jsonl`（16 个开发文档常用词）。

## 用真实语料实测（`tools/imme_stats.py`）

不要靠感觉调 `ROUTER_*` / `SHORTLIST_*`——喂它一段真实语料，直接看判定分布与延迟：

```bash
# 只看判定（零前向，最快）
python -m tools.imme_stats --corpus segments.jsonl --dry-run --fast-url http://127.0.0.1:3000

# 真跑：主模型 + 本机 marian-edge 作快通道，两遍（第二遍看缓存命中）
python -m tools.imme_stats --corpus segments.jsonl --model-dir /tmp/hymt2b4 \
    --fast-url http://127.0.0.1:3000 --glossary data/glossary.example.jsonl \
    --per-category-limit 40 --passes 2 --dump predictions.jsonl
```

输入是 JSONL：`{"text": "...", "src": "en", "tgt": "zh", "category": "docs_prose"}`。

在一份**真实语料**（从本机各仓库里抽出的 488 段：英文文档正文 220、短标签 70、
katago-cloud 的真实日文文案 90、真实中文文案 60、真实代码/标识符 23、重复 UI 25）上，
Hy-MT2-1.8B-4bit + marian-edge(31M) 快通道，取每类前 40 段的实测结果：

| 类别 | n | passthrough | cache | fast | main | 不进主模型 | ms/段 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 英文文档正文 (en→zh) | 40 | 0 | 0 | 40 | 0 | 100% | 18 |
| 英文短标签 (en→zh) | 40 | 17 | 0 | 23 | 0 | 100% | 9 |
| 日文真实文案 (ja→zh) | 40 | 0 | 0 | 0 | 40 | 0% | 144 |
| 中文真实文案 (zh→en) | 40 | 0 | 0 | 0 | 40 | 0% | 181 |
| 代码/标识符 | 23 | 21 | 0 | 2 | 0 | 100% | 3 |
| 重复 UI | 25 | 11 | 14 | 0 | 0 | 100% | 0 |
| **合计（冷启动）** | **208** | 49 | 14 | 65 | 80 | **62%** | 18 |
| **同一批第二次请求** | 208 | 49 | 159 | 0 | 0 | **100%** | 0 |

要点：

- 英文文档正文全部落在 31M 快通道，**18ms/段 vs 主模型 ~150ms**，约 8×；
- 真实日文/中文文案（快通道只配了 en→zh）走主模型；
- 冷启动 0 质量问题、0 次重试、0 次升级；第二遍 100% 缓存命中；
- 加上 `data/glossary.example.jsonl` 后，63/338 英文段命中术语（约 19%），这些段被
  **强制拉回主模型**以遵守术语，整体“不进主模型”从 62% 降到 57%，换来术语一致。

术语一致性（46 句真实文档，同一术语在不同句子里出现的译法个数）：

| 术语 | 无术语表 | 加术语表 | 结果 |
|---|---|---|---|
| `agent` | {代理, 智能体} | {代理} | 收敛 |
| `review` | {审查, 审核, 查看} | {审核} | 收敛 |
| `client` | {客户, 客户端} | {客户端} | 收敛 |
| `schema` | {架构, 模式} | {数据结构, 架构} | 部分收敛 |
| `deploy` | {其他} | {部署, 其他} | 部分收敛 |
| 其余 11 个（model/rule/audit/metric/token/session/prompt/template/server/request/worker 等） | 本来就 1 种 | 不变 | 已一致 |

两个实测出来的注意点：

- 术语按**词**匹配，不会动标识符：`playable-agent`、`PLAYABLE_AGENT_METRICS_PORT`、
  `worker.py` 都不注入（早期用子串匹配时 `playable-agent` 被译成“可玩的代理”，已修）。
- 术语表只保证**一致**，不保证**更好**：把 `review` 定成“审核”后，
  “review visualizations”也会变成“审核可视化”，在该语境下未必优于“查看可视化”。
  词义分支多的词建议按语境分别建条目，或干脆不建。

## 并发/性能建议

- **沉浸式翻译**：尽量走 `POST /imme` 一次带多段（本服务会尝试把 `text_list` 合并成一次生成；失败自动回退）。
- **Apple Silicon + MLX**：优先保持 `UVICORN_WORKERS=1`，再按需尝试 `MODEL_MAX_CONCURRENCY=2`（过大通常只会更慢）。
- **Docker/CPU（transformers）**：更优先调大 `UVICORN_WORKERS`，再按需调 `MODEL_MAX_CONCURRENCY`。
- **想要最低延迟**：让 `llama-server` 跑 `Hy-MT2-1.8B-Q4_K_M.gguf`（`-ngl 99` Metal），
  用 `BACKEND=http` + `MODEL_HTTP_BASE_URL` 指过去；再把 `ROUTER_FAST_URL` 指到一个
  31M 的 `marian-edge`（其 `/imme`）做快通道，短句先走 31M，长难句才升到 1.8B。

同一台 M1 Pro（16 GB）实测（同 prompt、temp 0、单段；`llama.cpp` 为 `llama-server` + `-ngl 99`）：

| 引擎 | 日文 17 字 | 日文 220 字 | 解码 |
|---|---:|---:|---:|
| Hy-MT2-1.8B 4bit MLX | 292 ms | 651 ms | 27–45 tok/s |
| Hy-MT2-1.8B Q4_K_M llama.cpp | 134 ms | 465 ms | 57–72 tok/s |
| （对照）marian-edge 31M | 33 ms | 162 ms | — |

## API（兼容 LinguaSpark/server）

- `POST /translate`：`{ "text": "...", "from": "en"(可选/auto), "to": "zh" }`
- `POST /detect`：`{ "text": "..." }`
- `POST /imme`：沉浸式翻译：`{ "source_lang": "auto"(可选), "target_lang": "zh", "text_list": ["..."] }`
- `POST /kiss`：同 `/translate`
- `POST /hcfy`：划词翻译兼容
- `POST /deeplx`：DeepLX 兼容
- `GET /health`
- `GET /stats`：路由计数与后端信息
- `POST /route`：只看路由判定，不做任何前向
- `POST /shortlist` / `POST /glossary`：追加翻译记忆 / 术语（可持久化到 JSONL）

`/translate` 与 `/imme` 的响应**新增可选字段**（`engine` / `reason` / `issues` / `similarity`），
旧的插件客户端可以忽略；`ROUTER_ENABLED=0` 时这些字段为 `null`，行为与旧版一致。

## 给沉浸式翻译插件用

把服务启动在本机后，一般填：

- Base URL：`http://127.0.0.1:3000`
- 端点：使用 `POST /imme`
- 如开启 `API_KEY`：在插件里配置请求头 `Authorization: Bearer <API_KEY>`（或用 `?token=` 方式）
