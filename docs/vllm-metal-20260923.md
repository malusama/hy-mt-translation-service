# vllm-metal 后端实测：Hy-MT2-1.8B-4bit 能跑，但慢到不可用（2026-09-23）

目的：vLLM 官方的 Apple Silicon 插件 [vllm-metal](https://github.com/vllm-project/vllm-metal)
能不能当本服务的替代推理后端，把 `/imme` 接过去（替代现在的 `BACKEND=mlx` / `BACKEND=http`）。

结论先说：**能起来、译文也对，但单条 8–20 秒，比同一台机器上的 `mlx-lm` 慢约 60–150 倍，现在不可用。**
慢的原因**尚未查清**（见「为什么慢：待查假设」），本文把过程、数据、证据和下一步都留在下面。

## 环境与版本

- 机器：MacBook Pro（Apple M1 Pro，10 核 CPU / 16 核 GPU，16 GB 内存），macOS 27.0（26A428）
- 安装：`curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash`
  → 建在 `~/.venv-vllm-metal`（独立 venv，不需要编译器）
- 版本（本次实际装到的）
  - `vllm` 0.30.0+cpu（`vllm-0.30.0+cpu-cp312-cp312-macosx_11_0_arm64.whl`）
  - `vllm-metal` 0.30.0.dev20260923050831（cp312 / macosx_15_0_arm64 wheel）
  - `mlx` 0.32.1、`mlx-lm` 0.32.0（`git+https://github.com/ml-explore/mlx-lm@9e6acca`）、`transformers` 5.17.0
- 模型：`mlx-community/Hy-MT2-1.8B-4bit`
  - 架构 `HunYuanDenseV1ForCausalLM` / `model_type: hunyuan_v1_dense`，GQA（16 头 / 4 KV 头）+ `use_qk_norm: true`
  - MLX 4-bit affine、group size 64；`model.safetensors` 1,007,776,125 B（≈1.0 GB），sha256 前缀 `a9cecbc77050b86b`（与 HF 索引一致）
- 为什么认为这条路"应该在支持范围内"：vllm-metal 的
  [`docs/supported_models.md`](https://github.com/vllm-project/vllm-metal/blob/main/docs/supported_models.md)
  里有 `| Hunyuan (dense) | ✅ | GQA + QK norm (paged) | ✅ | mlx-community/Hunyuan-1.8B-Instruct-4bit |`，
  也就是 Hunyuan dense + MLX 4-bit 量化是官方示例过的组合。

## 启动

```bash
export PATH="$HOME/.venv-vllm-metal/bin:$PATH"
vllm serve mlx-community/Hy-MT2-1.8B-4bit --port 8000 --max-model-len 4096
```

服务能正常起来：`GET /v1/models` 列出 `mlx-community/Hy-MT2-1.8B-4bit`，`/v1/chat/completions` 可用。

## 结果

### 1) 功能正确

prompt 用的是本服务的 `legacy` 措辞（`Translate the following segment into Chinese, without additional explanation.\n\n{text}`），
`temperature=0`：

```json
{"messages":[{"role":"user","content":"Translate the following segment into Chinese, without additional explanation.\n\nログイン"}]}
→ {"content":"登录","finish_reason":"stop","usage":{"prompt_tokens":17,"completion_tokens":2}}
```

### 2) 延迟不可用（同一进程、连续 warm 请求）

每条只输出 2 个 token，却要 8–20 秒（单位：秒）：

| 请求 | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| 单条 `/v1/chat/completions` | 20.09 | 20.46 | 19.54 | 12.90 | 8.72 |

（另有前序一次 20.74 s。请求之间没有出现"编译一次就变快"的稳定态。）

### 3) 与同一台机器上的 `mlx-lm` 对比

同一台 M1 Pro、同一批语料（90 条真实日文，见下），本服务 MLX 后端的 `batch_generate`：

| 批量 | 每条耗时 | 吞吐 |
|---|---:|---:|
| 批 1（逐条） | ≈240 ms | ~250 行/分 |
| 批 8 | 130–140 ms | ~430 行/分 |
| 批 24 | ≈110 ms | ~545 行/分 |
| 长句（>25 字） | 0.4–1.3 s | — |

→ vllm-metal 这条路径比 `mlx-lm` **慢约 60–150 倍**。

## 为什么慢：待查假设（未解决）

已排除/存疑的证据，先说清楚哪些**不能**当结论：

- 慢在"每请求固定开销"而不是解码：输出只有 2 个 token，5 次请求之间没有随次数下降。
- `sample <pid> 6` 抓到的栈是**空闲**事件循环（`uvloop … kevent` + tokenizers 的 rayon 线程池等待），
  对 Call graph 逐帧统计只有 `python3.12`(427) / `pthread`(61) / `tokenizers`(60) / `zmq`(9) 帧，
  **没有任何 torch / mlx 帧** —— 这次采样没有采到计算路径，不构成证据。
- 早先一次粗粒度 grep 整个采样文件得到 `libtorch` 14 帧 / `libmlx` 2 帧，但把 `Binary Images` 段也算进去了，
  不严谨，**不作为"落到 CPU/torch"的结论**。

待验证的假设（建议按序试）：

1. Hunyuan dense 这条路没有吃到 paged Metal 内核，退化成 SDPA/eager，或部分层走了 `vllm_metal/pytorch_backend/` 的 torch 桥。
2. MLX affine 4-bit 权重在每个请求里被重打包/反量化。
3. `--max-model-len 4096` 之外的 KV/调度配置（默认 KV 分配、chunked prefill 等）。
4. 该 arch 的 kernel 覆盖不全，每次按新 shape JIT。

## 回家后的排查步骤

1. 起服务把日志落盘，先看它选了哪个 runner / attention backend，以及有没有 fallback / dequant 类警告：

   ```bash
   export PATH="$HOME/.venv-vllm-metal/bin:$PATH"
   vllm serve mlx-community/Hy-MT2-1.8B-4bit --port 8000 --max-model-len 4096 2>&1 | tee /tmp/vllm-metal.log
   ```

2. 用官方示例 checkpoint 跑同一段：`mlx-community/Hunyuan-1.8B-Instruct-4bit`。
   它也慢 → 是 arch/后端问题；它不慢 → 是 Hy-MT2 权重或配置问题。
3. 同一 venv 里直接 `mlx_lm.generate` 加载 Hy-MT2-4bit，确认机器本身没问题（应为 ~0.1 s/行）。
4. 试开关：`--enforce-eager`、`--max-model-len 1024`、量化相关参数；`vllm_metal/envs.py` 里列了可开关的环境变量，逐个看一遍。
5. 拿到日志后再决定是否开 issue —— `supported_models.md` 明确要求"模型跑不通请开 issue，不要自行加行/加例"。

## 复现

本次的压测脚本随仓库一起提交：`tools/bench_vllm_metal.py`（对任意 OpenAI 兼容端点跑同一份 `/imme` 语料，
并发 1 / 8 / 24，打印每条耗时与吞吐；语料用 `tools/make_imme_corpus.py` 生成，其中 90 条是真实日文）。

```bash
# 1) 造语料（488 段，含 C_ja_real_copy 90 条真实日文）
python -m tools.make_imme_corpus
# 2) 压测一个已经起来的 vLLM / 其它 OpenAI 兼容端点
python tools/bench_vllm_metal.py --base-url http://127.0.0.1:8000 --model mlx-community/Hy-MT2-1.8B-4bit
```

## 环境副作用（记录用）

- 安装占用：`~/.venv-vllm-metal` ≈ 1.6 GB；模型 ≈ 1.0 GB（HF 缓存）。
- 跑的时候这台 16 GB 机器很吃紧：实验期间 1 分钟负载峰值到 70，`vm.swapusage` 已用到 8.5 / 9.2 GB；
  WindowServer、浏览器、IM 也在抢 CPU，所以"整机卡"不只来自这次实验。
- 实验结束已把 `vllm serve` 与压测进程全部停掉，8000 端口释放。
