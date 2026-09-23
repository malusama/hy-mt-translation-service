#!/usr/bin/env python3
"""Benchmark any OpenAI-compatible endpoint on the ``/imme`` corpus.

Written for the vllm-metal probe (see ``docs/vllm-metal-20260923.md``): it posts
the *same* prompt the service's ``legacy`` style builds (imported from
``hy_mt_mlx_server.prompts`` so the wording cannot drift), one request per line,
and reports wall time / ms per line / lines per minute at a few concurrency
levels. Each concurrency level takes a disjoint slice of the corpus, so the
result is not confounded by the router's exact-match cache.

    python -m tools.make_imme_corpus                 # writes /tmp/imme_corpus.jsonl
    python tools/bench_vllm_metal.py --base-url http://127.0.0.1:8000

Stdlib only, no aiohttp/httpx: this has to run next to a model server on a
16 GB machine that is already swapping.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))

from hy_mt_mlx_server.lang import target_language_name  # noqa: E402
from hy_mt_mlx_server.prompts import build_translation_prompt  # noqa: E402


def post_json(url, payload, timeout, api_key=''):
    headers = {'Content-Type': 'application/json'}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'
    request = urllib.request.Request(url, data=json.dumps(payload, ensure_ascii=False).encode(), headers=headers)
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.load(response)
    return time.perf_counter() - started, data


def load_corpus(path, src):
    rows = [json.loads(line) for line in pathlib.Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    picked = [row['text'] for row in rows if str(row.get('src', '')).lower() == src.lower()]
    if not picked:
        raise SystemExit(f'{path} 里没有 src={src} 的段落')
    return picked


def main():
    parser = argparse.ArgumentParser(description='Benchmark an OpenAI-compatible /v1/chat/completions endpoint.')
    parser.add_argument('--base-url', default='http://127.0.0.1:8000')
    parser.add_argument('--model', default='', help='默认取 /v1/models 的第一个')
    parser.add_argument('--corpus', default='/tmp/imme_corpus.jsonl')
    parser.add_argument('--src', default='ja')
    parser.add_argument('--tgt', default='zh')
    parser.add_argument('--style', default='legacy', choices=('legacy', 'official'))
    parser.add_argument('--concurrency', default='1,8,24', help='逗号分隔，每个值用一段互不重叠的语料')
    parser.add_argument('--limit', type=int, default=30, help='每个并发档位用多少条')
    parser.add_argument('--max-tokens', type=int, default=1024)
    parser.add_argument('--timeout', type=float, default=300)
    parser.add_argument('--api-key', default='')
    parser.add_argument('--samples', type=int, default=3, help='每档位打印几条原文->译文')
    args = parser.parse_args()

    base = args.base_url.rstrip('/')
    model = args.model
    if not model:
        with urllib.request.urlopen(base + '/v1/models', timeout=30) as response:
            model = json.load(response)['data'][0]['id']
    lines = load_corpus(args.corpus, args.src)
    target = target_language_name(args.tgt)
    print(f'端点 {base}  模型 {model}')
    print(f'语料 {args.corpus}: {len(lines)} 条 {args.src}->{args.tgt}（target name "{target}"，style {args.style}）')

    def one(text):
        prompt = build_translation_prompt(text, target, style=args.style)
        payload = {'model': model, 'temperature': 0, 'max_tokens': args.max_tokens,
                   'messages': [{'role': 'user', 'content': prompt}]}
        elapsed, data = post_json(base + '/v1/chat/completions', payload, args.timeout, args.api_key)
        choice = data['choices'][0]
        return elapsed, choice['message']['content'].strip(), data.get('usage', {}), choice.get('finish_reason')

    cold, text, usage, finish = one(lines[0])
    print(f'冷/首个请求: {cold:.2f}s -> {text!r} (finish={finish}, usage={usage})')

    offset = 1
    for level in [int(part) for part in args.concurrency.split(',') if part.strip()]:
        block = lines[offset:offset + args.limit]
        offset += len(block)
        if not block:
            print(f'并发 {level}: 语料不够，跳过')
            continue
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=level) as pool:
            out = list(pool.map(one, block))
        wall = time.perf_counter() - started
        per_line = wall / len(block)
        latencies = [item[0] for item in out]
        print(f'并发 {level:>2}: {len(block)} 条 | 墙钟 {wall:.2f}s | 每条 {per_line * 1000:.0f} ms '
              f'| 单请求 p50 {statistics.median(latencies) * 1000:.0f} ms / max {max(latencies) * 1000:.0f} ms '
              f'| {len(block) / wall * 60:.0f} 条/分')
        for source, (_, translation, _, _) in list(zip(block, out))[:args.samples]:
            print(f'      {source}  ->  {translation}')


if __name__ == '__main__':
    main()
