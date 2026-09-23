#!/usr/bin/env python3
"""Measure what the router actually does to an immersive-translate workload.

Feed it a JSONL of *real* segments and it reports, per category, which engine
served each segment, how much of the traffic never touched the main model, and
the latency cost. Use it before tuning ROUTER_* / SHORTLIST_* thresholds.

Corpus format (one JSON object per line):

    {"text": "...", "src": "en", "tgt": "zh", "category": "docs_prose"}

`src` may be omitted (the router then infers it from the script).

Examples:

    # real service model + the local marian-edge as fast path
    python -m tools.imme_stats --corpus /tmp/imme_corpus.jsonl \
        --model-dir /tmp/hymt2b4 --fast-url http://127.0.0.1:3000 --passes 2

    # no model at all: only report the routing decisions (zero latency cost)
    python -m tools.imme_stats --corpus /tmp/imme_corpus.jsonl --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hy_mt_mlx_server.http_backend import (  # noqa: E402
    LinguaSparkFastBackend,
    parse_pairs,
)
from hy_mt_mlx_server.model import GenerationParams, HyMtMlxModel  # noqa: E402
from hy_mt_mlx_server.router import Router, RouterConfig  # noqa: E402
from hy_mt_mlx_server.settings import Settings  # noqa: E402
from hy_mt_mlx_server.shortlist import Glossary, ShortlistStore  # noqa: E402

ENGINES = ("symbol", "passthrough", "identity", "cache", "memory", "fast", "main")
NO_MAIN = {"symbol", "passthrough", "identity", "cache", "memory", "fast"}


def load_corpus(path: Path, per_category: int) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if per_category <= 0:
        return rows
    kept: list[dict] = []
    seen: Counter[str] = Counter()
    for row in rows:
        cat = str(row.get("category", "all"))
        if seen[cat] < per_category:
            kept.append(row)
            seen[cat] += 1
    return kept


def build_router(args: argparse.Namespace):
    settings = Settings(PRELOAD_MODEL=False, MODEL_ID=args.model_dir or "mlx-community/Hy-MT2-1.8B-4bit")
    config = RouterConfig(
        fast_enabled=bool(args.fast_url),
        fast_max_chars=args.fast_max_chars,
        shortlist_top_k=settings.shortlist_top_k,
        reuse_threshold=settings.shortlist_reuse,
        reference_threshold=settings.shortlist_reference,
    )
    shortlist = ShortlistStore.from_jsonl(args.shortlist) if args.shortlist else None
    glossary = Glossary.from_jsonl(args.glossary) if args.glossary else None
    fast = None
    if args.fast_url:
        fast = LinguaSparkFastBackend(args.fast_url, pairs=parse_pairs(args.fast_pairs), timeout_s=30.0)
    if args.dry_run:
        model = None
    else:
        # The model is loaded lazily *inside* the backend's own executor thread
        # (via ensure_loaded below): MLX binds its Metal stream to the thread
        # that loaded the weights, so loading here on the main thread would
        # break generation ("There is no Stream(gpu, 1) in current thread").
        model = HyMtMlxModel(settings.model_id, prompt_style=args.prompt_style)
    router = Router(model, config=config, fast_backend=fast, shortlist=shortlist, glossary=glossary)
    return router, model


async def run_pass(router: Router, rows: list[dict], batch: int, params: GenerationParams) -> list[dict]:
    by_pair: defaultdict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        by_pair[(str(row.get("src") or ""), str(row.get("tgt") or "zh"))].append(row)

    out: list[dict] = []
    for (src, tgt), group in by_pair.items():
        for i in range(0, len(group), batch):
            chunk = group[i : i + batch]
            t0 = time.perf_counter()
            results = await router.translate_many(
                [r["text"] for r in chunk],
                source_lang=src or None,
                target_lang=tgt,
                params=params,
            )
            per_seg_ms = (time.perf_counter() - t0) * 1000 / max(1, len(chunk))
            for row, result in zip(chunk, results):
                out.append(
                    {
                        "category": row.get("category", "all"),
                        "src": src,
                        "tgt": tgt,
                        "text": row["text"],
                        "engine": result.engine,
                        "output": result.text,
                        "issues": result.issues,
                        "ms": per_seg_ms,
                    }
                )
    return out


async def run_passes(args: argparse.Namespace, router: Router, model, rows: list[dict], params: GenerationParams) -> None:
    """All passes share one event loop: the backend's semaphore is loop-bound."""
    await model.ensure_loaded()
    passes = max(1, args.passes)
    for index in range(passes):
        started = time.perf_counter()
        predictions = await run_pass(router, rows, args.batch, params)
        wall = time.perf_counter() - started
        report(predictions, f"pass {index + 1} ({len(rows)} segments, {wall:.1f}s wall, {len(rows)/wall:.1f} seg/s)")
        print(f"router stats: {json.dumps(router.stats, ensure_ascii=False)}")
        if args.dump and index == passes - 1:
            args.dump.write_text(
                "\n".join(json.dumps(p, ensure_ascii=False) for p in predictions) + "\n", encoding="utf-8"
            )
            print(f"wrote per-segment dump -> {args.dump}")


def report(predictions: list[dict], title: str) -> None:
    print(f"\n=== {title} ===")
    header = f"{'category':<22}{'n':>5}  " + "".join(f"{e[:4]:>6}" for e in ENGINES) + f"{'no-main':>9}{'ms/seg':>9}"
    print(header)
    print("-" * len(header))
    by_cat: defaultdict[str, list[dict]] = defaultdict(list)
    for p in predictions:
        by_cat[p["category"]].append(p)
    for cat in sorted(by_cat):
        items = by_cat[cat]
        engines = Counter(i["engine"] for i in items)
        cells = "".join(f"{engines.get(e, 0):>6}" for e in ENGINES)
        no_main = sum(engines.get(e, 0) for e in NO_MAIN) / len(items)
        ms = statistics.median(i["ms"] for i in items)
        print(f"{cat:<22}{len(items):>5}  {cells}{no_main:>8.0%}{ms:>9.0f}")
    engines = Counter(i["engine"] for i in predictions)
    cells = "".join(f"{engines.get(e, 0):>6}" for e in ENGINES)
    no_main = sum(engines.get(e, 0) for e in NO_MAIN) / max(1, len(predictions))
    ms = statistics.median(i["ms"] for i in predictions)
    print("-" * len(header))
    print(f"{'TOTAL':<22}{len(predictions):>5}  {cells}{no_main:>8.0%}{ms:>9.0f}")
    issues = Counter(issue for p in predictions for issue in p["issues"])
    if issues:
        print(f"quality issues: {dict(issues)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--model-dir", default="", help="local HF/MLX model dir or repo id")
    parser.add_argument("--prompt-style", default="legacy", choices=["legacy", "official"])
    parser.add_argument("--fast-url", default="", help="LinguaSpark/hy-mt compatible /imme service")
    parser.add_argument("--fast-pairs", default="en:zh")
    parser.add_argument("--fast-max-chars", type=int, default=240)
    parser.add_argument("--shortlist", default="")
    parser.add_argument("--glossary", default="")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--per-category-limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="routing decisions only, no model")
    parser.add_argument("--dump", type=Path, default=None, help="write per-segment results to JSONL")
    args = parser.parse_args()

    rows = load_corpus(args.corpus, args.per_category_limit)
    router, model = build_router(args)

    if args.dry_run:
        # route_only takes one language pair per call, so group and map back by
        # row identity -- zipping the flattened decisions would misalign them.
        decisions: dict[int, dict] = {}
        for src, tgt, group in _group(rows):
            routed = router.route_only([r["text"] for r in group], source_lang=src or None, target_lang=tgt)
            for row, decision in zip(group, routed):
                decisions[id(row)] = decision
        predictions = [
            {
                "category": row.get("category", "all"),
                "src": str(row.get("src") or ""),
                "tgt": str(row.get("tgt") or "zh"),
                "text": row["text"],
                "engine": decisions[id(row)]["engine"],
                "output": "",
                "issues": [],
                "ms": 0.0,
            }
            for row in rows
        ]
        report(predictions, "decisions only (no forward pass)")
        return

    params = GenerationParams(max_new_tokens=1024, temperature=0.0, top_p=0.6, top_k=20, repetition_penalty=1.05)
    asyncio.run(run_passes(args, router, model, rows, params))
    if model is not None:
        model.close()
    router.close()


def _group(rows: list[dict]):
    by_pair: "OrderedDict[tuple[str, str], list[dict]]" = OrderedDict()
    for row in rows:
        by_pair.setdefault((str(row.get("src") or ""), str(row.get("tgt") or "zh")), []).append(row)
    for (src, tgt), group in by_pair.items():
        yield src, tgt, group


if __name__ == "__main__":
    main()
