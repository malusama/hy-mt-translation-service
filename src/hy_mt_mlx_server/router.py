"""Request routing for the translation pipeline.

The router is deliberately a *decision* layer: it answers "does this segment
need the 1.8B model at all, and if so which engine" before any forward pass.

Order of decisions per segment (first hit wins):

1. ``symbol``  - no letters (numbers, punctuation, emoji, separators) -> echo;
2. ``identity``- the segment already reads as the target language -> echo;
3. ``cache``   - exact-match LRU over normalised (source, target, text);
4. ``memory``  - shortlist/translation-memory near-hit that passes the quality
   gate -> reuse without a forward pass;
5. ``fast``    - cascade fast path (e.g. the 31M en->zh student behind
   ``ROUTER_FAST_URL``), only for configured language pairs and short segments;
6. ``main``    - the configured HY-MT2 backend, optionally with glossary /
   reference-translation injection from the shortlist.

Fast-path output is quality-gated; a failed gate escalates the segment to the
main engine instead of shipping it. Main-engine output is gated too, and a
hard failure (echo / wrong script / empty) triggers one retry with the official
Hy-MT2 prompt wording. Only gate-passing translations enter the cache.
"""

from __future__ import annotations

import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from .lang import target_language_name
from .model import GenerationParams, TranslationModel
from .prompts import build_translation_prompt, normalize_style
from .quality import check_translation
from .content import has_translatable_prose, looks_like_code, looks_like_command
from .script import (
    distinct_language,
    guess_language_by_script,
    is_symbol_or_number_only,
)
from .shortlist import Glossary, ShortlistStore, rank_candidates

_HARD_ISSUES = {"empty", "echo", "missing_target_script", "repetition"}


def _norm_lang(code: Optional[str]) -> str:
    if not code:
        return ""
    return code.strip().lower().split("-")[0].split("_")[0]


def _cache_key(source_lang: Optional[str], target_lang: str, text: str) -> tuple[str, str, str]:
    return (_norm_lang(source_lang), _norm_lang(target_lang), re.sub(r"\s+", " ", text.strip()))


@dataclass
class SegmentResult:
    text: str
    source_lang: str
    target_lang: str
    engine: str
    reason: str = ""
    issues: list[str] = field(default_factory=list)
    similarity: Optional[float] = None
    latency_ms: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "detected_source_lang": self.source_lang,
            "text": self.text,
            "engine": self.engine,
        }
        if self.reason:
            payload["reason"] = self.reason
        if self.issues:
            payload["issues"] = self.issues
        if self.similarity is not None:
            payload["similarity"] = round(self.similarity, 4)
        return payload

    def route_dict(self, index: int | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "engine": self.engine,
            "reason": self.reason,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
        }
        if index is not None:
            payload["index"] = index
        if self.similarity is not None:
            payload["similarity"] = round(self.similarity, 4)
        if self.issues:
            payload["predicted_issues"] = self.issues
        return payload


@dataclass
class RouterConfig:
    enabled: bool = True
    skip_symbol_only: bool = True
    skip_untranslatable: bool = True
    skip_same_language: bool = True
    cache_size: int = 4096
    retry_on_hard_issues: bool = True
    prompt_style: str = "legacy"
    max_input_chars: int = 2000
    # fast path / cascade
    fast_enabled: bool = False
    fast_max_chars: int = 240
    # shortlist
    shortlist_top_k: int = 5
    reuse_threshold: float = 0.97
    reference_threshold: float = 0.80
    reference_limit: int = 1
    glossary_limit: int = 8
    rerank: str = "off"  # off | model


@dataclass
class _Plan:
    index: int
    text: str
    source_lang: str
    decision: str
    reason: str = ""
    result_text: str = ""
    issues: list[str] = field(default_factory=list)
    similarity: Optional[float] = None
    chunks: list[str] = field(default_factory=list)
    glossary: list[tuple[str, str]] = field(default_factory=list)
    reference: Optional[str] = None
    # (stored source, stored target, embedding score) awaiting a model rerank
    rerank_candidates: list[tuple[str, str, float]] = field(default_factory=list)


class Router:
    def __init__(
        self,
        model: TranslationModel,
        *,
        config: Optional[RouterConfig] = None,
        fast_backend: Any = None,
        shortlist: Optional[ShortlistStore] = None,
        glossary: Optional[Glossary] = None,
    ) -> None:
        self.model = model
        self.config = config or RouterConfig()
        self.fast_backend = fast_backend
        self.shortlist = shortlist
        self.glossary = glossary
        self._cache: "OrderedDict[tuple[str, str, str], str]" = OrderedDict()
        self._stats: dict[str, int] = {
            "segments": 0,
            "symbol": 0,
            "passthrough": 0,
            "identity": 0,
            "cache": 0,
            "memory": 0,
            "fast": 0,
            "fast_escalated": 0,
            "main": 0,
            "reference_injected": 0,
            "retried": 0,
            "chars": 0,
        }

    # ------------------------------------------------------------------ stats
    @property
    def stats(self) -> dict[str, Any]:
        total = max(1, self._stats["segments"])
        return {
            **self._stats,
            "cache_entries": len(self._cache),
            "shortlist_entries": len(self.shortlist) if self.shortlist else 0,
            "glossary_entries": len(self.glossary) if self.glossary else 0,
            # Share of segments that never touched the main (1.8B) model.
            "no_main_model_ratio": round(
                (
                    self._stats["symbol"]
                    + self._stats["passthrough"]
                    + self._stats["identity"]
                    + self._stats["cache"]
                    + self._stats["memory"]
                    + self._stats["fast"]
                )
                / total,
                4,
            ),
        }

    def _bump(self, **increments: int) -> None:
        # Single-threaded event loop: plain increments are enough, and keeping
        # this synchronous avoids "coroutine was never awaited" traps at call
        # sites that are not on the hot path.
        for key, value in increments.items():
            self._stats[key] = self._stats.get(key, 0) + value

    # ------------------------------------------------------------------ cache
    def _cache_get(self, key: tuple[str, str, str]) -> Optional[str]:
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def _cache_put(self, key: tuple[str, str, str], value: str) -> None:
        if self.config.cache_size <= 0 or not value:
            return
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > self.config.cache_size:
            self._cache.popitem(last=False)

    # ------------------------------------------------------------------ planning
    def _chunks(self, text: str) -> list[str]:
        from .imme import split_long_text

        limit = max(1, int(self.config.max_input_chars))
        return split_long_text(text, limit)

    def _plan(
        self,
        index: int,
        text: str,
        source_lang: Optional[str],
        target_lang: str,
    ) -> _Plan:
        resolved_source = _norm_lang(source_lang) or guess_language_by_script(text)
        plan = _Plan(index=index, text=text, source_lang=resolved_source, decision="main")
        cfg = self.config

        if not text.strip():
            plan.decision, plan.reason, plan.result_text = "identity", "empty input", text
            return plan

        if cfg.skip_symbol_only and is_symbol_or_number_only(text):
            plan.decision, plan.reason, plan.result_text = "symbol", "no letters to translate", text
            return plan

        # Identifiers, code, versions and bare URLs come back unchanged from any
        # engine, so do not spend a forward pass on them (and do not let the
        # "echo" gate fire on the correct answer).
        if cfg.skip_untranslatable and not has_translatable_prose(text):
            plan.decision, plan.reason, plan.result_text = "passthrough", "no translatable content", text
            return plan

        if cfg.skip_same_language and resolved_source and _norm_lang(resolved_source) == _norm_lang(target_lang):
            # ...unless the segment is plainly written in another language's
            # script (e.g. a client declaring en->en for a Chinese page).
            distinct = distinct_language(text)
            if not distinct or distinct == _norm_lang(target_lang):
                plan.decision, plan.reason, plan.result_text = "identity", "already target language", text
                return plan

        # Only trust the script when it is decisive (see distinct_language): a
        # latin page is never assumed to already be the latin target.
        if cfg.skip_same_language and not source_lang and distinct_language(text) == _norm_lang(target_lang):
            plan.decision, plan.reason, plan.result_text = "identity", "input already in target language", text
            return plan

        cached = self._cache_get(_cache_key(resolved_source, target_lang, text))
        if cached is not None:
            plan.decision, plan.reason, plan.result_text = "cache", "exact cache hit", cached
            return plan

        # Never inject terminology into code/commands: it turns `npx wrangler
        # deploy` into "npx wrangler 部署".
        if self.glossary is not None and not (looks_like_code(text) or looks_like_command(text)):
            plan.glossary = self.glossary.terms_for(
                text,
                source_lang=resolved_source,
                target_lang=target_lang,
                limit=cfg.glossary_limit,
            )

        if self.shortlist is not None and len(self.shortlist) > 0:
            matches = self.shortlist.query(
                text,
                source_lang=resolved_source,
                target_lang=target_lang,
                k=max(1, cfg.shortlist_top_k),
            )
            candidates = [m for m in matches if m.score >= cfg.reference_threshold]
            if candidates:
                best = candidates[0]
                plan.similarity = best.score
                if cfg.rerank == "model" and len(candidates) > 1:
                    # Ordering needs a forward pass, so it happens in the async
                    # phase; reuse/reference is decided there.
                    plan.rerank_candidates = [(m.entry.source, m.entry.target, m.score) for m in candidates]
                elif best.score >= cfg.reuse_threshold and self._memory_reuse_ok(
                    best.entry.source, best.entry.target, text, target_lang
                ):
                    plan.decision = "memory"
                    plan.reason = f"translation memory hit ({best.score:.3f})"
                    plan.result_text = best.entry.target
                    return plan
                elif cfg.reference_limit > 0:
                    plan.reference = best.entry.target
                    plan.reason = f"reference translation attached ({best.score:.3f})"

        plan.chunks = self._chunks(text)
        return plan

    async def _apply_rerank(self, plan: _Plan, target_lang: str) -> None:
        """Score the shortlisted candidates in one forward pass each, then
        decide reuse vs. reference on the winner."""
        entries = plan.rerank_candidates
        scorer = getattr(self.model, "score_continuations", None)
        if scorer is not None and len(entries) > 1:
            prompt = build_translation_prompt(
                plan.text,
                target_language_name(target_lang),
                style=normalize_style(self.config.prompt_style),
                glossary=plan.glossary,
            )
            targets = [target for _source, target, _score in entries]
            try:
                scores = list(await scorer(prompt, targets))
                by_target = dict(zip(targets, scores))
                ranked = rank_candidates(lambda _prompt, cs: [by_target[c] for c in cs], prompt, targets)
                order = {target: rank for rank, (target, _s) in enumerate(ranked)}
                entries = sorted(entries, key=lambda item: order.get(item[1], len(order)))
            except Exception:
                pass  # keep the embedding order on any scoring failure

        source, target, embed_score = entries[0]
        if embed_score >= self.config.reuse_threshold and self._memory_reuse_ok(
            source, target, plan.text, target_lang
        ):
            plan.decision = "memory"
            plan.reason = f"translation memory hit ({embed_score:.3f}, model-reranked)"
            plan.result_text = target
            return
        if self.config.reference_limit > 0:
            plan.reference = target
            plan.reason = f"reference translation attached ({embed_score:.3f}, model-reranked)"

    @staticmethod
    def _memory_reuse_ok(stored_source: str, stored_target: str, source: str, target_lang: str) -> bool:
        """Reuse only when both the stored pair and the new pair pass the gate."""
        if not check_translation(stored_source, stored_target, target_lang).ok:
            return False
        # The stored target must also still look right for the *new* source: the
        # two are near-identical by construction, but numbers/URLs may differ.
        return check_translation(source, stored_target, target_lang).ok

    # ------------------------------------------------------------------ main path
    async def _run_main(self, plans: list[_Plan], target_lang: str, params: GenerationParams) -> None:
        """Translate the pending plans with the main engine, in one batch when possible."""
        if not plans:
            return
        style = normalize_style(self.config.prompt_style)
        lang_name = target_language_name(target_lang)

        referenced = [p for p in plans if p.glossary or p.reference]
        plain = [p for p in plans if not (p.glossary or p.reference)]
        self._bump(reference_injected=len(referenced))

        submit_prompts = getattr(self.model, "translate_prompts", None)

        async def _plain(results: dict[int, str]) -> None:
            if not plain:
                return
            if len(plain) == 1:
                results[plain[0].index] = await self.model.translate(
                    text=plain[0].text, to_lang=target_lang, params=params
                )
                return
            texts = [p.text for p in plain]
            # Use the backend's own batch entry point: the MLX backend packs the
            # whole list into one JSON-array generation, which measured ~1.6x
            # faster per segment than one prompt per segment.
            outputs = list(await self.model.translate_many(texts, target_lang, params))
            for plan, out in zip(plain, outputs):
                results[plan.index] = str(out)

        async def _referenced(results: dict[int, str]) -> None:
            for plan in referenced:
                terms = list(plan.glossary)
                if plan.reference and not terms:
                    # A TM reference is passed as a *terminology* hint: the
                    # official Hy-MT2 template, which the model treats as
                    # advisory rather than as text to copy.
                    terms = _reference_terms(plan.text, plan.reference)
                prompt = build_translation_prompt(plan.text, lang_name, style=style, glossary=terms)
                if submit_prompts is not None:
                    out = (await submit_prompts([prompt], params))[0]
                else:  # pragma: no cover
                    out = await self.model.translate(text=plan.text, to_lang=target_lang, params=params)
                results[plan.index] = out

        results: dict[int, str] = {}
        await _plain(results)
        await _referenced(results)

        for plan in plans:
            plan.result_text = results.get(plan.index, "")

    # ------------------------------------------------------------------ fast path
    def _fast_eligible(self, plan: _Plan, source_lang: Optional[str], target_lang: str) -> bool:
        """Would the fast backend be asked to serve this segment?"""
        if not self.config.fast_enabled or self.fast_backend is None:
            return False
        if plan.glossary or plan.reference:
            # Terminology is a hard requirement and only the main engine gets the
            # reference block; a 31M student would silently ignore it.
            return False
        if len(plan.chunks) != 1 or len(plan.text) > self.config.fast_max_chars:
            return False
        return bool(self.fast_backend.supports_pair(plan.source_lang or source_lang, target_lang))

    async def _run_fast(self, plans: list[_Plan], source_lang: Optional[str], target_lang: str) -> list[_Plan]:
        """Fill plans from the fast backend; return the ones that must escalate."""
        fast = self.fast_backend
        if fast is None or not plans:
            return plans
        eligible: list[_Plan] = []
        escalate: list[_Plan] = []
        for plan in plans:
            (eligible if self._fast_eligible(plan, source_lang, target_lang) else escalate).append(plan)

        # The fast engine is asked one language pair at a time: a kana segment
        # resolves to ja while a latin one resolves to en, and mixing them in a
        # single /imme call would silently mistranslate one side.
        groups: dict[str, list[_Plan]] = {}
        for plan in eligible:
            groups.setdefault(plan.source_lang or _norm_lang(source_lang), []).append(plan)

        for group_source, group in groups.items():
            try:
                outputs = await fast.translate_list(
                    [p.text for p in group],
                    source_lang=group_source,
                    target_lang=_norm_lang(target_lang),
                )
            except Exception as exc:
                for plan in group:
                    plan.reason = f"fast path unavailable: {type(exc).__name__}"
                escalate.extend(group)
                continue

            self._bump(fast=len(group))
            retry: list[_Plan] = []
            for plan, out in zip(group, outputs):
                report = check_translation(plan.text, out, target_lang)
                # The fast path is a bonus, not a contract: anything the gate
                # dislikes (not just hard failures) goes to the main engine.
                if report.ok:
                    plan.decision = "fast"
                    plan.result_text = out
                else:
                    plan.issues = report.issues
                    plan.reason = f"fast path failed gate: {','.join(report.issues)}"
                    retry.append(plan)
            if retry:
                self._bump(fast_escalated=len(retry))
            escalate = retry + escalate
        return escalate

    # ------------------------------------------------------------------ public API
    async def translate_many(
        self,
        texts: Sequence[str],
        *,
        source_lang: Optional[str],
        target_lang: str,
        params: GenerationParams,
    ) -> list[SegmentResult]:
        started = time.perf_counter()
        plans = [self._plan(i, str(t), source_lang, target_lang) for i, t in enumerate(texts)]

        for plan in plans:
            if plan.decision == "main" and plan.rerank_candidates:
                await self._apply_rerank(plan, target_lang)

        resolved = [
            p for p in plans if p.decision == "main" and self.config.enabled
        ]
        if self.config.fast_enabled:
            resolved = await self._run_fast(resolved, source_lang, target_lang)
        if resolved:
            await self._run_main(resolved, target_lang, params)

        results: list[SegmentResult] = []
        for plan in plans:
            if plan.decision == "main":
                report = check_translation(plan.text, plan.result_text, target_lang)
                plan.issues = report.issues
                if plan.issues and self.config.retry_on_hard_issues and set(plan.issues) & _HARD_ISSUES:
                    retried = await self._retry(plan, target_lang, params)
                    if retried is not None:
                        plan.result_text = retried
                        report = check_translation(plan.text, plan.result_text, target_lang)
                        plan.issues = report.issues
                        self._bump(retried=1)
                if report.ok and plan.result_text:
                    self._cache_put(_cache_key(plan.source_lang, target_lang, plan.text), plan.result_text)
                self._bump(main=1)
            elif plan.decision == "fast":
                # Cache fast-path hits only when the stored source/target pair is
                # stable enough that a later exact hit is safe.
                self._cache_put(_cache_key(plan.source_lang, target_lang, plan.text), plan.result_text)
            results.append(
                SegmentResult(
                    text=plan.result_text,
                    source_lang=plan.source_lang,
                    target_lang=_norm_lang(target_lang),
                    engine=plan.decision,
                    reason=plan.reason,
                    issues=plan.issues,
                    similarity=plan.similarity,
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                )
            )

        self._bump(
            segments=len(texts),
            symbol=sum(1 for p in plans if p.decision == "symbol"),
            passthrough=sum(1 for p in plans if p.decision == "passthrough"),
            identity=sum(1 for p in plans if p.decision == "identity"),
            cache=sum(1 for p in plans if p.decision == "cache"),
            memory=sum(1 for p in plans if p.decision == "memory"),
            chars=sum(len(t) for t in texts),
        )
        return results

    async def _retry(self, plan: _Plan, target_lang: str, params: GenerationParams) -> Optional[str]:
        """One retry with the official prompt wording for hard failures."""
        style = normalize_style(self.config.prompt_style)
        other = "official" if style == "legacy" else "legacy"
        prompt = build_translation_prompt(
            plan.text,
            target_language_name(target_lang),
            style=other,
            glossary=plan.glossary,
        )
        submit = getattr(self.model, "translate_prompts", None)
        try:
            if submit is not None:
                out = (await submit([prompt], params))[0]
            else:  # pragma: no cover
                out = await self.model.translate(text=plan.text, to_lang=target_lang, params=params)
        except Exception:
            return None
        out = str(out).strip()
        if not out:
            return None
        if check_translation(plan.text, out, target_lang).ok:
            plan.reason = (plan.reason + "; " if plan.reason else "") + "recovered on retry"
            return out
        return None

    def route_only(
        self,
        texts: Sequence[str],
        *,
        source_lang: Optional[str],
        target_lang: str,
    ) -> list[dict[str, Any]]:
        """Decide without running any forward pass (Laya's ``Router.route``).

        For a segment that would go to the main engine, the reported engine is
        the *predicted* one: ``fast`` when the cascade would try the fast
        backend first. The real engine can still end up as ``main`` if the fast
        output fails the quality gate and escalates.
        """
        out: list[dict[str, Any]] = []
        for i, text in enumerate(texts):
            plan = self._plan(i, str(text), source_lang, target_lang)
            engine = plan.decision
            reason = plan.reason
            if engine == "main" and self._fast_eligible(plan, source_lang, target_lang):
                engine = "fast"
                reason = (reason + "; " if reason else "") + "fast path eligible"
            out.append(
                {
                    "index": i,
                    "engine": engine,
                    "reason": reason,
                    "chunks": len(plan.chunks) or 1,
                    "source_lang": plan.source_lang,
                    "target_lang": _norm_lang(target_lang),
                    "glossary_terms": [s for s, _ in plan.glossary],
                    "reference_attached": bool(plan.reference),
                    "similarity": round(plan.similarity, 4) if plan.similarity is not None else None,
                }
            )
        return out

    def close(self) -> None:
        if self.fast_backend is not None and hasattr(self.fast_backend, "close"):
            self.fast_backend.close()


def _reference_terms(source: str, reference: str) -> list[tuple[str, str]]:
    """Describe a TM reference as a terminology hint.

    The Hy-MT2 terminology template expects term pairs; using the whole
    segment keeps the reference advisory and avoids the model copying it
    verbatim when the new source differs slightly.
    """
    source = source.strip()
    reference = reference.strip()
    if not source or not reference:
        return []
    return [(source, reference)]
