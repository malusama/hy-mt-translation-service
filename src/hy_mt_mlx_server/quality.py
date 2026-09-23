"""Cheap, deterministic quality gates for a translated segment.

These checks exist so the router can *decide*, not so the service can claim a
COMET score.  Every signal here is observable in one segment pair and costs
microseconds, which is what a cascade needs: the fast engine's output must be
gated before we can trust it, and the main engine's output must be gated before
we cache it.

Signals:

* ``empty``            - nothing was produced;
* ``echo``             - output is (a copy of) the source;
* ``ratio_*``          - length ratio is far outside the band for the pair;
* ``missing_target_script`` - target language script absent from the output;
* ``lost_placeholder`` - a URL/code/format token from the source did not
  survive, which is the failure mode that actually breaks page rendering;
* ``repetition``       - degenerate repeated n-grams.
* ``possibly_truncated`` - the source was a finished sentence but the output
  stops without terminal punctuation (the 31M fast-path student truncates here;
  the ratio gate cannot see it because denser scripts halve the length).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .content import extract_tokens, has_translatable_prose, looks_like_code, word_count
from .script import expected_scripts, has_letters, scripts_in

_MIN_ECHO_CHARS = 8
_REPEAT_NGRAM = 5
_REPEAT_LIMIT = 3
_TERMINAL = ".。!！?？…"


@dataclass
class SegmentReport:
    """Result of the quality gate for one segment."""

    ok: bool
    issues: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {"ok": self.ok, "issues": self.issues}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _ratio_band(target_lang: str | None, source: str) -> tuple[float, float]:
    # CJK targets are far denser than their latin sources; latin targets the
    # other way round. Bands are intentionally loose -- this is a sanity gate,
    # not a length heuristic for MT.
    lang = (target_lang or "").lower()
    latin_target = lang not in {"zh", "zh-hant", "ja", "ko", "yue"}
    if latin_target and any(not ch.isascii() for ch in source if ch.isalpha()):
        # CJK source -> latin target expands a lot (定式讲解史 -> "Standard
        # Explanation of History"); the band has to allow that.
        return 0.1, 10.0
    if lang in {"zh", "zh-hant", "ja", "ko", "yue"}:
        return 0.08, 2.5
    return 0.2, 6.0


def check_translation(
    source: str,
    output: str,
    target_lang: str | None,
    *,
    check_placeholders: bool = True,
) -> SegmentReport:
    issues: list[str] = []
    out = output.strip()
    src = source.strip()

    if not out:
        return SegmentReport(ok=False, issues=["empty"])

    norm_src, norm_out = _normalize(src), _normalize(out)
    # An unchanged output is only suspicious for prose: identifiers, code,
    # commands and bare URLs are *supposed* to come back untouched.
    prose = has_translatable_prose(src)
    code_like = looks_like_code(src)
    if prose and not code_like:
        if (len(out) >= _MIN_ECHO_CHARS and norm_src == norm_out) or (
            len(out) >= 32 and norm_src and (norm_src in norm_out or norm_out in norm_src)
        ):
            issues.append("echo")

    # Ratios over very short sources are meaningless (5 chars -> 30 chars is a
    # normal expansion, not a runaway).
    if len(src) >= 8 and has_letters(out):
        lo, hi = _ratio_band(target_lang, src)
        ratio = len(out) / max(1, len(src))
        if ratio < lo:
            issues.append("ratio_low")
        elif ratio > hi:
            issues.append("ratio_high")

    expected = expected_scripts(target_lang)
    # A single word (brand, product name, loanword) may legitimately stay as it
    # is; only multi-word prose is expected to change script.
    if prose and word_count(src) >= 2 and expected and has_letters(out) and not (scripts_in(out) & set(expected)):
        issues.append("missing_target_script")

    if check_placeholders:
        src_tokens = extract_tokens(src)
        if src_tokens:
            out_cf = out.casefold()
            if any(tok.casefold() not in out_cf for tok in src_tokens):
                issues.append("lost_placeholder")

    if _has_degenerate_repetition(out):
        issues.append("repetition")

    if prose and _looks_truncated(src, out):
        issues.append("possibly_truncated")

    return SegmentReport(ok=not issues, issues=issues)


def _looks_truncated(source: str, output: str) -> bool:
    if len(output) < 12 or not source:
        return False
    if source.rstrip()[-1] not in _TERMINAL:
        return False
    return output.rstrip()[-1] not in _TERMINAL


def _has_degenerate_repetition(text: str, n: int = _REPEAT_NGRAM, limit: int = _REPEAT_LIMIT) -> bool:
    words = text.split()
    if len(words) < n * limit:
        return False
    seen: dict[tuple[str, ...], int] = {}
    for i in range(len(words) - n + 1):
        gram = tuple(words[i : i + n])
        count = seen.get(gram, 0) + 1
        if count >= limit:
            return True
        seen[gram] = count
    return False
