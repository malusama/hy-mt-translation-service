"""Shortlist / translation-memory retrieval, in the spirit of Laya's
``predict_shortlist``: rank candidates with a cheap embedding first, and only
let the big model see the top-k.

Two data files, both optional, both JSONL so they can be curated by hand or
exported from a CAT tool:

* ``GLOSSARY_PATH``  -- ``{"source_term","target_term","source_lang","target_lang"}``
  injected into the prompt via the Hy-MT2 terminology template;
* ``SHORTLIST_PATH`` -- ``{"source","target","source_lang","target_lang"}``
  translation memory. A near-identical source segment whose stored target
  passes the quality gate is reused *without* a forward pass; a merely similar
  one is attached as a reference translation for the model to verify.

The default embedder is a deterministic hashed character n-gram TF vector. It
needs no model download, no GPU and no network, and it is stable across
processes (``zlib.crc32`` instead of ``hash()``, which is salted per run). The
interface accepts any ``Embedder`` so a sentence encoder can be dropped in
later without touching the router.
"""

from __future__ import annotations

import json
import math
import re
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Protocol, Sequence

TermPair = tuple[str, str]


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class HashingNgramEmbedder:
    """Hashed char n-gram term-frequency vectors, L2 normalised."""

    def __init__(self, dim: int = 256, ngrams: Sequence[int] = (2, 3, 4)) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = int(dim)
        self.ngrams = tuple(int(n) for n in ngrams if int(n) > 0) or (3,)

    def _grams(self, text: str) -> Iterable[str]:
        norm = re.sub(r"\s+", " ", text.strip().casefold())
        if len(norm) < min(self.ngrams):
            yield norm
            return
        for n in self.ngrams:
            for i in range(len(norm) - n + 1):
                yield norm[i : i + n]

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            for gram in self._grams(text):
                vec[zlib.crc32(gram.encode("utf-8")) % self.dim] += 1.0
            norm = math.sqrt(sum(v * v for v in vec))
            if norm > 0:
                vec = [v / norm for v in vec]
            out.append(vec)
        return out


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return float(sum(x * y for x, y in zip(a, b)))


def _norm_lang(code: str | None) -> str:
    if not code:
        return ""
    return code.strip().lower().split("-")[0].split("_")[0]


def _lang_matches(entry_lang: str, requested: str | None) -> bool:
    stored = _norm_lang(entry_lang)
    if not stored:
        return True
    return stored == _norm_lang(requested)


def _term_pattern(term: str) -> re.Pattern[str]:
    """Match a glossary term as a word, never inside an identifier.

    Plain substring search turned ``playable-agent`` into "可玩的代理": the
    product name contains the term ``agent``. Requiring non-identifier
    characters around the match keeps product names, env vars and paths intact.
    """
    escaped = re.escape(term)
    flags = re.IGNORECASE if term.isascii() else 0
    # Allow the regular English plural so `workers`/`agents` still match.
    suffix = r"(?:s|es)?" if term.isascii() else ""
    # Block identifier context on both sides: letters/digits/underscore/hyphen
    # before, and the same plus a file extension (`.` + alphanumeric) after.
    # A sentence-final "." is not identifier context, so it still matches.
    return re.compile(rf"(?<![A-Za-z0-9_/-]){escaped}{suffix}(?![A-Za-z0-9_-]|\.[A-Za-z0-9])", flags)


@dataclass(frozen=True)
class TranslationMemoryEntry:
    source: str
    target: str
    source_lang: str = ""
    target_lang: str = ""
    meta: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Match:
    entry: TranslationMemoryEntry
    score: float


class ShortlistStore:
    """In-memory translation memory with embedding retrieval."""

    def __init__(self, entries: Sequence[TranslationMemoryEntry] = (), embedder: Embedder | None = None) -> None:
        self.embedder = embedder or HashingNgramEmbedder()
        self._entries: list[TranslationMemoryEntry] = []
        self._vectors: list[list[float]] = []
        if entries:
            self.add_many(entries)

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> tuple[TranslationMemoryEntry, ...]:
        return tuple(self._entries)

    @classmethod
    def from_jsonl(cls, path: str | Path, embedder: Embedder | None = None, *, strict: bool = False) -> "ShortlistStore":
        store = cls(embedder=embedder)
        file = Path(path)
        if not file.exists():
            if strict:
                raise FileNotFoundError(f"shortlist file not found: {file}")
            return store
        for line in file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            raw = json.loads(line)
            store.add(
                TranslationMemoryEntry(
                    source=str(raw.get("source", "")).strip(),
                    target=str(raw.get("target", "")).strip(),
                    source_lang=str(raw.get("source_lang", "") or ""),
                    target_lang=str(raw.get("target_lang", "") or ""),
                    meta={k: str(v) for k, v in (raw.get("meta") or {}).items()},
                )
            )
        return store

    def add(self, entry: TranslationMemoryEntry) -> None:
        if not entry.source or not entry.target:
            return
        self.add_many([entry])

    def add_many(self, entries: Sequence[TranslationMemoryEntry]) -> None:
        prepared = [e for e in entries if e.source and e.target]
        if not prepared:
            return
        vectors = self.embedder.embed([e.source for e in prepared])
        self._entries.extend(prepared)
        self._vectors.extend(vectors)

    def query(
        self,
        text: str,
        *,
        source_lang: str | None = None,
        target_lang: str | None = None,
        k: int = 5,
    ) -> list[Match]:
        if not self._entries or not text.strip() or k <= 0:
            return []
        query_vec = self.embedder.embed([text])[0]
        scored: list[Match] = []
        for entry, vec in zip(self._entries, self._vectors):
            if not _lang_matches(entry.source_lang, source_lang):
                continue
            if not _lang_matches(entry.target_lang, target_lang):
                continue
            scored.append(Match(entry=entry, score=cosine(query_vec, vec)))
        scored.sort(key=lambda m: m.score, reverse=True)
        return scored[:k]


@dataclass(frozen=True)
class GlossaryEntry:
    source_term: str
    target_term: str
    source_lang: str = ""
    target_lang: str = ""


class Glossary:
    """Exact-term glossary. Longest match wins, case-insensitive for latin."""

    def __init__(self, entries: Sequence[GlossaryEntry] = ()) -> None:
        self._entries: list[GlossaryEntry] = []
        for entry in entries:
            self.add(entry)

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> tuple[GlossaryEntry, ...]:
        return tuple(self._entries)

    @classmethod
    def from_jsonl(cls, path: str | Path, *, strict: bool = False) -> "Glossary":
        file = Path(path)
        if not file.exists():
            if strict:
                raise FileNotFoundError(f"glossary file not found: {file}")
            return cls()
        entries: list[GlossaryEntry] = []
        for line in file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            raw = json.loads(line)
            entries.append(
                GlossaryEntry(
                    source_term=str(raw.get("source_term", "")).strip(),
                    target_term=str(raw.get("target_term", "")).strip(),
                    source_lang=str(raw.get("source_lang", "") or ""),
                    target_lang=str(raw.get("target_lang", "") or ""),
                )
            )
        return cls(entries)

    def add(self, entry: GlossaryEntry) -> None:
        if entry.source_term and entry.target_term:
            self._entries.append(entry)

    def add_many(self, entries: Sequence[GlossaryEntry]) -> None:
        for entry in entries:
            self.add(entry)

    def terms_for(
        self,
        text: str,
        *,
        source_lang: str | None = None,
        target_lang: str | None = None,
        limit: int = 8,
    ) -> list[TermPair]:
        if not self._entries or not text or limit <= 0:
            return []
        hits: list[tuple[int, TermPair]] = []
        for entry in self._entries:
            if not _lang_matches(entry.source_lang, source_lang):
                continue
            if not _lang_matches(entry.target_lang, target_lang):
                continue
            match = _term_pattern(entry.source_term).search(text)
            if match is None:
                continue
            hits.append((match.start(), (entry.source_term, entry.target_term)))
        hits.sort(key=lambda item: (item[0], -len(item[1][0])))
        out: list[TermPair] = []
        for _, pair in hits:
            if pair not in out:
                out.append(pair)
            if len(out) >= limit:
                break
        return out


def rank_candidates(
    scorer,
    prompt: str,
    candidates: Sequence[str],
) -> list[tuple[str, float]]:
    """Order candidates by one forward pass each (Laya's ``predict_shortlist``
    does the same thing with a decision head instead of a log-probability).

    ``scorer(prompt, candidates) -> list[float]`` is supplied by the backend;
    it is expected to *score*, never to generate.
    """
    unique: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in unique:
            unique.append(candidate)
    if len(unique) < 2:
        return [(c, 0.0) for c in unique]
    scores = list(scorer(prompt, unique))
    if len(scores) != len(unique):
        raise ValueError("scorer returned a mismatched number of scores")
    return sorted(zip(unique, scores), key=lambda item: item[1], reverse=True)


def append_jsonl(path: str | Path, payload: dict[str, object]) -> None:
    """Append one JSON object as a line, creating the file if needed."""
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
