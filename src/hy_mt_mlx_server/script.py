"""Cheap script/language routing.

Everything here is pure-Python, allocation-light and free of model calls: the
router must be able to decide *before* any forward pass (the same contract as
Laya's ``Router.route``).  Character-class checks over ``str`` are fast enough
for the immersive-translate payload sizes this service accepts (64 KiB body).

The module intentionally avoids ``langid``; the statistical detector is still
used by ``/detect`` for the public contract, but routing must not depend on a
classifier that is wrong exactly where it matters (Han-only Japanese).
"""

from __future__ import annotations

# Unicode blocks (inclusive ranges) for the scripts this service can see.
_RANGES: tuple[tuple[str, int, int], ...] = (
    ("latin", 0x0041, 0x024F),
    ("latin", 0x1E00, 0x1EFF),
    ("greek", 0x0370, 0x03FF),
    ("cyrillic", 0x0400, 0x04FF),
    ("hebrew", 0x0590, 0x05FF),
    ("arabic", 0x0600, 0x06FF),
    ("arabic", 0x0750, 0x077F),
    ("devanagari", 0x0900, 0x097F),
    ("bengali", 0x0980, 0x09FF),
    ("gujarati", 0x0A80, 0x0AFF),
    ("tamil", 0x0B80, 0x0BFF),
    ("telugu", 0x0C00, 0x0C7F),
    ("kannada", 0x0C80, 0x0CFF),
    ("malayalam", 0x0D00, 0x0D7F),
    ("thai", 0x0E00, 0x0E7F),
    ("myanmar", 0x1000, 0x109F),
    ("georgian", 0x10A0, 0x10FF),
    ("ethiopic", 0x1200, 0x137F),
    ("khmer", 0x1780, 0x17FF),
    ("mongolian", 0x1800, 0x18AF),
    ("hangul", 0x1100, 0x11FF),
    ("hangul", 0x3130, 0x318F),
    ("hangul", 0xAC00, 0xD7AF),
    ("kana", 0x3040, 0x30FF),
    ("kana", 0x31F0, 0x31FF),
    ("han", 0x3400, 0x4DBF),
    ("han", 0x4E00, 0x9FFF),
    ("han", 0xF900, 0xFAFF),
    ("tibetan", 0x0F00, 0x0FFF),
)

# Languages that Firefox/Marian style pipelines treat as one family.
_SCRIPT_BY_LANG: dict[str, tuple[str, ...]] = {
    "zh": ("han",),
    "zh-hant": ("han",),
    "yue": ("han",),
    "ja": ("kana", "han"),
    "ko": ("hangul", "han"),
    "en": ("latin",),
    "fr": ("latin",),
    "de": ("latin",),
    "es": ("latin",),
    "pt": ("latin",),
    "it": ("latin",),
    "nl": ("latin",),
    "pl": ("latin",),
    "cs": ("latin",),
    "tr": ("latin",),
    "id": ("latin",),
    "ms": ("latin",),
    "vi": ("latin",),
    "tl": ("latin",),
    "ru": ("cyrillic",),
    "uk": ("cyrillic",),
    "ar": ("arabic",),
    "fa": ("arabic",),
    "ur": ("arabic",),
    "he": ("hebrew",),
    "hi": ("devanagari",),
    "mr": ("devanagari",),
    "bn": ("bengali",),
    "gu": ("gujarati",),
    "ta": ("tamil",),
    "te": ("telugu",),
    "kn": ("kannada",),
    "ml": ("malayalam",),
    "th": ("thai",),
    "my": ("myanmar",),
    "km": ("khmer",),
    "mn": ("cyrillic", "latin"),
    "kk": ("cyrillic",),
    "bo": ("tibetan",),
    "ug": ("arabic",),
}


def _script_of(ch: str) -> str | None:
    cp = ord(ch)
    for name, lo, hi in _RANGES:
        if lo <= cp <= hi:
            return name
    return None


def scripts_in(text: str) -> set[str]:
    """Return the set of scripts appearing in ``text`` (ignores digits/punct)."""
    found: set[str] = set()
    for ch in text:
        if ch.isalpha():
            name = _script_of(ch)
            if name is not None:
                found.add(name)
    return found


def dominant_script(text: str) -> str:
    counts: dict[str, int] = {}
    for ch in text:
        if not ch.isalpha():
            continue
        name = _script_of(ch)
        if name is None:
            continue
        counts[name] = counts.get(name, 0) + 1
    if not counts:
        return "none"
    return max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]


def has_letters(text: str) -> bool:
    return any(ch.isalpha() for ch in text)


def expected_scripts(lang: str | None) -> tuple[str, ...]:
    if not lang:
        return ()
    return _SCRIPT_BY_LANG.get(lang.strip().lower(), ())


def distinct_language(text: str) -> str:
    """Language implied by script alone, for scripts that are unambiguous.

    Returns ``""`` when the script cannot settle the question. Latin is
    deliberately excluded: "the page is in latin letters" says nothing about
    whether it is English or French, so a latin target must never be inferred
    from the script (that would pass French through an en-target request).
    Kana/cyrillic/etc. *are* decisive, which is what makes this usable as an
    "input is already the target language" gate.

    Han-only text maps to ``zh``: for a zh target that is the desired skip, and
    for a ja target it correctly refuses to skip (kanji-only Japanese is
    ambiguous and must be translated).
    """
    found = scripts_in(text)
    if not found:
        return ""
    if "kana" in found:
        return "ja"
    if "hangul" in found:
        return "ko"
    if "han" in found:
        return "zh"
    for lang, scripts in _SCRIPT_BY_LANG.items():
        if lang in {"zh", "ja", "ko", "en"} or "latin" in scripts:
            continue
        if scripts[0] in found:
            return lang
    return ""


def guess_language_by_script(text: str, default: str = "en") -> str:
    """Kana implies Japanese; other scripts narrow the candidate set.

    Han-only text stays ambiguous (Japanese kanji-only strings exist), so we
    return the ``default`` rather than claiming ``zh`` -- callers pass an
    explicit source language for subtitle clients, exactly like marian-edge.
    """
    found = scripts_in(text)
    if not found:
        return default
    if "kana" in found:
        return "ja"
    if "hangul" in found:
        return "ko"
    if "han" in found:
        return "zh"
    for lang, scripts in _SCRIPT_BY_LANG.items():
        if len(scripts) == 1 and scripts[0] in found and lang not in {"zh", "ja"}:
            return lang
    if "latin" in found:
        return default
    return default


def is_symbol_or_number_only(text: str) -> bool:
    """Segments with no letters carry no translatable content.

    Immersive-translate payloads are full of these: separators, dates, numbers,
    keyboard hints, emoji and bare URLs. Sending them to a 1.8B model costs a
    full forward pass and usually comes back unchanged.
    """
    return not has_letters(text)

