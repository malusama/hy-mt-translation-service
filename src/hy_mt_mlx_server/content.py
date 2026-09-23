"""What is in a segment, and is any of it translatable?

Two cheap, deterministic classifiers that the router and the quality gate share:

* :func:`extract_tokens` - spans that must survive translation verbatim
  (URLs, e-mails, printf/format placeholders, template braces, tags, code spans,
  versions, identifier-like ALL_CAPS names);
* :func:`has_translatable_prose` - does this segment contain anything a
  translator would actually translate?

Both matter for first-shot correctness: identifiers, paths, versions and code
come back unchanged from *any* MT system, which is the right answer, so asking
the model at all wastes a forward pass and then trips an "echo" alarm.
"""

from __future__ import annotations

import re

# Spans that must be preserved verbatim.
_TOKEN_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"https?://\S+"),
    re.compile(r"www\.\S+"),
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    re.compile(r"%(?:\d+\$)?[sdfxX]"),
    re.compile(r"\{\{[^{}]{1,40}\}\}"),
    re.compile(r"\{[A-Za-z_][A-Za-z0-9_]{0,40}\}"),
    re.compile(r"\$\{[^{}]{1,40}\}"),
    re.compile(r"\{%.{1,60}?%\}"),  # Django/Jinja tags
    re.compile(r"</?[A-Za-z][A-Za-z0-9-]{0,20}>"),
    re.compile(r"`[^`]{1,60}`"),
    # Identifier-like caps: require a digit or underscore so marketing prose in
    # caps ("SAVE 20% TODAY") is not mistaken for code.
    re.compile(r"\b[A-Z][A-Z0-9_]*[_0-9][A-Z0-9_]*\b"),
    re.compile(r"\bv?\d+(?:\.\d+){1,3}\b"),
)

# Signals that a segment is code/command rather than prose. Unchanged output is
# expected for these, so the "echo" gate must stay quiet.
_CODE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"https?://\S+"),
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    re.compile(r"`[^`]{1,60}`"),
    re.compile(r"</?[A-Za-z][A-Za-z0-9-]{0,20}>"),
    re.compile(r"\{\{?[^{}]{1,40}\}?\}"),
    re.compile(r"\{%.{1,60}?%\}"),
    re.compile(r"%[sdfxX]"),
    re.compile(r"(?:^|\s)-{1,2}[A-Za-z][\w-]*"),  # command flags: -g, --model
    re.compile(r"\w+\("),  # call syntax: fn( / obj.method(
    re.compile(r"[A-Za-z_]\w*\.\w+"),  # attribute access, file names, versions
    re.compile(r"^[A-Z][A-Z0-9_]{2,}="),  # env assignment: MODEL_ID=...
    re.compile(r"[\"']\w[\w-]*[\"']\s*:"),  # JSON/JS key
    re.compile(r"\d{4}-\d{2}-\d{2}T?\d?"),  # ISO timestamp/date
    # CLI invocations: `npx wrangler deploy` is a command, not a sentence.
    # (see looks_like_command: a sentence that merely *starts* with a tool name
    # is not a command)
    re.compile(r"\b\w+\.(?:py|js|ts|rs|go|json|ya?ml|toml|md|sh|gguf|bin)\b"),
)

_CLI_PATTERN = re.compile(
    r"\s*(?:npx|npm|pnpm|yarn|bun|cargo|go|git|docker|kubectl|helm|make|cmake|python3?|pip3?|uv|"
    r"curl|wget|brew|gh|wrangler|task|just|lsof|kill|ssh|scp|rsync|jq|sed|awk|grep|find|xargs)\b"
)

_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")
_SENTENCE_PUNCT = re.compile(r"[.!?。！？](?:\s|$)")
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_.\-]*")
# Spans removed before asking "is there prose left?".
_STRIP_PATTERNS: tuple[re.Pattern[str], ...] = _TOKEN_PATTERNS + (
    re.compile(r"</?[A-Za-z][A-Za-z0-9-]{0,20}>"),
    re.compile(r"`[^`]{1,60}`"),
)


def extract_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for pattern in _TOKEN_PATTERNS:
        tokens.extend(pattern.findall(text))
    return tokens


def looks_like_code(text: str) -> bool:
    return any(pattern.search(text) for pattern in _CODE_PATTERNS)


def looks_like_command(text: str) -> bool:
    """A shell/CLI invocation: starts with a known tool and is not a sentence."""
    return bool(_CLI_PATTERN.match(text)) and not _SENTENCE_PUNCT.search(text)


def strip_protected(text: str) -> str:
    residue = text
    for pattern in _STRIP_PATTERNS:
        residue = pattern.sub(" ", residue)
    return residue


def word_count(text: str) -> int:
    """Number of latin words left after removing protected spans.

    Used by the quality gate: a single word (a brand, a product name) carries
    too little signal to demand that the target script appear in the output.
    """
    return len(_WORD.findall(strip_protected(text)))

def has_translatable_prose(text: str) -> bool:
    """True when a human translator would have something to translate.

    ``TODO``, ``SHA256``, ``v1.2.3``, ``GitHub``, ``a``, ``</div>`` and bare
    URLs/e-mails are not prose: every engine returns them unchanged, and that is
    the correct output. Multi-word caps (``SAVE 20% TODAY``) *is* prose, because
    it is a phrase rather than a name.
    """
    residue = strip_protected(text)
    letters = [ch for ch in residue if ch.isalpha()]
    if len(letters) < 2:
        return False
    # CJK/kana/hangul/cyrillic/... are always worth translating.
    if any(not ch.isascii() for ch in letters):
        return True
    words = _WORD.findall(residue)
    if not words:
        return False
    # A lone snake_case/dotted token is an environment variable, path or file
    # name, not a word.
    stripped = residue.strip()
    if _IDENTIFIER.fullmatch(stripped) and ("_" in stripped or "." in stripped):
        return False
    # A command line carries no sentence punctuation and no prose: shell
    # commands, tags and flags are meant to stay as they are.
    if (looks_like_code(text) or looks_like_command(text)) and not _SENTENCE_PUNCT.search(residue):
        return False
    # A word carrying lowercase (Hello, GitHub) or a multi-word phrase (SAVE 20%
    # TODAY) is prose; a lone ALL-CAPS token (TODO, OK, README) is a name, and
    # single-letter fragments (the "T"/"Z" of a timestamp) are not words.
    if any(word != word.upper() for word in words):
        return True
    return len(words) >= 2 and all(len(word) >= 2 for word in words)
