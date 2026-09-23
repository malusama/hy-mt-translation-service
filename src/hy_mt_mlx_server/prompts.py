"""Prompt construction shared by every backend.

Keeping this in one place matters once the router starts injecting glossary
references: MLX, transformers and the HTTP/llama.cpp engine must send byte
identical prompts, otherwise the shortlist and the cascade stop being
comparable.

``legacy`` is the prompt this service shipped with (a short instruction). It
measured no worse than the official wording on an en/ja/zh spot check and is
~15% cheaper to prefill, so it stays the default. ``official`` is the wording
from the Hy-MT2 model card, incl. its terminology template.
"""

from __future__ import annotations

from typing import Sequence

STYLE_LEGACY = "legacy"
STYLE_OFFICIAL = "official"
STYLES = (STYLE_LEGACY, STYLE_OFFICIAL)

TermPair = tuple[str, str]


def _reference_block(terms: Sequence[TermPair]) -> str:
    lines = ["Reference the following translations:"]
    for source, target in terms:
        lines.append(f"{source} translates to {target}")
    return "\n".join(lines) + "\n\n"


def build_translation_prompt(
    text: str,
    to_lang_name: str,
    *,
    style: str = STYLE_LEGACY,
    glossary: Sequence[TermPair] = (),
) -> str:
    reference = _reference_block(glossary) if glossary else ""
    style = (style or STYLE_LEGACY).strip().lower()
    # A reference/terminology block only exists in the official template. Mixing
    # it with the short legacy instruction is an untrained combination and made
    # the model hallucinate an unrelated sentence (observed on Hy-MT2-1.8B), so
    # referenced prompts always use the official wording.
    if style == STYLE_OFFICIAL or reference:
        return (
            f"{reference}Translate the following text into {to_lang_name}. Note that you must "
            f"only output the translated result without any additional explanation:\n\n{text}"
        )
    return (
        f"{reference}Translate the following segment into {to_lang_name}, "
        f"without additional explanation.\n\n{text}"
    )


def normalize_style(style: str | None) -> str:
    s = (style or "").strip().lower()
    return s if s in STYLES else STYLE_LEGACY
