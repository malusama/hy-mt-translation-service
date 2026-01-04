from __future__ import annotations

import re
from typing import Optional

from .lang import detect_language_code
from .model import GenerationParams, TranslationModel


def split_long_text(text: str, max_chars: int) -> list[str]:
    """
    Best-effort chunking for very long inputs.

    - Keep formatting boundaries (newlines) as much as possible.
    - Keep each chunk <= max_chars (approx), so generation isn't silently truncated by output/context limits.
    """
    if max_chars <= 0:
        return [text]
    if len(text) <= max_chars:
        return [text]

    # Split paragraphs but keep separators.
    parts = re.split(r"(\n{2,})", text)
    chunks: list[str] = []

    def flush(buf: str) -> None:
        if buf:
            chunks.append(buf)

    for part in parts:
        if not part:
            continue
        if part.startswith("\n"):
            # Keep paragraph separators attached to previous chunk when possible.
            if chunks and len(chunks[-1]) + len(part) <= max_chars:
                chunks[-1] += part
            else:
                chunks.append(part)
            continue

        if len(part) <= max_chars:
            chunks.append(part)
            continue

        # Further split by sentences (keep punctuation).
        sentences = re.split(r"(?<=[。！？!?\\.])", part)
        buf = ""
        for s in sentences:
            if not s:
                continue
            if len(s) > max_chars:
                flush(buf)
                buf = ""
                for i in range(0, len(s), max_chars):
                    chunks.append(s[i : i + max_chars])
                continue

            if not buf:
                buf = s
                continue

            if len(buf) + len(s) <= max_chars:
                buf += s
            else:
                flush(buf)
                buf = s
        flush(buf)

    # Final merge: avoid tiny chunks by merging when safe.
    merged: list[str] = []
    for c in chunks:
        if not merged:
            merged.append(c)
            continue
        if len(merged[-1]) + len(c) <= max_chars:
            merged[-1] += c
        else:
            merged.append(c)
    return merged


async def translate_text(
    model: TranslationModel,
    *,
    text: str,
    to_lang: str,
    params: GenerationParams,
    max_input_chars: int,
) -> str:
    max_chars = max(1, int(max_input_chars))
    if len(text) <= max_chars:
        return await model.translate(text=text, to_lang=to_lang, params=params)

    chunks = split_long_text(text, max_chars)
    out_parts: list[str] = []
    for ch in chunks:
        out_parts.append(await model.translate(text=ch, to_lang=to_lang, params=params))
    return "".join(out_parts)


async def translate_text_list(
    model: TranslationModel,
    *,
    texts: list[str],
    to_lang: str,
    params: GenerationParams,
    max_input_chars: int,
    batch_mode: str,
    batch_size: int,
) -> list[str]:
    if not texts:
        return []
    if len(texts) == 1:
        return [await translate_text(model, text=texts[0], to_lang=to_lang, params=params, max_input_chars=max_input_chars)]

    mode = (batch_mode or "auto").strip().lower()
    if mode == "off":
        return [
            await translate_text(model, text=t, to_lang=to_lang, params=params, max_input_chars=max_input_chars)
            for t in texts
        ]

    size = max(1, int(batch_size))
    out: list[str] = []
    for i in range(0, len(texts), size):
        batch = texts[i : i + size]
        if any(len(t) > max_input_chars for t in batch):
            for t in batch:
                out.append(await translate_text(model, text=t, to_lang=to_lang, params=params, max_input_chars=max_input_chars))
            continue

        try:
            out.extend(await model.translate_many(batch, to_lang, params))
        except Exception:
            if mode == "on":
                raise
            for t in batch:
                out.append(await translate_text(model, text=t, to_lang=to_lang, params=params, max_input_chars=max_input_chars))
    return out


async def translate_imme(
    model: TranslationModel,
    *,
    text_list: list[str],
    target_lang: str,
    source_lang: Optional[str],
    params: GenerationParams,
    max_input_chars: int,
    batch_mode: str,
    batch_size: int,
) -> list[dict[str, str]]:
    if not text_list:
        return []

    detected_list = [
        source_lang if source_lang is not None else detect_language_code(str(t)) for t in text_list
    ]
    translated_list = await translate_text_list(
        model,
        texts=[str(t) for t in text_list],
        to_lang=target_lang,
        params=params,
        max_input_chars=max_input_chars,
        batch_mode=batch_mode,
        batch_size=batch_size,
    )

    if len(translated_list) != len(detected_list):
        # Keep correctness over speed.
        raise RuntimeError("translation length mismatch")

    return [
        {"detected_source_lang": detected_list[i], "text": translated_list[i]} for i in range(len(detected_list))
    ]

