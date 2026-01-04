from __future__ import annotations

import re
from typing import Optional

import langid


def normalize_lang(code: Optional[str]) -> Optional[str]:
    if code is None:
        return None
    c = code.strip()
    if not c:
        return None

    c_lower = c.lower()
    if c_lower in {"auto", "detect"}:
        return None
    if c_lower in {"zh-cn", "zh-hans"}:
        return "zh"
    if c_lower in {"zh-tw", "zh-hk", "zh-hant"}:
        return "zh-Hant"
    if c_lower in {"jp"}:
        return "ja"
    if c_lower == "cn":
        return "zh"

    m = re.match(r"^([a-z]{2,3})(?:[-_].+)?$", c_lower)
    if m:
        return m.group(1)
    return c


def detect_language_code(text: str) -> str:
    if not text.strip():
        return "en"
    code, _ = langid.classify(text)
    # keep compatible-ish with LinguaSpark/server's short codes
    if code == "ja":
        return "ja"
    if code == "zh":
        return "zh"
    return code


def target_language_name(code: str) -> str:
    # For the HY-MT prompt we can use English language names.
    # Keep minimal mapping; fallback to the code itself.
    mapping = {
        "zh": "Chinese",
        "zh-Hant": "Traditional Chinese",
        "en": "English",
        "fr": "French",
        "pt": "Portuguese",
        "es": "Spanish",
        "ja": "Japanese",
        "tr": "Turkish",
        "ru": "Russian",
        "ar": "Arabic",
        "ko": "Korean",
        "th": "Thai",
        "it": "Italian",
        "de": "German",
        "vi": "Vietnamese",
        "ms": "Malay",
        "id": "Indonesian",
        "tl": "Filipino",
        "hi": "Hindi",
        "pl": "Polish",
        "cs": "Czech",
        "nl": "Dutch",
        "km": "Khmer",
        "my": "Burmese",
        "fa": "Persian",
        "gu": "Gujarati",
        "ur": "Urdu",
        "te": "Telugu",
        "mr": "Marathi",
        "he": "Hebrew",
        "bn": "Bengali",
        "ta": "Tamil",
        "uk": "Ukrainian",
        "bo": "Tibetan",
        "kk": "Kazakh",
        "mn": "Mongolian",
        "ug": "Uyghur",
        "yue": "Cantonese",
    }
    return mapping.get(code, code)

