from __future__ import annotations

from hy_mt_mlx_server.script import (
    distinct_language,
    dominant_script,
    guess_language_by_script,
    is_symbol_or_number_only,
    scripts_in,
)


def test_scripts_in_ignores_digits_and_punctuation():
    assert scripts_in("2026-09-23 12:30") == set()
    assert scripts_in("abc") == {"latin"}
    assert scripts_in("こんにちは") == {"kana"}
    assert scripts_in("你好") == {"han"}
    assert scripts_in("한국어") == {"hangul"}


def test_kana_implies_japanese_han_stays_ambiguous():
    assert guess_language_by_script("会議は明日です") == "ja"
    assert guess_language_by_script("会議明日") == "zh"
    assert guess_language_by_script("hello world") == "en"
    assert guess_language_by_script("русский") == "ru"


def test_symbol_or_number_segments_are_skippable():
    assert is_symbol_or_number_only("12:30")
    assert is_symbol_or_number_only("—— ... !!!")
    assert not is_symbol_or_number_only("OK")


def test_distinct_language_only_trusts_decisive_scripts():
    assert distinct_language("你好世界") == "zh"
    assert distinct_language("こんにちは") == "ja"
    assert distinct_language("한국어") == "ko"
    assert distinct_language("русский текст") == "ru"
    # Latin never settles the language: an en-target request must not be
    # short-circuited for a French page.
    assert distinct_language("hello world") == ""
    assert distinct_language("12:30") == ""


def test_dominant_script():
    assert dominant_script("hello 你好") == "latin"
    assert dominant_script("你好世界朋友们 hello") == "han"
    assert dominant_script("1234") == "none"
