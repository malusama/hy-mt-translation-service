from __future__ import annotations

import pytest

from hy_mt_mlx_server.content import extract_tokens, has_translatable_prose, looks_like_code
from hy_mt_mlx_server.quality import check_translation


@pytest.mark.parametrize(
    "text",
    ["TODO", "OK", "SHA256", "v1.2.3", "a", "</div>", "https://example.com/a/b?c=1",
     "user@example.com", "{% csrf_token %}", "npm_config_cache", ""],
)
def test_identifiers_and_code_are_not_prose(text):
    assert not has_translatable_prose(text)


@pytest.mark.parametrize(
    "text",
    ["Hello", "Read more", "SAVE 20% TODAY", "The service is ready.", "服务", "設定",
     "HTTP 500 Internal Server Error", "GitHub Actions failed on the arm64 job."],
)
def test_prose_is_detected(text):
    assert has_translatable_prose(text)


def test_unchanged_identifier_is_not_reported_as_echo():
    # The model correctly returns these unchanged; the gate must not cry wolf.
    for source in ["SHA256", "v1.2.3", "https://example.com/a/b?c=1", "</div>"]:
        report = check_translation(source, source, "zh")
        assert report.ok, (source, report.issues)


def test_unchanged_prose_is_still_reported_as_echo():
    report = check_translation("The service is ready to use.", "The service is ready to use.", "zh")
    assert "echo" in report.issues


def test_command_echo_is_not_flagged():
    command = "npm install -g marian-edge"
    assert check_translation(command, command, "zh").ok


def test_missing_target_script_is_prose_only():
    # No Chinese in the output, but the source is only an identifier -> not an error.
    assert check_translation("SHA256", "SHA256", "zh").ok
    # A lone brand name may legitimately stay as it is.
    assert check_translation("GitHub", "GitHub", "zh").ok
    # Multi-word prose that stays in latin script for a zh target is an error.
    assert "missing_target_script" in check_translation("The service is ready.", "The service is ready.", "zh").issues


def test_caps_prose_is_not_treated_as_a_protected_token():
    tokens = extract_tokens("SAVE 20% TODAY")
    assert "SAVE" not in tokens and "TODAY" not in tokens
    assert "HTTP_500" in extract_tokens("HTTP_500 error") or "HTTP_500" in extract_tokens("HTTP_500")
