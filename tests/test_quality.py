from __future__ import annotations

from hy_mt_mlx_server.quality import check_translation, extract_tokens


def test_empty_output_fails():
    report = check_translation("hello", "   ", "zh")
    assert not report.ok and report.issues == ["empty"]


def test_echo_detection():
    report = check_translation("Local inference keeps the model on the device.", "Local inference keeps the model on the device.", "zh")
    assert "echo" in report.issues


def test_target_script_missing_is_flagged():
    report = check_translation("The service is ready.", "The service is ready to use.", "zh")
    assert "missing_target_script" in report.issues


def test_clean_translation_passes():
    report = check_translation("The service is ready.", "服务已就绪。", "zh")
    assert report.ok, report.issues


def test_lost_placeholder_is_flagged():
    source = "Restart the service at https://example.com/docs before 12:30"
    good = "请在 12:30 之前重启服务：https://example.com/docs"
    bad = "请在 12:30 之前重启服务。"
    assert check_translation(source, good, "zh").ok
    assert "lost_placeholder" in check_translation(source, bad, "zh").issues


def test_ratio_gate_flags_runaway_output():
    report = check_translation("Hi there!", "这是一个非常长的译文" * 12, "zh")
    assert "ratio_high" in report.issues


def test_ratio_gate_ignores_short_sources():
    # 5 chars -> 33 chars is a normal CJK title expansion, not a runaway.
    report = check_translation("定式讲解史", "Standard Explanation of History", "en")
    assert report.ok, report.issues


def test_degenerate_repetition_is_flagged():
    repeated = " ".join(["word"] * 20)
    report = check_translation("hello", repeated, "en")
    assert "repetition" in report.issues


def test_extract_tokens_finds_urls_and_placeholders():
    tokens = extract_tokens("see https://a.example/x and %s plus {user} at v1.2.3")
    assert "https://a.example/x" in tokens
    assert "%s" in tokens
    assert "{user}" in tokens
    assert "v1.2.3" in tokens


def test_truncated_output_is_flagged():
    source = "This paragraph is deliberately long enough to exceed the fast path budget."
    truncated = "这段文字刻意足够长,足以超过快速路径的预算"  # no terminal punctuation
    assert "possibly_truncated" in check_translation(source, truncated, "zh").issues
    assert "possibly_truncated" not in check_translation(source, "这段文字刻意足够长，足以超过快速路径的预算。", "zh").issues
    # A source that is not a finished sentence must not trigger the check.
    assert "possibly_truncated" not in check_translation("no terminator here", truncated, "zh").issues
