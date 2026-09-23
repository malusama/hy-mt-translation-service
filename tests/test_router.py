from __future__ import annotations

import asyncio

from helpers import FakeFastBackend, FakeModel

from hy_mt_mlx_server.model import GenerationParams
from hy_mt_mlx_server.router import Router, RouterConfig
from hy_mt_mlx_server.shortlist import (
    Glossary,
    GlossaryEntry,
    ShortlistStore,
    TranslationMemoryEntry,
)

PARAMS = GenerationParams(max_new_tokens=64)
EN_ZH = {"source_lang": "en", "target_lang": "zh", "params": PARAMS}


def run(coro):
    return asyncio.run(coro)


def test_symbol_only_segment_skips_the_model():
    model = FakeModel()
    router = Router(model)
    (result,) = run(router.translate_many(["12:30 —— ..."], **EN_ZH))
    assert result.engine == "symbol"
    assert result.text == "12:30 —— ..."
    assert model.calls == []


def test_segment_already_in_target_language_is_echoed():
    model = FakeModel()
    router = Router(model)
    chinese, japanese = run(
        router.translate_many(["这是中文。", "会議は明日です。"], source_lang=None, target_lang="zh", params=PARAMS)
    )
    assert chinese.engine == "identity"
    assert chinese.text == "这是中文。"
    # Kana implies ja -> not the target language -> goes to the engine.
    assert japanese.engine == "main"
    assert model.calls == [("会議は明日です。", "zh")]
    assert router.stats["identity"] == 1


def test_exact_repeat_hits_the_cache():
    model = FakeModel()
    router = Router(model)
    first = run(router.translate_many(["The service is ready."], **EN_ZH))[0]
    second = run(router.translate_many(["The service is ready."], **EN_ZH))[0]
    assert first.engine == "main"
    assert second.engine == "cache"
    assert second.text == first.text
    assert len(model.calls) == 1


def test_translation_memory_reuse_avoids_a_forward_pass():
    model = FakeModel()
    tm = ShortlistStore(
        [TranslationMemoryEntry("Restart the service after updating the configuration.", "更新配置后重启服务。", "en", "zh")]
    )
    router = Router(model, shortlist=tm)
    (result,) = run(
        router.translate_many(["Restart the service after updating the configuration."], **EN_ZH)
    )
    assert result.engine == "memory"
    assert result.text == "更新配置后重启服务。"
    assert model.calls == []
    assert result.similarity is not None and result.similarity > 0.97


def test_similar_segment_is_attached_as_a_reference():
    model = FakeModel()
    tm = ShortlistStore(
        [TranslationMemoryEntry("Restart the service after updating the configuration.", "更新配置后重启服务。", "en", "zh")]
    )
    config = RouterConfig(reuse_threshold=0.999, reference_threshold=0.3)
    router = Router(model, config=config, shortlist=tm)
    (result,) = run(router.translate_many(["Restart the service after updating config."], **EN_ZH))
    assert result.engine == "main"
    assert model.prompts, "referenced segments must go through translate_prompts"
    assert "Reference the following translations:" in model.prompts[0]
    assert "更新配置后重启服务。" in model.prompts[0]


def test_glossary_terms_are_injected():
    model = FakeModel()
    glossary = Glossary([GlossaryEntry("service", "服务", "en", "zh")])
    router = Router(model, glossary=glossary)
    (result,) = run(router.translate_many(["The service is down."], **EN_ZH))
    assert result.engine == "main"
    assert model.prompts and "service translates to 服务" in model.prompts[0]
    assert router.stats["reference_injected"] == 1


def test_fast_path_serves_short_pairs():
    model = FakeModel()
    fast = FakeFastBackend()
    router = Router(model, config=RouterConfig(fast_enabled=True), fast_backend=fast)
    (result,) = run(router.translate_many(["The service is ready."], **EN_ZH))
    assert result.engine == "fast"
    assert model.calls == []
    assert fast.calls and fast.calls[0][1] == "en"
    assert router.stats["fast"] == 1


def test_fast_path_output_is_gated_and_escalates():
    model = FakeModel()
    fast = FakeFastBackend({"The service is ready.": "totally untranslated english"})
    router = Router(model, config=RouterConfig(fast_enabled=True), fast_backend=fast)
    (result,) = run(router.translate_many(["The service is ready."], **EN_ZH))
    assert result.engine == "main"
    assert model.calls, "escalated segments must reach the main engine"
    assert router.stats["fast_escalated"] == 1


def test_fast_path_failure_falls_back_to_main():
    model = FakeModel()
    fast = FakeFastBackend()
    fast.fail = RuntimeError("connection refused")
    router = Router(model, config=RouterConfig(fast_enabled=True), fast_backend=fast)
    (result,) = run(router.translate_many(["The service is ready."], **EN_ZH))
    assert result.engine == "main"
    assert "fast path unavailable" in result.reason


def test_hard_quality_failure_triggers_one_retry_with_other_prompt_style():
    calls = {"n": 0}

    def transform(text: str, to_lang: str) -> str:
        calls["n"] += 1
        if calls["n"] == 1:  # first attempt comes back as an echo
            return text
        return "这是译文" + text

    model = FakeModel(transform=transform)
    router = Router(model)
    (result,) = run(router.translate_many(["The service is ready."], **EN_ZH))
    assert result.engine == "main"
    assert result.issues == []
    assert "recovered on retry" in result.reason
    assert calls["n"] == 2
    assert "Translate the following text into Chinese" in model.prompts[0]
    assert router.stats["retried"] == 1


def test_soft_quality_issues_do_not_trigger_a_retry():
    # ratio_high only: a long CJK output that is not an echo, so the retry
    # heuristic (hard issues only) must stay out of the way.
    model = FakeModel(transform=lambda text, to_lang: "请在确认之后重启服务，并检查配置是否已经生效。" * 2)
    router = Router(model)
    (result,) = run(router.translate_many(["Restart at 12:30"], **EN_ZH))
    assert result.engine == "main"
    assert result.issues == ["ratio_high"]
    assert model.prompts == []


def test_route_only_never_calls_the_model():
    model = FakeModel()
    router = Router(model)
    routes = router.route_only(
        ["12:30", "The service is ready.", "这是中文。"],
        source_lang=None,
        target_lang="zh",
    )
    assert [r["engine"] for r in routes] == ["symbol", "main", "identity"]
    assert model.calls == []


def test_batch_keeps_per_segment_decisions():
    model = FakeModel()
    router = Router(model)
    results = run(
        router.translate_many(
            ["12:30", "The service is ready.", "The cache is cold."],
            **EN_ZH,
        )
    )
    assert [r.engine for r in results] == ["symbol", "main", "main"]
    assert model.many_calls, "multi-segment batches go through translate_many"
    stats = router.stats
    assert stats["segments"] == 3
    assert stats["main"] == 2
    assert stats["no_main_model_ratio"] > 0.0


def test_long_input_is_chunked_before_the_engine():
    model = FakeModel()
    router = Router(model, config=RouterConfig(max_input_chars=40))
    text = ("First sentence about the service. " * 4).strip()
    (result,) = run(router.translate_many([text], **EN_ZH))
    assert result.engine == "main"
    assert len(model.calls) >= 2, "chunked input must be translated chunk by chunk"


def test_model_rerank_orders_shortlist_candidates():
    model = FakeModel()
    model.score_map = {"更新配置后重启服务。": -1.0, "重新启动配置更新后的服务。": -0.1}
    tm = ShortlistStore(
        [
            TranslationMemoryEntry("Restart the service after updating the configuration.", "更新配置后重启服务。", "en", "zh"),
            TranslationMemoryEntry("Restart the service after updating the configuration", "重新启动配置更新后的服务。", "en", "zh"),
        ]
    )
    config = RouterConfig(reuse_threshold=0.999, reference_threshold=0.3, rerank="model")
    router = Router(model, config=config, shortlist=tm)
    (result,) = run(router.translate_many(["Restart the service after updating the configuration."], **EN_ZH))
    assert model.scored, "SHORTLIST_RERANK=model must score candidates in one forward pass"
    assert set(model.scored[0][1]) == {"更新配置后重启服务。", "重新启动配置更新后的服务。"}
    # The higher log-probability candidate wins and is attached as reference.
    assert result.engine == "main"
    assert model.prompts and "重新启动配置更新后的服务。" in model.prompts[0]


def test_declared_same_language_still_translates_a_different_script():
    model = FakeModel()
    router = Router(model)
    # A client that declares en->en for a Chinese page must not be short-circuited.
    (result,) = run(router.translate_many(["这是中文。"], source_lang="en", target_lang="en", params=PARAMS))
    assert result.engine == "main"
    # Latin text declared en->en is a genuine no-op and may be echoed.
    (echo,) = run(router.translate_many(["already english."], source_lang="en", target_lang="en", params=PARAMS))
    assert echo.engine == "identity"


def test_identifiers_are_passed_through_without_the_model():
    model = FakeModel()
    router = Router(model)
    cases = ["TODO", "SHA256", "v1.2.3", "</div>", "https://example.com/a/b?c=1", "user@example.com"]
    results = run(router.translate_many(cases, source_lang="en", target_lang="zh", params=PARAMS))
    assert [r.engine for r in results] == ["passthrough"] * len(cases)
    assert [r.text for r in results] == cases
    assert model.calls == []
    assert router.stats["passthrough"] == len(cases)


def test_caps_marketing_prose_is_not_passed_through():
    model = FakeModel()
    router = Router(model)
    (result,) = run(router.translate_many(["SAVE 20% TODAY"], **EN_ZH))
    assert result.engine == "main"
    assert model.calls


def test_passthrough_can_be_disabled():
    model = FakeModel()
    router = Router(model, config=RouterConfig(skip_untranslatable=False))
    (result,) = run(router.translate_many(["SHA256"], **EN_ZH))
    assert result.engine == "main"
