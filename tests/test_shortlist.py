from __future__ import annotations

from hy_mt_mlx_server.shortlist import (
    Glossary,
    GlossaryEntry,
    HashingNgramEmbedder,
    ShortlistStore,
    TranslationMemoryEntry,
    append_jsonl,
    cosine,
    rank_candidates,
)


def test_embedder_is_deterministic_and_normalised():
    embedder = HashingNgramEmbedder(dim=128)
    first = embedder.embed(["Restart the service"])[0]
    second = embedder.embed(["Restart the service"])[0]
    assert first == second
    assert abs(cosine(first, first) - 1.0) < 1e-9
    assert cosine(first, embedder.embed(["totally different text"])[0]) < 0.9


def test_store_ranks_near_duplicates_first():
    store = ShortlistStore(
        [
            TranslationMemoryEntry("Restart the service after updating the configuration.", "更新配置后重启服务。", "en", "zh"),
            TranslationMemoryEntry("Delete the cached model files.", "删除缓存的模型文件。", "en", "zh"),
        ]
    )
    top = store.query("Restart the service after updating the config.", source_lang="en", target_lang="zh", k=2)
    assert top and top[0].entry.target == "更新配置后重启服务。"
    assert top[0].score > 0.7


def test_store_filters_by_language_pair():
    # "" means "any language"; a concrete mismatch must not match.
    store = ShortlistStore(
        [
            TranslationMemoryEntry("hello", "你好", "en", "zh"),
            TranslationMemoryEntry("hello", "bonjour", "en", "fr"),
            TranslationMemoryEntry("hello", "哈囉", "", ""),
        ]
    )
    zh_hits = store.query("hello", source_lang="en", target_lang="zh", k=5)
    assert {hit.entry.target for hit in zh_hits} == {"你好", "哈囉"}
    # A concrete but mismatching pair is filtered out; the language-agnostic
    # entry still matches.
    ja_hits = store.query("hello", source_lang="ja", target_lang="zh", k=5)
    assert {hit.entry.target for hit in ja_hits} == {"哈囉"}


def test_glossary_matches_longest_term_and_respects_language():
    glossary = Glossary(
        [
            GlossaryEntry("service", "服务", "en", "zh"),
            GlossaryEntry("service mesh", "服务网格", "en", "zh"),
            GlossaryEntry("service", "サービス", "en", "ja"),
        ]
    )
    assert glossary.terms_for("The service mesh handles it.", source_lang="en", target_lang="zh") == [
        ("service mesh", "服务网格"),
        ("service", "服务"),
    ]
    assert glossary.terms_for("The service is down.", source_lang="en", target_lang="ja") == [("service", "サービス")]
    assert glossary.terms_for("dienst", source_lang="nl", target_lang="zh") == []


def test_jsonl_round_trip(tmp_path):
    path = tmp_path / "tm.jsonl"
    append_jsonl(path, {"source": "hello", "target": "你好", "source_lang": "en", "target_lang": "zh"})
    store = ShortlistStore.from_jsonl(path)
    assert len(store) == 1
    assert store.entries[0].target == "你好"


def test_rank_candidates_uses_the_scorer():
    def scorer(prompt: str, candidates: list[str]) -> list[float]:
        return [float(len(c)) for c in candidates]

    ranked = rank_candidates(scorer, "prompt", ["aa", "aaaa", "aaa"])
    assert [c for c, _ in ranked] == ["aaaa", "aaa", "aa"]


def test_glossary_does_not_match_inside_identifiers():
    glossary = Glossary([GlossaryEntry("agent", "代理", "en", "zh")])
    assert glossary.terms_for("The agent handles retries.", source_lang="en", target_lang="zh") == [("agent", "代理")]
    # Product names, paths and env vars must not be rewritten.
    for text in ("playable-agent uploads the zip", "PLAYABLE_AGENT_METRICS_PORT is set", "src/agent/run.ts"):
        assert glossary.terms_for(text, source_lang="en", target_lang="zh") == []


def test_glossary_is_case_insensitive_for_latin_terms():
    glossary = Glossary([GlossaryEntry("worker", "工作进程", "en", "zh")])
    assert glossary.terms_for("The Worker pool is busy.", source_lang="en", target_lang="zh") == [("worker", "工作进程")]
