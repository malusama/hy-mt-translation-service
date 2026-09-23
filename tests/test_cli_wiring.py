from __future__ import annotations

from helpers import FakeModel

from hy_mt_mlx_server.cli import build_router
from hy_mt_mlx_server.settings import Settings


def _settings(**kwargs) -> Settings:
    return Settings(PRELOAD_MODEL=False, MODEL_ID="fake-model", **kwargs)


def test_stores_exist_without_backing_files():
    router = build_router(_settings(), FakeModel())
    # Runtime additions must work even when no JSONL is configured.
    assert router.shortlist is not None and len(router.shortlist) == 0
    assert router.glossary is not None and len(router.glossary) == 0
    assert router.fast_backend is None


def test_paths_are_loaded_when_configured(tmp_path):
    tm = tmp_path / "tm.jsonl"
    tm.write_text('{"source": "hello", "target": "你好", "source_lang": "en", "target_lang": "zh"}\n', encoding="utf-8")
    gl = tmp_path / "glossary.jsonl"
    gl.write_text('{"source_term": "service", "target_term": "服务", "source_lang": "en", "target_lang": "zh"}\n', encoding="utf-8")
    router = build_router(
        _settings(SHORTLIST_PATH=str(tm), GLOSSARY_PATH=str(gl)),
        FakeModel(),
    )
    assert len(router.shortlist) == 1
    assert len(router.glossary) == 1


def test_missing_path_is_tolerated():
    router = build_router(_settings(SHORTLIST_PATH="/nonexistent/tm.jsonl"), FakeModel())
    assert router.shortlist is not None and len(router.shortlist) == 0


def test_fast_backend_is_configured_from_url_and_pairs():
    router = build_router(
        _settings(ROUTER_FAST_URL="http://127.0.0.1:3000", ROUTER_FAST_PAIRS="en:zh,en:ja"),
        FakeModel(),
    )
    assert router.fast_backend is not None
    assert router.fast_backend.supports_pair("en", "zh")
    assert router.fast_backend.supports_pair("en", "ja")
    assert not router.fast_backend.supports_pair("ja", "zh")
    router.close()
