from __future__ import annotations

import json

from fastapi.testclient import TestClient
from helpers import FakeFastBackend, FakeModel

from hy_mt_mlx_server.api import make_app
from hy_mt_mlx_server.router import Router, RouterConfig
from hy_mt_mlx_server.settings import Settings
from hy_mt_mlx_server.shortlist import Glossary, ShortlistStore


def build(settings_kwargs=None, router_kwargs=None):
    settings = Settings(
        PRELOAD_MODEL=False,
        MODEL_ID="fake-model",
        MAX_NEW_TOKENS=256,
        **(settings_kwargs or {}),
    )
    model = FakeModel()
    router = Router(model, **(router_kwargs or {}))
    return TestClient(make_app(model, settings, router)), model, router


def test_health_and_detect():
    client, _model, _router = build()
    assert client.get("/health").json() == {"status": "ok"}
    assert client.post("/detect", json={"text": "hello"}).json()["language"] == "en"
    assert client.post("/detect", json={"text": "你好"}).json()["language"] == "zh"


def test_translate_reports_the_engine():
    client, model, _router = build()
    response = client.post("/translate", json={"text": "The service is ready.", "from": "en", "to": "zh"})
    assert response.status_code == 200
    body = response.json()
    assert body["from"] == "en" and body["to"] == "zh"
    assert body["engine"] == "main"
    assert body["text"].startswith("这是译文")
    assert len(model.calls) == 1


def test_imme_skips_symbol_only_segments():
    client, model, _router = build()
    response = client.post(
        "/imme",
        json={"source_lang": "en", "target_lang": "zh", "text_list": ["12:30", "The service is ready."]},
    )
    assert response.status_code == 200
    items = response.json()["translations"]
    assert [item["engine"] for item in items] == ["symbol", "main"]
    assert items[0]["text"] == "12:30"
    assert len(model.calls) == 1


def test_route_endpoint_runs_no_forward_pass():
    client, model, _ = build()
    response = client.post("/route", json={"text_list": ["12:30", "The cache is cold."], "target_lang": "zh"})
    assert response.status_code == 200
    routes = response.json()["routes"]
    assert [r["engine"] for r in routes] == ["symbol", "main"]
    assert model.calls == []


def test_shortlist_and_glossary_additions_are_counted_and_persisted(tmp_path):
    tm_path = tmp_path / "tm.jsonl"
    gl_path = tmp_path / "glossary.jsonl"
    client, _model, _router = build(
        {"SHORTLIST_PATH": str(tm_path), "GLOSSARY_PATH": str(gl_path)},
        {"shortlist": ShortlistStore(), "glossary": Glossary()},
    )
    added = client.post(
        "/shortlist",
        json={"entries": [{"source": "Restart the service.", "target": "重启服务。", "source_lang": "en", "target_lang": "zh"}]},
    ).json()
    assert added == {"added": 1, "shortlist_entries": 1, "glossary_entries": 0, "persisted": True}
    assert json.loads(tm_path.read_text().splitlines()[0])["target"] == "重启服务。"

    terms = client.post(
        "/glossary",
        json={"terms": [{"source_term": "service", "target_term": "服务", "source_lang": "en", "target_lang": "zh"}]},
    ).json()
    assert terms["glossary_entries"] == 1
    assert json.loads(gl_path.read_text().splitlines()[0])["target_term"] == "服务"

    stats = client.get("/stats").json()
    assert stats["router"]["shortlist_entries"] == 1
    assert stats["router"]["glossary_entries"] == 1


def test_shortlist_addition_takes_effect_immediately():
    client, model, _router = build(router_kwargs={"shortlist": ShortlistStore()})
    client.post(
        "/shortlist",
        json={"entries": [{"source": "The cache is cold.", "target": "缓存已失效。", "source_lang": "en", "target_lang": "zh"}]},
    )
    response = client.post("/translate", json={"text": "The cache is cold.", "from": "en", "to": "zh"})
    assert response.json()["engine"] == "memory"
    assert response.json()["text"] == "缓存已失效。"
    assert model.calls == []


def test_api_key_enforcement():
    client, _model, _router = build({"API_KEY": "secret"})
    assert client.post("/translate", json={"text": "hello", "from": "en", "to": "zh"}).status_code == 401
    ok = client.post(
        "/translate",
        json={"text": "hello", "from": "en", "to": "zh"},
        headers={"Authorization": "Bearer secret"},
    )
    assert ok.status_code == 200
    assert client.post("/translate?token=secret", json={"text": "hello", "from": "en", "to": "zh"}).status_code == 200


def test_stats_reports_the_fast_path_and_backend():
    client, _model, _router = build(
        {"ROUTER_FAST_URL": "http://127.0.0.1:3000", "BACKEND": "mlx"},
        {"config": RouterConfig(fast_enabled=True), "fast_backend": FakeFastBackend()},
    )
    stats = client.get("/stats").json()
    assert stats["backend"] == "mlx"
    assert stats["fast_path"] == "http://127.0.0.1:3000"
    assert "router" in stats


def test_router_can_be_disabled_to_restore_legacy_imme_path():
    client, model, _router = build({"ROUTER_ENABLED": False}, {"config": RouterConfig(enabled=False)})
    response = client.post(
        "/imme",
        json={"source_lang": "en", "target_lang": "zh", "text_list": ["12:30", "The service is ready."]},
    )
    items = response.json()["translations"]
    # Legacy path translates everything, including the symbol-only segment, and
    # reports no per-segment engine (the field serialises as null).
    assert [item["engine"] for item in items] == [None, None]
    assert any(text == "12:30" for text, _lang in model.calls)
