"""Regression guard: every prompt must reach the model through its chat template.

Hy-MT2 fed a raw prompt continues the text instead of translating (measured);
the glossary/reference path used to bypass the template because it submitted an
already-built prompt straight to the generator.
"""

from __future__ import annotations

import asyncio

import pytest

from hy_mt_mlx_server.model import GenerationParams, HyMtMlxModel, HyMtTransformersModel


class FakeTokenizer:
    chat_template = "fake"

    def __init__(self) -> None:
        self.seen: list[str] = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True) -> str:
        self.seen.append(messages[0]["content"])
        return "<CHAT>" + messages[0]["content"] + "<END>"


@pytest.mark.parametrize("model_cls", [HyMtMlxModel, HyMtTransformersModel])
def test_apply_chat_wraps_glossary_prompts(model_cls):
    model = model_cls("fake-model")
    model._tokenizer = FakeTokenizer()
    prompt = model._build_prompt("hello", "zh", glossary=[("service", "服务")])
    assert prompt.startswith("<CHAT>") and prompt.endswith("<END>")
    assert "Reference the following translations:" in prompt


@pytest.mark.parametrize("model_cls", [HyMtMlxModel, HyMtTransformersModel])
def test_apply_chat_passes_plain_prompts_through(model_cls):
    model = model_cls("fake-model")
    model._tokenizer = None
    assert model._apply_chat("raw prompt") == "raw prompt"


def test_mlx_translate_prompts_templates_its_input():
    model = HyMtMlxModel("fake-model")
    model._model = object()  # short-circuit ensure_loaded
    model._tokenizer = FakeTokenizer()
    captured: dict[str, list[str]] = {}

    def fake_generate(prompts, params):
        captured["prompts"] = list(prompts)
        return ["out" for _ in prompts]

    model._generate_prompts_sync = fake_generate  # type: ignore[assignment]
    out = asyncio.run(model.translate_prompts(["raw one", "raw two"], GenerationParams(max_new_tokens=8)))
    assert out == ["out", "out"]
    assert all(p.startswith("<CHAT>") and p.endswith("<END>") for p in captured["prompts"])
