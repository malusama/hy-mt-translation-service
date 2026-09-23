"""Shared fakes: the router must be testable without mlx or a GPU."""

from __future__ import annotations

from typing import Optional, Sequence

from hy_mt_mlx_server.model import GenerationParams

CJK_TARGETS = {"zh", "zh-hant", "ja", "ko"}


def script_aware(text: str, to_lang: str) -> str:
    """Produce output that passes the target-script gate."""
    if to_lang.lower() in CJK_TARGETS:
        return "这是译文" + text
    return "translated: " + text


class FakeModel:
    """Minimal TranslationModel implementation that records every call."""

    def __init__(self, transform=None) -> None:
        self.model_id = "fake-model"
        self.transform = transform or script_aware
        self.calls: list[tuple[str, str]] = []
        self.prompts: list[str] = []
        self.many_calls: list[list[str]] = []
        self.scored: list[tuple[str, list[str]]] = []
        self.score_map: dict[str, float] = {}

    async def ensure_loaded(self) -> None:
        return None

    def _render(self, text: str, to_lang: str) -> str:
        return self.transform(text, to_lang)

    async def translate(self, text: str, to_lang: str, params: GenerationParams) -> str:
        self.calls.append((text, to_lang))
        return self._render(text, to_lang)

    async def translate_many(self, texts: list[str], to_lang: str, params: GenerationParams) -> list[str]:
        self.many_calls.append(list(texts))
        self.calls.extend((t, to_lang) for t in texts)
        return [self._render(t, to_lang) for t in texts]

    async def translate_prompts(self, prompts: Sequence[str], params: GenerationParams) -> list[str]:
        self.prompts.extend(prompts)
        out: list[str] = []
        for prompt in prompts:
            source = prompt.split("\n\n")[-1]
            self.calls.append((source, "prompt"))
            out.append(self._render(source, "zh" if _cjk_prompt(prompt) else "en"))
        return out

    async def score_continuations(self, prompt: str, candidates: Sequence[str]) -> list[float]:
        self.scored.append((prompt, list(candidates)))
        return [self.score_map.get(c, 0.0) for c in candidates]

    def close(self) -> None:
        return None


def _cjk_prompt(prompt: str) -> bool:
    head = prompt.split("\n\n")[0]
    return any(word in head for word in ("Chinese", "Japanese", "Korean"))


class FakeFastBackend:
    """Stand-in for the cascade's fast path (marian-edge style /imme)."""

    def __init__(self, outputs: Optional[dict[str, str]] = None, *, pairs=(("en", "zh"),)) -> None:
        self.outputs = outputs or {}
        self.pairs = set(pairs)
        self.calls: list[tuple[tuple[str, ...], str, str]] = []
        self.fail: Exception | None = None

    def supports_pair(self, source_lang: Optional[str], target_lang: Optional[str]) -> bool:
        return (source_lang or "", target_lang or "") in self.pairs

    async def translate_list(self, texts: Sequence[str], *, source_lang: str, target_lang: str) -> list[str]:
        self.calls.append((tuple(texts), source_lang, target_lang))
        if self.fail is not None:
            raise self.fail
        return [self.outputs.get(t, "快速译文" + t) for t in texts]

    def close(self) -> None:
        return None
