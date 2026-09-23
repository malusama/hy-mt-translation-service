"""HTTP backends.

``HyMtHttpModel``
    OpenAI-compatible chat-completions engine (``llama-server``, the bundled
    Rust web, RunPod or another hy-mt instance). Measured on an M1 Pro with
    Hy-MT2-1.8B Q4_K_M + llama.cpp Metal this is the fastest engine available
    to this service (~67 tok/s decode, ~620 tok/s prefill), so the Python
    process can keep its router/shortlist while the heavy lifting stays in
    llama.cpp.

``LinguaSparkFastBackend``
    A second, cheaper service speaking the LinguaSpark/hy-mt API (``/imme``).
    This is the cascade's fast path -- e.g. marian-edge's 31M en->zh student
    behind ``ROUTER_FAST_URL``.

Both are thin and dependency-free (``urllib`` inside a thread executor) so the
service keeps working on a machine without ``requests``/``httpx``.
"""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional, Sequence

from .lang import target_language_name
from .model import GenerationParams
from .prompts import build_translation_prompt, normalize_style


class HttpError(RuntimeError):
    pass


def _post_json(url: str, payload: dict[str, Any], *, timeout_s: float, api_key: str = "") -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    headers = {"content-type": "application/json; charset=utf-8"}
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, method="POST", data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:  # pragma: no cover - network path
        detail = exc.read().decode("utf-8", errors="replace")[:400]
        raise HttpError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except Exception as exc:  # pragma: no cover - network path
        raise HttpError(f"{type(exc).__name__} calling {url}: {exc}") from exc
    if not body.strip():
        raise HttpError(f"empty response from {url}")
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:  # pragma: no cover - network path
        raise HttpError(f"invalid JSON from {url}: {body[:200]!r}") from exc


class HyMtHttpModel:
    """OpenAI-compatible chat-completions engine."""

    def __init__(
        self,
        model_id: str,
        *,
        base_url: str,
        api_key: str = "",
        timeout_s: float = 120.0,
        max_concurrency: int = 1,
        prompt_style: str = "legacy",
    ) -> None:
        if not base_url:
            raise ValueError("MODEL_HTTP_BASE_URL is required for BACKEND=http")
        self._model_id = model_id
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout_s = float(timeout_s)
        self._style = normalize_style(prompt_style)
        self._semaphore = asyncio.Semaphore(max(1, max_concurrency))
        self._executor = ThreadPoolExecutor(max_workers=max(1, max_concurrency), thread_name_prefix="hy-mt-http")

    @property
    def model_id(self) -> str:
        return self._model_id

    async def ensure_loaded(self) -> None:
        # The remote engine owns its own lifecycle.
        return None

    def _payload(self, prompt: str, params: GenerationParams) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": float(params.temperature),
            "max_tokens": int(params.max_new_tokens),
        }
        if params.temperature > 0:
            payload["top_p"] = float(params.top_p)
            payload["top_k"] = int(params.top_k)
        if self._model_id:
            payload["model"] = self._model_id
        return payload

    def _generate_sync(self, prompt: str, params: GenerationParams) -> str:
        body = _post_json(
            f"{self._base_url}/v1/chat/completions",
            self._payload(prompt, params),
            timeout_s=self._timeout_s,
            api_key=self._api_key,
        )
        try:
            choices = body["choices"]
            message = choices[0]["message"]
            content = message.get("content") or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise HttpError(f"unexpected chat-completions payload: {str(body)[:200]}") from exc
        return str(content).strip()

    def _build_prompt(self, text: str, to_lang: str, glossary: Sequence[tuple[str, str]] = ()) -> str:
        return build_translation_prompt(
            text,
            target_language_name(to_lang),
            style=self._style,
            glossary=glossary,
        )

    async def _run(self, prompt: str, params: GenerationParams) -> str:
        async with self._semaphore:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self._executor, self._generate_sync, prompt, params)

    async def translate(self, text: str, to_lang: str, params: GenerationParams) -> str:
        return await self._run(self._build_prompt(text, to_lang), params)

    async def translate_prompts(self, prompts: Sequence[str], params: GenerationParams) -> list[str]:
        """Translate raw, already-built prompts (used for glossary injection)."""
        return list(await asyncio.gather(*(self._run(p, params) for p in prompts)))

    async def translate_many(self, texts: list[str], to_lang: str, params: GenerationParams) -> list[str]:
        if not texts:
            return []
        return await self.translate_prompts([self._build_prompt(t, to_lang) for t in texts], params)

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


class LinguaSparkFastBackend:
    """Cascade fast path: another hy-mt/LinguaSpark-compatible service."""

    def __init__(
        self,
        base_url: str,
        *,
        pairs: Sequence[tuple[str, str]] = (("en", "zh"),),
        timeout_s: float = 30.0,
        api_key: str = "",
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_s = float(timeout_s)
        self._api_key = api_key.rstrip()
        self._pairs = {(_norm(a), _norm(b)) for a, b in pairs}
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hy-mt-fast")

    @property
    def base_url(self) -> str:
        return self._base_url

    def supports_pair(self, source_lang: str | None, target_lang: str | None) -> bool:
        if not self._pairs:
            return False
        return (_norm(source_lang), _norm(target_lang)) in self._pairs

    def _call_sync(self, texts: Sequence[str], source_lang: str, target_lang: str) -> list[str]:
        payload = {"source_lang": source_lang, "target_lang": target_lang, "text_list": list(texts)}
        body = _post_json(f"{self._base_url}/imme", payload, timeout_s=self._timeout_s, api_key=self._api_key)
        items = body.get("translations")
        if not isinstance(items, list) or len(items) != len(texts):
            raise HttpError(f"fast backend returned {type(items).__name__} of unexpected length")
        return [str((item or {}).get("text", "")) for item in items]

    async def translate_list(
        self,
        texts: Sequence[str],
        *,
        source_lang: str,
        target_lang: str,
    ) -> list[str]:
        if not texts:
            return []
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor, self._call_sync, list(texts), source_lang, target_lang
        )

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


def _norm(code: Optional[str]) -> str:
    if not code:
        return ""
    return code.strip().lower().split("-")[0].split("_")[0]


def parse_pairs(spec: str) -> list[tuple[str, str]]:
    """``ROUTER_FAST_PAIRS="en:zh,en:ja"`` -> ``[("en","zh"), ("en","ja")]``."""
    pairs: list[tuple[str, str]] = []
    for chunk in (spec or "").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"invalid pair {chunk!r}; expected 'src:tgt'")
        src, tgt = chunk.split(":", 1)
        src, tgt = _norm(src), _norm(tgt)
        if src and tgt:
            pairs.append((src, tgt))
    return pairs
