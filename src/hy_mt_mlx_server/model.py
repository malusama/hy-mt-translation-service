from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
from typing import Any, Optional, Protocol, Sequence

from .lang import target_language_name
from .prompts import build_translation_prompt, normalize_style


@dataclass(frozen=True)
class GenerationParams:
    max_new_tokens: int
    temperature: float = 0.7
    top_p: float = 0.6
    top_k: int = 20
    repetition_penalty: float = 1.05


class TranslationModel(Protocol):
    @property
    def model_id(self) -> str: ...

    async def ensure_loaded(self) -> None: ...

    async def translate(self, text: str, to_lang: str, params: GenerationParams) -> str: ...

    async def translate_many(self, texts: list[str], to_lang: str, params: GenerationParams) -> list[str]: ...

    async def translate_prompts(self, prompts: Sequence[str], params: GenerationParams) -> list[str]: ...

    def close(self) -> None: ...


def _extract_json_array(text: str) -> list[Any]:
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON array found")
    return json.loads(text[start : end + 1])


class HyMtMlxModel:
    def __init__(
        self,
        model_id: str,
        *,
        max_concurrency: int = 1,
        prompt_style: str = "legacy",
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")

        self._model_id = model_id
        self._style = normalize_style(prompt_style)
        self._model: Optional[Any] = None
        self._tokenizer: Optional[Any] = None
        self._load_lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._executor = ThreadPoolExecutor(max_workers=max_concurrency, thread_name_prefix="hy-mt-mlx")

    @property
    def model_id(self) -> str:
        return self._model_id

    def load_sync(self) -> None:
        # Lazy import so that CLI --help works even if mlx isn't installed yet.
        from mlx_lm import load  # type: ignore

        model, tokenizer = load(self._model_id)
        self._model = model
        self._tokenizer = tokenizer

    async def ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        async with self._load_lock:
            if self._model is not None and self._tokenizer is not None:
                return
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self._executor, self.load_sync)

    def _build_prompt(self, text: str, to_lang: str, glossary: Sequence[tuple[str, str]] = ()) -> str:
        prompt = build_translation_prompt(
            text,
            target_language_name(to_lang),
            style=self._style,
            glossary=glossary,
        )
        return self._apply_chat(prompt)

    def _apply_chat(self, prompt: str) -> str:
        """Wrap a raw prompt in the model's chat template.

        Hy-MT2 is only reliable when the template is applied: feeding it a raw
        prompt makes it continue the text instead of translating (measured --
        the raw variant hallucinates an unrelated sentence, the templated one
        does not). Both the plain and the glossary/reference paths must go
        through here.
        """
        tok = self._tokenizer
        if tok is None:
            return prompt

        if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None) is not None:
            messages = [{"role": "user", "content": prompt}]
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        return prompt

    def _generate_sync(self, prompt: str, params: GenerationParams) -> str:
        from mlx_lm import generate  # type: ignore
        from mlx_lm.sample_utils import make_logits_processors, make_sampler  # type: ignore

        if self._model is None or self._tokenizer is None:
            raise RuntimeError("model not loaded")

        sampler = make_sampler(
            temp=params.temperature,
            top_p=params.top_p,
            top_k=params.top_k,
        )
        logits_processors = make_logits_processors(
            repetition_penalty=params.repetition_penalty,
        )

        text = generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=params.max_new_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
            verbose=False,
        )
        return str(text).strip()

    def _generate_prompts_sync(self, prompts: Sequence[str], params: GenerationParams) -> list[str]:
        return [self._generate_sync(prompt, params) for prompt in prompts]

    def _score_sync(self, prompt: str, candidates: Sequence[str]) -> list[float]:
        """Mean token log-probability of each candidate continuation.

        One forward pass per candidate, no decoding -- the same "score, don't
        generate" primitive Laya exposes through ``predict_shortlist``. Used by
        the shortlist reranker (``SHORTLIST_RERANK=model``).
        """
        import mlx.core as mx  # type: ignore

        if self._model is None or self._tokenizer is None:
            raise RuntimeError("model not loaded")

        prompt_ids = list(self._tokenizer.encode(prompt))
        scores: list[float] = []
        for candidate in candidates:
            candidate_ids = list(self._tokenizer.encode(candidate))
            if not candidate_ids or not prompt_ids:
                scores.append(float("-inf"))
                continue
            logits = self._model(mx.array([prompt_ids + candidate_ids]))
            start = len(prompt_ids) - 1
            window = logits[0][start : start + len(candidate_ids)]
            log_probs = window - mx.logsumexp(window, axis=-1, keepdims=True)
            token_log_probs = log_probs[mx.arange(len(candidate_ids)), mx.array(candidate_ids)]
            scores.append(float(mx.mean(token_log_probs)))
        return scores

    async def translate(self, text: str, to_lang: str, params: GenerationParams) -> str:
        await self.ensure_loaded()
        prompt = self._build_prompt(text=text, to_lang=to_lang)
        async with self._semaphore:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self._executor, self._generate_sync, prompt, params)

    async def translate_prompts(self, prompts: Sequence[str], params: GenerationParams) -> list[str]:
        if not prompts:
            return []
        await self.ensure_loaded()
        templated = [self._apply_chat(prompt) for prompt in prompts]
        async with self._semaphore:
            loop = asyncio.get_running_loop()
            return list(await loop.run_in_executor(self._executor, self._generate_prompts_sync, templated, params))

    async def score_continuations(self, prompt: str, candidates: Sequence[str]) -> list[float]:
        await self.ensure_loaded()
        async with self._semaphore:
            loop = asyncio.get_running_loop()
            return list(await loop.run_in_executor(self._executor, self._score_sync, prompt, list(candidates)))

    async def translate_many(self, texts: list[str], to_lang: str, params: GenerationParams) -> list[str]:
        if not texts:
            return []
        if len(texts) == 1:
            return [await self.translate(texts[0], to_lang, params)]

        # Heuristics: avoid creating extremely long prompts / outputs.
        if len(texts) > 32 or sum(len(t) for t in texts) > 15000:
            return [await self.translate(t, to_lang, params) for t in texts]

        await self.ensure_loaded()
        lang_name = target_language_name(to_lang)
        payload = json.dumps(texts, ensure_ascii=False)
        batch_params = GenerationParams(
            max_new_tokens=params.max_new_tokens * min(len(texts), 8),
            temperature=params.temperature,
            top_p=params.top_p,
            top_k=params.top_k,
            repetition_penalty=params.repetition_penalty,
        )
        prompt = (
            "You are a translation engine.\n"
            f"Translate each item in the JSON array below into {lang_name}.\n"
            "Return ONLY a JSON array of strings, same length and same order as input.\n"
            "Do not include any additional explanation.\n\n"
            f"Input: {payload}\n\n"
            "Output:"
        )

        async with self._semaphore:
            loop = asyncio.get_running_loop()
            raw = await loop.run_in_executor(self._executor, self._generate_sync, prompt, batch_params)

        try:
            arr = _extract_json_array(raw)
            if not isinstance(arr, list):
                raise ValueError("output is not a JSON array")
            out = [str(x) for x in arr]
            if len(out) != len(texts):
                raise ValueError("output length mismatch")
            return out
        except Exception:
            return [await self.translate(t, to_lang, params) for t in texts]

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


class HyMtTransformersModel:
    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        dtype: str = "auto",
        *,
        max_concurrency: int = 1,
        prompt_style: str = "legacy",
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")

        self._model_id = model_id
        self._style = normalize_style(prompt_style)
        self._device = device
        self._dtype = dtype
        self._model: Optional[Any] = None
        self._tokenizer: Optional[Any] = None
        self._load_lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._executor = ThreadPoolExecutor(max_workers=max_concurrency, thread_name_prefix="hy-mt-torch")

    @property
    def model_id(self) -> str:
        return self._model_id

    def _resolve_device_sync(self) -> str:
        import torch  # type: ignore

        if self._device != "auto":
            return self._device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _resolve_dtype_sync(self) -> Any:
        import torch  # type: ignore

        if self._dtype == "float16":
            return torch.float16
        if self._dtype == "bfloat16":
            return torch.bfloat16
        if self._dtype == "float32":
            return torch.float32
        return "auto"

    def load_sync(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        from .runpod_cache import maybe_resolve_runpod_cached_model_path

        import torch  # type: ignore

        device = self._resolve_device_sync()
        torch_dtype = self._resolve_dtype_sync()

        resolved_id = maybe_resolve_runpod_cached_model_path(self._model_id)

        tokenizer = AutoTokenizer.from_pretrained(resolved_id, trust_remote_code=True)
        # Some causal-LM tokenizers don't define a pad token; ensure batching works.
        if getattr(tokenizer, "pad_token", None) is None:
            if getattr(tokenizer, "eos_token", None) is not None:
                tokenizer.pad_token = tokenizer.eos_token
            elif getattr(tokenizer, "unk_token", None) is not None:
                tokenizer.pad_token = tokenizer.unk_token
        if hasattr(tokenizer, "padding_side"):
            tokenizer.padding_side = "left"

        if device == "cuda":
            # Speed up matmul on Ampere+ (slight numeric differences; generally fine for translation).
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            except Exception:
                pass
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        model = AutoModelForCausalLM.from_pretrained(
            resolved_id,
            trust_remote_code=True,
            torch_dtype=None if torch_dtype == "auto" else torch_dtype,
            low_cpu_mem_usage=True,
            device_map=None,
        )

        model.eval()
        model.to(device)
        try:
            if getattr(model.config, "pad_token_id", None) is None and getattr(tokenizer, "pad_token_id", None) is not None:
                model.config.pad_token_id = tokenizer.pad_token_id
        except Exception:
            pass

        self._model = model
        self._tokenizer = tokenizer

    async def ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        async with self._load_lock:
            if self._model is not None and self._tokenizer is not None:
                return
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self._executor, self.load_sync)

    def _build_prompt(self, text: str, to_lang: str, glossary: Sequence[tuple[str, str]] = ()) -> str:
        prompt = build_translation_prompt(
            text,
            target_language_name(to_lang),
            style=self._style,
            glossary=glossary,
        )
        return self._apply_chat(prompt)

    def _apply_chat(self, prompt: str) -> str:
        """See HyMtMlxModel._apply_chat: the chat template is not optional."""
        tok = self._tokenizer
        if tok is None:
            return prompt
        if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None) is not None:
            messages = [{"role": "user", "content": prompt}]
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt

    def _generate_sync(self, prompt: str, params: GenerationParams) -> str:
        import torch  # type: ignore

        if self._model is None or self._tokenizer is None:
            raise RuntimeError("model not loaded")

        tokenizer = self._tokenizer
        model = self._model

        inputs = tokenizer(prompt, return_tensors="pt")
        # Some tokenizers return token_type_ids even when the model doesn't use them.
        inputs.pop("token_type_ids", None)
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        do_sample = params.temperature > 0
        with torch.inference_mode():
            gen_kwargs: dict[str, object] = {
                "do_sample": do_sample,
                "repetition_penalty": params.repetition_penalty,
                "max_new_tokens": params.max_new_tokens,
                "use_cache": True,
            }
            if do_sample:
                gen_kwargs.update(
                    {
                        "temperature": params.temperature,
                        "top_p": params.top_p,
                        "top_k": params.top_k,
                    }
                )
            if getattr(tokenizer, "pad_token_id", None) is not None:
                gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
            if getattr(tokenizer, "eos_token_id", None) is not None:
                gen_kwargs["eos_token_id"] = tokenizer.eos_token_id
            out = model.generate(**inputs, **gen_kwargs)

        generated = out[0][inputs["input_ids"].shape[1] :]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        return str(text).strip()

    async def translate(self, text: str, to_lang: str, params: GenerationParams) -> str:
        await self.ensure_loaded()
        prompt = self._build_prompt(text=text, to_lang=to_lang)
        async with self._semaphore:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self._executor, self._generate_sync, prompt, params)

    def _generate_many_sync(self, prompts: list[str], params: GenerationParams) -> list[str]:
        import torch  # type: ignore

        if self._model is None or self._tokenizer is None:
            raise RuntimeError("model not loaded")

        tokenizer = self._tokenizer
        model = self._model

        inputs = tokenizer(prompts, return_tensors="pt", padding=True)
        inputs.pop("token_type_ids", None)
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        prompt_lens = inputs["attention_mask"].sum(dim=1)
        do_sample = params.temperature > 0
        with torch.inference_mode():
            gen_kwargs: dict[str, object] = {
                "do_sample": do_sample,
                "repetition_penalty": params.repetition_penalty,
                "max_new_tokens": params.max_new_tokens,
                "use_cache": True,
            }
            if do_sample:
                gen_kwargs.update(
                    {
                        "temperature": params.temperature,
                        "top_p": params.top_p,
                        "top_k": params.top_k,
                    }
                )
            if getattr(tokenizer, "pad_token_id", None) is not None:
                gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
            if getattr(tokenizer, "eos_token_id", None) is not None:
                gen_kwargs["eos_token_id"] = tokenizer.eos_token_id
            out = model.generate(**inputs, **gen_kwargs)

        results: list[str] = []
        for i in range(out.shape[0]):
            generated = out[i][int(prompt_lens[i]) :]
            results.append(str(tokenizer.decode(generated, skip_special_tokens=True)).strip())
        return results

    async def translate_many(self, texts: list[str], to_lang: str, params: GenerationParams) -> list[str]:
        if not texts:
            return []
        if len(texts) == 1:
            return [await self.translate(texts[0], to_lang, params)]

        await self.ensure_loaded()
        prompts = [self._build_prompt(text=t, to_lang=to_lang) for t in texts]

        async with self._semaphore:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self._executor, self._generate_many_sync, prompts, params)

    async def translate_prompts(self, prompts: Sequence[str], params: GenerationParams) -> list[str]:
        if not prompts:
            return []
        await self.ensure_loaded()
        templated = [self._apply_chat(prompt) for prompt in prompts]
        async with self._semaphore:
            loop = asyncio.get_running_loop()
            return list(await loop.run_in_executor(self._executor, self._generate_many_sync, templated, params))

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


def has_mlx_backend() -> bool:
    try:
        import mlx_lm  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


def has_transformers_backend() -> bool:
    try:
        import transformers  # type: ignore  # noqa: F401
        import torch  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False
