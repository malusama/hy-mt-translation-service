from __future__ import annotations

import os
import json
import urllib.request
from typing import Any, Optional

import runpod

RUST_WEB_URL = os.environ.get("RUST_WEB_URL", "http://127.0.0.1:3000").rstrip("/")


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default


def _post_json(path: str, payload: dict[str, Any], timeout_s: int = 600) -> dict[str, Any]:
    url = f"{RUST_WEB_URL}{path}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        method="POST",
        data=data,
        headers={"content-type": "application/json; charset=utf-8"},
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as res:
        body = res.read().decode("utf-8", errors="replace")
        if not body:
            return {}
        return json.loads(body)


def _get_json(path: str, timeout_s: int = 60) -> dict[str, Any]:
    url = f"{RUST_WEB_URL}{path}"
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout_s) as res:
        body = res.read().decode("utf-8", errors="replace")
        if not body:
            return {}
        return json.loads(body)


def _normalize_lang(code: Any) -> Optional[str]:
    if code is None:
        return None
    s = str(code).strip()
    if not s:
        return None
    sl = s.lower()
    if sl in {"auto", "detect"}:
        return None
    if sl in {"zh-cn", "zh-hans", "cn"}:
        return "zh"
    if sl in {"zh-tw", "zh-hk", "zh-hant"}:
        return "zh-Hant"
    if sl == "jp":
        return "ja"
    # keep short normalized code for compatibility
    out = ""
    for ch in sl:
        if "a" <= ch <= "z":
            out += ch
            if len(out) >= 3:
                break
        else:
            break
    return out if len(out) >= 2 else s


def _strip_control_fields(payload: dict[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    out.pop("action", None)
    out.pop("path", None)
    out.pop("endpoint", None)
    return out


def _handle(payload: dict[str, Any]) -> dict[str, Any]:
    action = payload.get("action") or payload.get("path") or payload.get("endpoint")
    if isinstance(action, str):
        a = action.strip().lstrip("/").lower()
    else:
        a = ""

    # Some callers wrap the real payload under "body"/"payload"/"input".
    data = payload
    for k in ("body", "payload", "input"):
        v = payload.get(k)
        if isinstance(v, dict):
            data = v
            break

    # Action might also live inside the wrapped object.
    if not a:
        inner_action = data.get("action") or data.get("path") or data.get("endpoint")
        if isinstance(inner_action, str):
            a = inner_action.strip().lstrip("/").lower()

    imme_max_texts = _int_env("IMME_MAX_TEXTS", 1024)

    # Infer action by schema if not explicitly provided.
    if not a:
        if "messages" in data:
            a = "v1/chat/completions"
        elif "text_list" in data and "target_lang" in data:
            a = "imme"
        elif "text" in data and ("to" in data or "target_lang" in data):
            a = "translate"
        elif "text" in data:
            a = "detect"

    if a == "detect":
        text = str(data.get("text", ""))
        return _post_json("/detect", {"text": text})

    if a in {"translate", "kiss"}:
        text = str(data.get("text", ""))
        to_lang = _normalize_lang(data.get("to", data.get("target_lang"))) or "en"
        from_lang_raw = data.get("from", data.get("from_lang", data.get("source_lang")))
        from_lang = _normalize_lang(from_lang_raw)
        req: dict[str, Any] = {"text": text, "to": to_lang}
        if from_lang is not None:
            req["from"] = from_lang
        return _post_json("/translate", req)

    if a == "imme":
        text_list = data.get("text_list", [])
        if not isinstance(text_list, list):
            raise ValueError("text_list must be a list")
        if not text_list:
            return {"translations": []}
        if imme_max_texts > 0 and len(text_list) > imme_max_texts:
            raise ValueError(f"text_list too large ({len(text_list)}); limit is IMME_MAX_TEXTS={imme_max_texts}")

        source_lang = _normalize_lang(data.get("source_lang"))
        target_lang = _normalize_lang(data.get("target_lang")) or "en"
        req: dict[str, Any] = {"text_list": [str(t) for t in text_list], "target_lang": target_lang}
        if source_lang is not None:
            req["source_lang"] = source_lang
        return _post_json("/imme", req)

    # OpenAI-style endpoints (non-streaming only in Job mode).
    if a in {"v1/chat/completions", "chat/completions"}:
        body = data if data is not payload else _strip_control_fields(payload)
        if bool(body.get("stream")):
            raise ValueError("streaming is not supported in RunPod Job API mode; use Load Balancing / HTTP Workers instead")
        return _post_json("/v1/chat/completions", body)

    if a in {"v1/models", "models"}:
        return _get_json("/v1/models")

    if a in {"health"}:
        return _get_json("/health")

    raise ValueError(
        "unrecognized input schema; provide action/path or fields like {text,to} / {target_lang,text_list}"
    )


def handler(job: dict[str, Any]) -> dict[str, Any]:
    payload = job.get("input")
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ValueError("job.input must be an object")
    return _handle(payload)


runpod.serverless.start({"handler": handler})
