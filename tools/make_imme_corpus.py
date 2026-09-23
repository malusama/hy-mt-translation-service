"""Build an immersive-translate-shaped corpus from whatever real text is on this machine.

The bundled numbers in the README come from running this over the repos under
--root; point it at your own checkouts to reproduce with your own content.
"""
import json
import os
import pathlib
import random
import re
from collections import Counter

random.seed(20260923)
BASE = pathlib.Path(os.environ.get("IMME_CORPUS_ROOT", "/Users/chensicheng/gitpath"))
OUT = pathlib.Path(os.environ.get("IMME_CORPUS_OUT", "/tmp/imme_corpus.jsonl"))

JA = re.compile(r"[\u3040-\u30ff]")
HAN = re.compile(r"[\u4e00-\u9fff]")
URL = re.compile(r"https?://")
CODEISH = re.compile(r"[{}<>;=]{2,}|^\s*(?:def|class|fn|const|let|import|from|pub|use|let mut)\b")

def clean(s: str) -> str:
    s = re.sub(r"`+", "", s)
    s = re.sub(r"[*_#>\-]{2,}", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def sentences(text: str):
    for raw in re.split(r"(?<=[.!?。！？])\s+", text):
        yield clean(raw)

rows = []

# A: real English docs prose (GitHub-page-like)
docs = []
for repo in ("marian-edge", "hy-mt-translation-service", "katago-cloud", "read19", "model-gateway", "qianshou-audit2-web", "loop"):
    p = BASE / repo
    if not p.is_dir():
        continue
    for f in list(p.rglob("*.md"))[:60]:
        if f.stat().st_size > 300_000 or ".git" in str(f):
            continue
        docs.append((repo, f))
en = []
for repo, f in docs:
    try:
        text = f.read_text(errors="ignore")
    except Exception:
        continue
    for s in sentences(text):
        if not s or JA.search(s) or HAN.search(s) or URL.search(s) or CODEISH.search(s):
            continue
        words = s.split()
        if 5 <= len(words) <= 45 and 32 <= len(s) <= 220 and s.endswith((".", "!", "?")):
            en.append((repo, s))
random.shuffle(en)
for repo, s in en[:220]:
    rows.append({"text": s, "src": "en", "tgt": "zh", "category": "A_en_docs_prose", "origin": f"{repo}:docs"})

# B: real short labels/headings (UI strings, FAQ bullets)
short = []
for repo, f in docs:
    try:
        text = f.read_text(errors="ignore")
    except Exception:
        continue
    for raw in text.splitlines():
        s = clean(raw)
        if not s or JA.search(s) or HAN.search(s) or URL.search(s):
            continue
        if s.startswith(("|", "#", "```", "-", "*", ">")):
            continue
        if 2 <= len(s.split()) <= 5 and 6 <= len(s) <= 28 and s.endswith((".", "?")) is False:
            short.append(s)
seen = set()
for s in short:
    if s.lower() in seen:
        continue
    seen.add(s.lower())
    rows.append({"text": s, "src": "en", "tgt": "zh", "category": "B_en_ui_label", "origin": "docs:heading"})
    if sum(1 for r in rows if r["category"] == "B_en_ui_label") >= 70:
        break

# C: real Japanese copy (katago-cloud i18n tables + ja prose)
ja_rows = []
for repo in ("katago-cloud", "m1-video-to-115", "anime-seichi-planner"):
    p = BASE / repo
    if not p.is_dir():
        continue
    for f in list(p.rglob("*.tsx"))[:80] + list(p.rglob("*.ts"))[:80] + list(p.rglob("*.swift"))[:80] + list(p.rglob("*.md"))[:40]:
        try:
            text = f.read_text(errors="ignore")
        except Exception:
            continue
        for m in re.finditer(r'"([^"\\]*[\u3040-\u30ff][^"\\]*)"', text):
            s = clean(m.group(1))
            if 4 <= len(s) <= 220 and not URL.search(s):
                ja_rows.append((repo, s))
seen = set()
for repo, s in ja_rows:
    if s in seen:
        continue
    seen.add(s)
    rows.append({"text": s, "src": "ja", "tgt": "zh", "category": "C_ja_real_copy", "origin": f"{repo}:i18n"})
    if sum(1 for r in rows if r["category"] == "C_ja_real_copy") >= 90:
        break

# D: real Chinese copy -> en
zh_rows = []
for repo in ("katago-cloud", "read19", "qianshou-audit2-web"):
    p = BASE / repo
    if not p.is_dir():
        continue
    for f in list(p.rglob("*.tsx"))[:80] + list(p.rglob("*.md"))[:40] + list(p.rglob("*.vue"))[:40]:
        try:
            text = f.read_text(errors="ignore")
        except Exception:
            continue
        for m in re.finditer(r'"([^"\\]*[\u4e00-\u9fff][^"\\]*)"', text):
            s = clean(m.group(1))
            if 4 <= len(s) <= 220 and not URL.search(s):
                zh_rows.append((repo, s))
seen = set()
for repo, s in zh_rows:
    if s in seen:
        continue
    seen.add(s)
    rows.append({"text": s, "src": "zh", "tgt": "en", "category": "D_zh_real_copy", "origin": f"{repo}:i18n"})
    if sum(1 for r in rows if r["category"] == "D_zh_real_copy") >= 60:
        break

# E: real code / identifiers / urls / versions (should never be "translated")
code_rows = [
    "npm install -g marian-edge", "cargo build --release --features metal", "curl -fsSL https://example.com/install.sh | sh",
    "const client = new OpenAI({ apiKey })", "fn main() -> Result<()>", "export interface RouterConfig { enabled: boolean }",
    "git rev-parse --short HEAD", "pip install -r builder/requirements.txt", "SELECT * FROM models WHERE id = ?",
    "v0.7.0", "SHA256", "HTTP 500", "application/json; charset=utf-8", "TODO", "README", "openapi.yaml",
    "https://github.com/malusama/marian-edge", "user@example.com", "{% csrf_token %}", "</div>",
    "MARIAN_EDGE_JA_EN_MODEL_DIR=models/jaen", "kwargs.get('timeout_s', 30)", "2026-09-23T07:44:05Z",
]
for s in code_rows:
    rows.append({"text": s, "src": "en", "tgt": "zh", "category": "E_code_identifier", "origin": "code"})

# F: repeats (UI strings repeat constantly on a real page)
repeats = [r for r in rows if r["category"] in ("B_en_ui_label", "C_ja_real_copy")][:25]
for r in repeats:
    rows.append({**r, "category": "F_repeat"})

OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
c = Counter(r["category"] for r in rows)
print(f"wrote {len(rows)} segments -> {OUT}")
for k, v in sorted(c.items()):
    print(f"  {k:<20} {v}")
for cat in sorted(c):
    ex = next(r for r in rows if r["category"] == cat)
    print(f"  e.g. {cat:<20} {ex['src']}->{ex['tgt']}  {ex['text'][:70]!r}")
