"""Text comparison metrics. String metrics work on characters, which suits Thai (no spaces between words).

Answer formats handled:
- plain text                          -> string metrics
- JSON list of strings '["a", "b"]'   -> several acceptable answers; best match is used (as in ANLS)
- JSON object / list of objects       -> also json_f1: F1 over flattened key/value pairs
"""
from __future__ import annotations

import json
import re
import string
import unicodedata
from collections import Counter

from rapidfuzz.distance import Levenshtein

_FENCE_RE = re.compile(r"^```[\w-]*\s*\n?(.*?)\n?```$", re.DOTALL)
_PUNCT = set(string.punctuation) | set("“”‘’«»…–—")

# Higher is better for all except cer.
METRIC_NAMES = ["exact_match", "edit_similarity", "anls", "cer", "char_f1", "token_f1", "json_f1"]


def strip_fences(text: str) -> str:
    text = text.strip()
    m = _FENCE_RE.match(text)
    return m.group(1).strip() if m else text


def normalize(text, lowercase=False, strip_punctuation=False, remove_whitespace=False) -> str:
    if text is None:
        return ""
    text = strip_fences(unicodedata.normalize("NFC", str(text)))
    if lowercase:
        text = text.lower()
    if strip_punctuation:
        text = "".join(ch for ch in text if ch not in _PUNCT)
    return "".join(text.split()) if remove_whitespace else " ".join(text.split())


# --------------------------------------------------------------------------- JSON helpers
def try_parse_json(text: str):
    """Parse JSON from a model answer: whole text, fenced block, or first {...} / [...] span."""
    if not text:
        return None
    text = strip_fences(unicodedata.normalize("NFC", str(text)))
    try:
        return json.loads(text)
    except ValueError:
        pass
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start, end = text.find(open_ch), text.rfind(close_ch)
        if 0 <= start < end:
            try:
                return json.loads(text[start:end + 1])
            except ValueError:
                continue
    return None


def gold_alternatives(gold: str) -> list[str]:
    """'["8", "อันดับ 8"]' -> ["8", "อันดับ 8"]; anything else -> [gold]."""
    s = (gold or "").strip()
    if s.startswith("["):
        try:
            val = json.loads(s)
        except ValueError:
            return [gold]
        if isinstance(val, list) and val and all(isinstance(v, (str, int, float)) for v in val):
            return [str(v) for v in val]
    return [gold]


def is_structured(value) -> bool:
    return isinstance(value, dict) or (isinstance(value, list) and any(isinstance(v, (dict, list)) for v in value))


def _flatten(obj, prefix="", out=None, norm=None):
    out = [] if out is None else out
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten(v, f"{prefix}.{norm(k)}" if prefix else norm(k), out, norm)
    elif isinstance(obj, list):
        # List positions are ignored so the order of rows/items does not matter.
        for v in obj:
            _flatten(v, f"{prefix}[]", out, norm)
    else:
        out.append((prefix, norm("" if obj is None else str(obj))))
    return out


def _f1(pred: list, gold: list) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    common = sum((Counter(pred) & Counter(gold)).values())
    if common == 0:
        return 0.0
    p, r = common / len(pred), common / len(gold)
    return 2 * p * r / (p + r)


def json_f1(pred_obj, gold_obj, norm) -> float:
    if pred_obj is None:
        return 0.0
    return _f1(_flatten(pred_obj, norm=norm), _flatten(gold_obj, norm=norm))


# --------------------------------------------------------------------------- main entry
def _string_metrics(pred: str, gold: str, anls_threshold: float) -> dict:
    dist = Levenshtein.distance(pred, gold)
    longest = max(len(pred), len(gold))
    sim = 1.0 if longest == 0 else 1 - dist / longest
    return {
        "exact_match": float(pred == gold),
        "edit_similarity": sim,
        "anls": sim if sim >= anls_threshold else 0.0,
        # CER can exceed 1 when the prediction is much longer than the answer.
        "cer": (dist / len(gold)) if gold else float(bool(pred)),
        "char_f1": _f1(list(pred.replace(" ", "")), list(gold.replace(" ", ""))),
        "token_f1": _f1(pred.split(), gold.split()),
    }


def compute_all(pred_raw: str, gold_raw: str, norm_opts: dict, anls_threshold: float = 0.5) -> dict:
    """Return every metric; json_f1 is None when the answer is not structured JSON."""
    def norm(t):
        return normalize(t, **norm_opts)

    pred = norm(pred_raw)
    best: dict = {}
    for alt in gold_alternatives(gold_raw):
        scores = _string_metrics(pred, norm(alt), anls_threshold)
        for k, v in scores.items():
            if k not in best:
                best[k] = v
            else:
                best[k] = min(best[k], v) if k == "cer" else max(best[k], v)

    gold_obj = try_parse_json(gold_raw) if gold_raw and gold_raw.strip()[:1] in "{[" else None
    best["json_f1"] = json_f1(try_parse_json(pred_raw), gold_obj, norm) if is_structured(gold_obj) else None
    return best
