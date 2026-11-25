"""Steps 1 + 2: send each image to the API and store every response as JSON."""
from __future__ import annotations

import json
import mimetypes
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from .clients import ApiResult, RetryableError, make_client
from .dataset import load_items


def safe_name(item_id: str) -> str:
    return re.sub(r"[^\w.-]", "_", item_id)[:150]


def build_prompt(cfg: dict, question: str) -> str:
    p = cfg.get("prompt", {})
    question = question or p.get("default_question", "Extract all text in this image.")
    system = (p.get("system") or "").strip()
    return f"{system}\n\n{question}" if system else question


def _call_with_retries(client, max_retries: int, **kwargs) -> tuple[ApiResult, int]:
    attempt = 0
    while True:
        try:
            return client.call(**kwargs), attempt
        except RetryableError as e:
            if attempt >= max_retries:
                return ApiResult(error=f"failed after {attempt + 1} attempts: {e}"), attempt
            time.sleep(min(60, 2 ** attempt * 2) + random.random())
            attempt += 1
        except Exception as e:  # unexpected client error: record and move on
            return ApiResult(error=f"{type(e).__name__}: {e}"), attempt


def _process(item: dict, cfg: dict, client, out_dir: Path, max_retries: int) -> dict:
    path = Path(item["image_path"])
    record = {"id": item["id"], "image": path.name, "task": item["task"], "category": item["category"],
              "question": item["question"], "expected": item["answer"]}
    prompt = build_prompt(cfg, item["question"])
    record["prompt"] = prompt
    if not path.exists():
        record.update(response_text="", error=f"image not found: {path}", latency_ms=0, attempts=0,
                      usage={}, raw_response=None)
    else:
        image = path.read_bytes()
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        t0 = time.perf_counter()
        result, retries = _call_with_retries(client, max_retries, image=image, mime=mime, prompt=prompt,
                                             filename=path.name, expected=item["answer"])
        record.update(response_text=result.text, error=result.error,
                      latency_ms=round((time.perf_counter() - t0) * 1000), attempts=retries + 1,
                      usage=result.usage, raw_response=result.raw)
    record["model"] = client.model_name
    record["timestamp"] = datetime.now(timezone.utc).isoformat()
    (out_dir / f"{safe_name(item['id'])}.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return record


def _is_done(path: Path) -> bool:
    """A saved response counts as done unless it failed at the transport level."""
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    err = rec.get("error") or ""
    return not (err.startswith("failed after") or err.startswith("HTTP 5") or "image not found" in err)


def run(cfg: dict, run_dir: Path, limit: int | None = None, rerun: bool = False) -> Path:
    resp_dir = run_dir / "responses"
    resp_dir.mkdir(parents=True, exist_ok=True)
    items = load_items(cfg)
    if not items:
        raise SystemExit("No images found. Put images in the images folder or run `python main.py prepare`.")
    if limit:
        items = items[:limit]

    todo = [it for it in items if rerun or not _is_done(resp_dir / f"{safe_name(it['id'])}.json")]
    print(f"{len(items)} images, {len(items) - len(todo)} already done, {len(todo)} to send")

    client = make_client(cfg)
    api = cfg.get("api", {})
    meta = {"provider": api.get("provider"), "model": client.model_name,
            "started": datetime.now(timezone.utc).isoformat(), "n_items": len(items),
            "ground_truth": cfg.get("dataset", {}).get("ground_truth"),
            "system_prompt": cfg.get("prompt", {}).get("system")}
    (run_dir / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    errors = 0
    with ThreadPoolExecutor(max_workers=int(api.get("concurrency", 4))) as pool:
        futures = [pool.submit(_process, it, cfg, client, resp_dir, int(api.get("max_retries", 3)))
                   for it in todo]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Sending images"):
            if fut.result().get("error"):
                errors += 1

    # Rebuild the combined file from the per-image files, in item order.
    combined = run_dir / "responses.jsonl"
    with open(combined, "w", encoding="utf-8") as f:
        for it in items:
            p = resp_dir / f"{safe_name(it['id'])}.json"
            if p.exists():
                rec = json.loads(p.read_text(encoding="utf-8"))
                rec.pop("raw_response", None)  # keep the combined file small; raw stays per-image
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    meta["finished"] = datetime.now(timezone.utc).isoformat()
    (run_dir / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved responses to {combined} ({errors} errors this run)")
    return combined
