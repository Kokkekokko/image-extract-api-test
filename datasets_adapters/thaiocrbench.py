"""Download a sample of ThaiOCRBench (typhoon-ai/ThaiOCRBench) into images/ + data/ground_truth.jsonl.

The dataset is a single `test` split (2,808 rows) with columns:
Id, image, Task, question, answer, category. Streaming lets us take a few rows per task
without downloading the full 1.9 GB.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from tqdm import tqdm

DATASET = "typhoon-ai/ThaiOCRBench"


def prepare(images_dir: Path, gt_path: Path, per_task: int | None = 5, limit: int | None = None) -> None:
    from datasets import load_dataset

    images_dir.mkdir(parents=True, exist_ok=True)
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(DATASET, split="test", streaming=True)

    counts: Counter = Counter()
    rows = []
    bar = tqdm(desc="Reading ThaiOCRBench", unit="row")
    for ex in ds:
        bar.update(1)
        task = ex["Task"]
        if per_task and counts[task] >= per_task:
            # ThaiOCRBench has 13 task types: stop once each one has enough rows.
            if len(counts) >= 13 and all(v >= per_task for v in counts.values()):
                break
            continue
        filename = f"{ex['Id']}.png"
        img = ex["image"]
        if img.mode not in ("RGB", "L", "RGBA"):
            img = img.convert("RGB")
        img.save(images_dir / filename)
        rows.append({"id": ex["Id"], "image": filename, "question": ex["question"], "answer": ex["answer"],
                     "task": task, "category": ex.get("category") or ""})
        counts[task] += 1
        if limit and len(rows) >= limit:
            break
    bar.close()

    with open(gt_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(rows)} images to {images_dir}/ and ground truth to {gt_path}")
    print("Images per task:")
    for task, n in sorted(counts.items()):
        print(f"  {n:4d}  {task}")
    print("\nExample answer per task (first 120 chars):")
    shown = set()
    for r in rows:
        if r["task"] not in shown:
            shown.add(r["task"])
            print(f"  [{r['task']}] {r['answer'][:120]!r}")
