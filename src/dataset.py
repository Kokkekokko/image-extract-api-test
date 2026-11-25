"""Load the ground-truth file and pair it with the images folder."""
from __future__ import annotations

import csv
import json
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}

DEFAULT_FIELDS = {"id": "id", "image": "image", "question": "question", "answer": "answer",
                  "task": "task", "category": "category"}


def _read_rows(path: Path) -> list[dict]:
    if path.suffix.lower() == ".csv":
        with open(path, encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else data.get("items", [])
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_items(cfg: dict) -> list[dict]:
    """Return items: {id, image_path, question, answer, task, category}. answer=None means unscored."""
    ds = cfg.get("dataset", {})
    images_dir = Path(ds.get("images_dir", "images"))
    fields = {**DEFAULT_FIELDS, **(ds.get("fields") or {})}
    items: list[dict] = []
    seen_images: set[str] = set()

    gt = ds.get("ground_truth")
    if gt and Path(gt).exists():
        for i, row in enumerate(_read_rows(Path(gt))):
            image_name = str(row.get(fields["image"]) or "")
            item_id = str(row.get(fields["id"]) or Path(image_name).stem or i)
            if not image_name:
                raise ValueError(f"Row {i} in {gt} has no '{fields['image']}' column value")
            answer = row.get(fields["answer"])
            items.append({
                "id": item_id,
                "image_path": str(images_dir / image_name),
                "question": row.get(fields["question"]) or "",
                "answer": None if answer is None else str(answer),
                "task": row.get(fields["task"]) or "",
                "category": row.get(fields["category"]) or "",
            })
            seen_images.add(Path(image_name).name)
    elif gt:
        print(f"[warn] ground truth file '{gt}' not found - responses will be collected but not scored")

    # Images in the folder that have no ground-truth row are still sent (unscored).
    if images_dir.exists():
        for p in sorted(images_dir.iterdir()):
            if p.suffix.lower() in IMAGE_EXTS and p.name not in seen_images:
                items.append({"id": p.stem, "image_path": str(p), "question": "", "answer": None,
                              "task": "", "category": ""})
    return items
