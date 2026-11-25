"""Step 3: compare each response with the correct answer."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .metrics import METRIC_NAMES, compute_all

SCORE_COLUMNS = ["id", "image", "task", "category", "status", "primary_metric", "score",
                 *METRIC_NAMES, "latency_ms", "error", "expected", "predicted"]


def score(cfg: dict, run_dir: Path) -> Path:
    responses = run_dir / "responses.jsonl"
    if not responses.exists():
        raise SystemExit(f"{responses} not found - run the `run` step first.")
    m = cfg.get("metrics", {})
    norm = m.get("normalize", {}) or {}
    threshold = float(m.get("anls_threshold", 0.5))
    default_primary = m.get("default_primary", "edit_similarity")
    by_task = m.get("primary_by_task") or {}
    by_task_lower = {k.lower(): v for k, v in by_task.items()}

    rows = []
    with open(responses, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            task = rec.get("task") or ""
            primary = by_task_lower.get(task.lower(), default_primary)
            row = {"id": rec["id"], "image": rec.get("image"), "task": task,
                   "category": rec.get("category") or "", "primary_metric": primary,
                   "latency_ms": rec.get("latency_ms"), "error": rec.get("error") or "",
                   "expected": rec.get("expected"), "predicted": rec.get("response_text") or ""}
            if rec.get("expected") is None:
                row["status"] = "unscored"
            else:
                # A truncated / refused answer is still scored on whatever text came back.
                pred_raw = rec.get("response_text") or ""
                scores = compute_all(pred_raw, rec["expected"], norm, threshold)
                if scores.get(primary) is None:  # e.g. json_f1 on a non-JSON answer
                    primary = "edit_similarity"
                    row["primary_metric"] = primary
                row.update({k: (None if v is None else round(v, 4)) for k, v in scores.items()})
                row["score"] = round(scores[primary], 4)
                row["status"] = "error" if rec.get("error") and not pred_raw.strip() else "scored"
            rows.append(row)

    out = run_dir / "scores.csv"
    with open(out, "w", encoding="utf-8-sig", newline="") as f:  # BOM so Excel shows Thai correctly
        w = csv.DictWriter(f, fieldnames=SCORE_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"Scored {sum(r['status'] != 'unscored' for r in rows)} of {len(rows)} responses -> {out}")
    return out
