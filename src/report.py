"""Step 4: build the accuracy report (Markdown, HTML and summary.json)."""
from __future__ import annotations

import csv
import html
import json
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from .metrics import METRIC_NAMES

TABLE_METRICS = ["score", "exact_match", "edit_similarity", "anls", "char_f1", "json_f1", "cer"]


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return statistics.fmean(vals) if vals else None


def _pct(v):
    return "-" if v is None else f"{v * 100:.1f}%"


def _fmt(metric, v):
    if v is None:
        return "-"
    return f"{v:.3f}" if metric == "cer" else _pct(v)


def _load(run_dir: Path):
    with open(run_dir / "scores.csv", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ["score", *METRIC_NAMES, "latency_ms"]:
            r[k] = float(r[k]) if r.get(k) not in (None, "") else None
    meta_path = run_dir / "run_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return rows, meta


def _usage_totals(run_dir: Path) -> dict:
    totals: dict = defaultdict(int)
    path = run_dir / "responses.jsonl"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                for k, v in (json.loads(line).get("usage") or {}).items():
                    if isinstance(v, (int, float)):
                        totals[k] += v
    return dict(totals)


def _group(rows, key):
    groups = defaultdict(list)
    for r in rows:
        groups[r[key] or "(none)"].append(r)
    out = []
    for name, items in sorted(groups.items()):
        entry = {"name": name, "n": len(items), "primary_metric": ", ".join(sorted({i["primary_metric"] for i in items}))}
        for m in TABLE_METRICS:
            entry[m] = _mean(i[m] for i in items)
        out.append(entry)
    return out


def build_summary(rows, meta, usage, worst_n):
    scored = [r for r in rows if r["status"] != "unscored"]
    latencies = sorted(r["latency_ms"] for r in rows if r["latency_ms"])
    summary = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "run": meta,
        "counts": {"total": len(rows), "scored": len(scored),
                   "errors": sum(1 for r in rows if r["error"]),
                   "unscored": len(rows) - len(scored)},
        "overall": {m: _mean(r[m] for r in scored) for m in TABLE_METRICS},
        "latency_ms": {"mean": _mean(latencies),
                       "p50": latencies[len(latencies) // 2] if latencies else None,
                       "p95": latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))] if latencies else None},
        "usage": usage,
        "by_task": _group(scored, "task"),
        "by_category": _group(scored, "category"),
        "worst": [{k: r[k] for k in ("id", "image", "task", "score", "primary_metric", "expected", "predicted", "error")}
                  for r in sorted(scored, key=lambda r: r["score"] or 0)[:worst_n]],
    }
    return summary


def _md_table(entries, title):
    lines = [f"| {title} | N | Primary metric | Score | Exact match | Edit sim. | ANLS | Char F1 | JSON F1 | CER |",
             "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for e in entries:
        lines.append(f"| {e['name']} | {e['n']} | {e['primary_metric']} | " +
                     " | ".join(_fmt(m, e[m]) for m in TABLE_METRICS) + " |")
    return "\n".join(lines)


def _clip(text, n=300):
    text = (text or "").replace("\n", " ⏎ ")
    return text if len(text) <= n else text[:n] + "…"


def write_markdown(s, path: Path):
    o, c, run = s["overall"], s["counts"], s["run"]
    md = [f"# AI accuracy report",
          "",
          f"- **Provider / model:** {run.get('provider')} / {run.get('model')}",
          f"- **Ground truth:** {run.get('ground_truth')}",
          f"- **Run:** {run.get('started', '-')} → {run.get('finished', '-')}",
          f"- **Images:** {c['total']} total, {c['scored']} scored, {c['errors']} with errors, {c['unscored']} unscored",
          f"- **Latency:** mean {s['latency_ms']['mean'] or 0:.0f} ms, p50 {s['latency_ms']['p50'] or 0:.0f} ms, p95 {s['latency_ms']['p95'] or 0:.0f} ms",
          f"- **Token usage:** {', '.join(f'{k}={v:,}' for k, v in s['usage'].items()) or '-'}",
          "",
          f"## Overall accuracy: **{_pct(o['score'])}**",
          "",
          "Score = mean of each item's primary metric (set per task in config). "
          "Other columns are averages over all scored items (JSON F1 only over JSON answers). CER: lower is better.",
          "",
          "| Score | Exact match | Edit sim. | ANLS | Char F1 | JSON F1 | CER |",
          "|---:|---:|---:|---:|---:|---:|---:|",
          "| " + " | ".join(_fmt(m, o[m]) for m in TABLE_METRICS) + " |",
          "",
          "## By task", "", _md_table(s["by_task"], "Task"), "",
          "## By category", "", _md_table(s["by_category"], "Category"), "",
          f"## Lowest-scoring {len(s['worst'])} items", "",
          "| ID | Task | Score | Expected | Predicted | Error |", "|---|---|---:|---|---|---|"]
    for w in s["worst"]:
        cells = [w["id"], w["task"], _pct(w["score"]), _clip(w["expected"]), _clip(w["predicted"]), w["error"] or ""]
        md.append("| " + " | ".join(str(x).replace("|", "\\|") for x in cells) + " |")
    path.write_text("\n".join(md) + "\n", encoding="utf-8")


def _html_table(entries, title):
    head = "".join(f"<th>{h}</th>" for h in [title, "N", "Primary metric", "Score", "Exact match", "Edit sim.", "ANLS", "Char F1", "JSON F1", "CER"])
    body = ""
    for e in entries:
        score = e["score"] or 0
        bar = f'<div class="bar"><span style="width:{score * 100:.1f}%"></span></div>'
        body += (f"<tr><td>{html.escape(e['name'])}</td><td class=num>{e['n']}</td><td>{html.escape(e['primary_metric'])}</td>"
                 f"<td class=num>{_fmt('score', e['score'])}{bar}</td>" +
                 "".join(f"<td class=num>{_fmt(m, e[m])}</td>" for m in TABLE_METRICS[1:]) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def write_html(s, path: Path):
    o, c, run = s["overall"], s["counts"], s["run"]
    worst = "".join(
        f"<tr><td>{html.escape(w['id'])}</td><td>{html.escape(w['task'] or '')}</td><td class=num>{_pct(w['score'])}</td>"
        f"<td><pre>{html.escape(w['expected'] or '')}</pre></td><td><pre>{html.escape(w['predicted'] or '')}</pre></td>"
        f"<td>{html.escape(w['error'] or '')}</td></tr>" for w in s["worst"])
    tiles = "".join(f'<div class="tile"><div class="label">{label}</div><div class="value">{value}</div></div>' for label, value in [
        ("Overall accuracy", _pct(o["score"])), ("Exact match", _pct(o["exact_match"])),
        ("Edit similarity", _pct(o["edit_similarity"])), ("CER", _fmt("cer", o["cer"])),
        ("Scored / total", f"{c['scored']} / {c['total']}"), ("Errors", str(c["errors"])),
        ("p95 latency", f"{(s['latency_ms']['p95'] or 0) / 1000:.1f} s")])
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>AI accuracy report</title>
<style>
:root {{ --bg:#fff; --fg:#1d1d1f; --muted:#6b6b70; --line:#e3e3e6; --accent:#2f6fde; --tile:#f6f6f8; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#17171a; --fg:#ececef; --muted:#9a9aa2; --line:#33333a; --accent:#6d9cf0; --tile:#222227; }} }}
body {{ font-family: system-ui, "Segoe UI", "Noto Sans Thai", sans-serif; background:var(--bg); color:var(--fg); max-width:1200px; margin:2rem auto; padding:0 1rem; line-height:1.45; }}
h1 {{ margin-bottom:.2rem }} .meta {{ color:var(--muted); font-size:.9rem }}
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:.75rem; margin:1.5rem 0 }}
.tile {{ background:var(--tile); border-radius:10px; padding:.8rem 1rem }} .label {{ color:var(--muted); font-size:.8rem }} .value {{ font-size:1.5rem; font-weight:600; font-variant-numeric:tabular-nums }}
table {{ border-collapse:collapse; width:100%; margin:.5rem 0 2rem; font-size:.9rem }}
th, td {{ border-bottom:1px solid var(--line); padding:.45rem .6rem; text-align:left; vertical-align:top }}
th {{ color:var(--muted); font-weight:500 }} .num {{ text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap }}
.bar {{ height:4px; background:var(--line); border-radius:2px; margin-top:4px }} .bar span {{ display:block; height:100%; background:var(--accent); border-radius:2px }}
pre {{ white-space:pre-wrap; word-break:break-word; margin:0; max-height:12rem; overflow:auto; font-size:.8rem }}
</style></head><body>
<h1>AI accuracy report</h1>
<div class="meta">{html.escape(str(run.get('provider')))} / {html.escape(str(run.get('model')))} · ground truth: {html.escape(str(run.get('ground_truth')))} · generated {s['generated']}</div>
<div class="tiles">{tiles}</div>
<p class="meta">Overall accuracy is the mean of each item's primary metric (set per task in config). CER: lower is better.</p>
<h2>By task</h2>{_html_table(s['by_task'], 'Task')}
<h2>By category</h2>{_html_table(s['by_category'], 'Category')}
<h2>Lowest-scoring items</h2>
<table><thead><tr><th>ID</th><th>Task</th><th>Score</th><th>Expected</th><th>Predicted</th><th>Error</th></tr></thead><tbody>{worst}</tbody></table>
</body></html>"""
    path.write_text(doc, encoding="utf-8")


def report(cfg: dict, run_dir: Path) -> Path:
    if not (run_dir / "scores.csv").exists():
        raise SystemExit(f"{run_dir / 'scores.csv'} not found - run the `score` step first.")
    rows, meta = _load(run_dir)
    worst_n = int(cfg.get("report", {}).get("worst_examples", 10))
    s = build_summary(rows, meta, _usage_totals(run_dir), worst_n)
    (run_dir / "summary.json").write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(s, run_dir / "report.md")
    write_html(s, run_dir / "report.html")
    print(f"Overall accuracy: {_pct(s['overall']['score'])}  ({s['counts']['scored']} scored, {s['counts']['errors']} errors)")
    print(f"Report: {run_dir / 'report.md'} and {run_dir / 'report.html'}")
    return run_dir / "report.md"
