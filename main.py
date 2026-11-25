"""Image-extraction AI API test harness.

  python main.py prepare thaiocrbench --per-task 5     # build images/ + data/ground_truth.jsonl
  python main.py run    --run-name test1               # 1+2: send images, save JSON responses
  python main.py score  --run-name test1               # 3: compare with correct answers
  python main.py report --run-name test1               # 4: accuracy report
  python main.py all    --run-name test1               # run + score + report
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):  # print Thai text on Windows consoles
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _run_dir(args) -> Path:
    name = args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    return Path(args.results_dir) / name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="download a dataset sample into images/ + ground truth file")
    p.add_argument("dataset", choices=["thaiocrbench"])
    p.add_argument("--per-task", type=int, default=5, help="images per task type (default 5)")
    p.add_argument("--all", action="store_true", help="take the whole dataset")
    p.add_argument("--limit", type=int, help="stop after this many images in total")
    p.add_argument("--images-dir", default="images")
    p.add_argument("--ground-truth", default="data/ground_truth.jsonl")

    for name in ("run", "score", "report", "all"):
        sp = sub.add_parser(name)
        sp.add_argument("--config", default="config.yaml")
        sp.add_argument("--run-name", help="results/<run-name>/ (default: timestamp; required for score/report)")
        sp.add_argument("--results-dir", default="results")
        if name in ("run", "all"):
            sp.add_argument("--limit", type=int, help="only send the first N images")
            sp.add_argument("--rerun", action="store_true", help="ignore saved responses and send everything again")

    args = parser.parse_args()

    if args.command == "prepare":
        from datasets_adapters import thaiocrbench
        thaiocrbench.prepare(Path(args.images_dir), Path(args.ground_truth),
                             per_task=None if args.all else args.per_task, limit=args.limit)
        return

    from src.config import load_config
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise SystemExit(f"{cfg_path} not found. Copy config.example.yaml to config.yaml and edit it.")
    cfg = load_config(cfg_path)

    if args.command in ("score", "report") and not args.run_name:
        raise SystemExit("--run-name is required for score/report")
    run_dir = _run_dir(args)
    print(f"Run folder: {run_dir}")

    if args.command in ("run", "all"):
        from src.runner import run
        run(cfg, run_dir, limit=args.limit, rerun=args.rerun)
    if args.command in ("score", "all"):
        from src.scorer import score
        score(cfg, run_dir)
    if args.command in ("report", "all"):
        from src.report import report
        report(cfg, run_dir)


if __name__ == "__main__":
    main()
