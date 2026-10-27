"""Build the data the GitHub Pages site in docs/ reads.

  python make_site.py --run runs/v3            # training status -> docs/data/progress.json
  python make_site.py --run runs/v3 --episodes # also re-export the champion's replay for the lessons

The pages themselves are static HTML in docs/; this script only writes JSON next to them.
"""
import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import common

DOCS = common.ROOT / "docs"


def downsample(rows, n=400):
    if len(rows) <= n:
        return rows
    step = len(rows) / n
    return [rows[int(i * step)] for i in range(n)] + [rows[-1]]


def progress(run_dir):
    rows = list(csv.DictReader(open(run_dir / "stats.csv")))
    num = lambda v: None if v in ("", None) else float(v)
    pts = [{"g": int(r["generation"]), "best": num(r["best_fitness"]), "mean": num(r["mean_fitness"]),
            "sigma": num(r["sigma"]), "falls": num(r["fall_frac"]), "x": num(r["mean_x"]),
            "hof": num(r["hof_best"]), "champ": num(r["best_score"]),
            "steps": int(r["env_steps_total"]), "wall": num(r["wall_seconds"])} for r in rows]
    hof = [p for p in pts if p["hof"] is not None]
    meta = json.loads((run_dir / "best" / "meta.json").read_text()) if (run_dir / "best" / "meta.json").exists() else {}
    cfg = json.loads((run_dir / "run_config.json").read_text())
    return {
        "run": run_dir.name, "updated": time.strftime("%Y-%m-%d %H:%M"),
        "generation": pts[-1]["g"] + 1 if pts else 0,
        "env_steps": pts[-1]["steps"] if pts else 0,
        "hours": round((pts[-1]["wall"] or 0) / 3600, 2) if pts else 0,
        "champion": {k: meta.get(k) for k in ("mean", "std", "falls", "generation", "label", "mean_x")},
        "config": cfg["effective"], "history": cfg["history"],
        "curve": downsample(pts), "hof": hof,
        "milestones": sorted(p.name for p in (run_dir / "milestones").iterdir()) if (run_dir / "milestones").exists() else [],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--episodes", action="store_true", help="re-export the champion replay JSON")
    args = ap.parse_args()
    run_dir = Path(args.run)
    out = DOCS / "data" / "progress.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(progress(run_dir), separators=(",", ":")))
    print(f"-> {out}")
    if args.episodes and (run_dir / "best" / "params.npy").exists():
        subprocess.run([sys.executable, "experiments/export_episode.py", "--run", str(run_dir),
                        "--label", "CMA-ES champion", "--out", str(DOCS / "data" / "episodes" / "champion.json")],
                       check=True)


if __name__ == "__main__":
    main()
