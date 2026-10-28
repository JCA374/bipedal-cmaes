"""Build docs/data/curriculum.json for the site's Curriculum tab.

  python make_curriculum_page.py            # usually called by curriculum_publish.sh

Reads docs/data/curriculum_attempts.json (which attempts exist, their log files), each stage's
run dir (stats.csv, gait.csv, best/), the curriculum logs (stage starts, graduation checks,
outcomes), the journal, and runs the goal check on each attempt's latest walking stage.
Re-exports a replay of a stage champion only when its params changed.
"""
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "docs" / "data"
STAGES = [("1-balance", "Balance", "stay up and level, 500-step episodes"),
          ("2-march", "Step", "alternating front-foot changes, even in place, 600 steps"),
          ("3-walk", "Walk", "distance with the gait tax, full episodes"),
          ("4-fast", "Walk fast", "the real reward plus the gait tax")]
EXTRA_RUNS = ["v7-alt"]   # comparison runs shown below the attempts
sys.path.insert(0, str(ROOT / "experiments"))


def downsample(rows, n=300):
    if len(rows) <= n:
        return rows
    k = len(rows) / n
    return [rows[int(i * k)] for i in range(n)] + [rows[-1]]


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def run_summary(run):
    d = ROOT / "runs" / run
    if not (d / "stats.csv").exists():
        return None
    st = list(csv.DictReader(open(d / "stats.csv")))
    gt = list(csv.DictReader(open(d / "gait.csv"))) if (d / "gait.csv").exists() else []
    gmap = {r["generation"]: r for r in gt}
    curve = [{"g": int(r["generation"]), "best": num(r["best_fitness"]), "mean": num(r["mean_fitness"]),
              "champ": num(r["best_score"]), "falls": num(r["fall_frac"]), "x": num(r["mean_x"]),
              "lead": num(gmap.get(r["generation"], {}).get("lead_changes")),
              "front2": num(gmap.get(r["generation"], {}).get("front2_share"))} for r in st]
    meta = json.loads((d / "best" / "meta.json").read_text()) if (d / "best" / "meta.json").exists() else {}
    cfg = json.loads((d / "run_config.json").read_text()).get("effective", {}) if (d / "run_config.json").exists() else {}
    rep = DATA / "episodes" / f"cur_{run}.json"
    pm = d / "best" / "params.npy"
    if pm.exists() and (not rep.exists() or rep.stat().st_mtime < pm.stat().st_mtime):
        subprocess.run([sys.executable, str(ROOT / "experiments" / "export_episode.py"), "--run", str(d), "--seed", "30000008",
                        "--label", f"{run} champion", "--out", str(rep)], capture_output=True)
    return {"run": run, "generations": len(st), "hours": round((num(st[-1]["wall_seconds"]) or 0) / 3600, 2),
            "sigma": num(st[-1]["sigma"]), "champion": {k: meta.get(k) for k in ("generation", "label", "mean", "selection", "falls",
                                                                               "front2_share", "lead_changes", "abs_lead_mean", "mean_x")},
            "config": cfg, "curve": downsample(curve), "replay": f"episodes/{rep.name}" if rep.exists() else None}


def parse_log(path):
    """{run: {"started", "ended", "outcome", "checks": [..]}} from a curriculum log."""
    out, cur = {}, None
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        m = re.match(r"\[(\d\d:\d\d:\d\d)\] === stage (\S+)", line)
        if m:
            cur = m.group(2)
            out[cur] = {"started": m.group(1), "checks": [], "outcome": "running"}
            continue
        m = re.match(r"\[(\d\d:\d\d:\d\d)\] === (\S+) done: (.+)", line)
        if m and m.group(2) in out:
            out[m.group(2)].update(ended=m.group(1), outcome=m.group(3))
            continue
        m = re.match(r"\[(\d\d:\d\d:\d\d)\]\s+check: (\{.*\})", line)
        if m and cur:
            try:
                out[cur]["checks"].append({"time": m.group(1), **json.loads(m.group(2))})
            except json.JSONDecodeError:
                pass
    return out


def main():
    from goal_check import GOAL, measure
    attempts = json.loads((DATA / "curriculum_attempts.json").read_text())
    res = {"updated": time.strftime("%a %d %b %H:%M"), "deadline": "2026-10-30T17:00", "goal": GOAL, "stages": STAGES,
           "attempts": [], "extra": [], "journal": json.loads((DATA / "curriculum_journal.json").read_text())[::-1]}
    for a in attempts:
        log = parse_log(ROOT / a["log"])
        stages = []
        for key, name, desc in STAGES:
            run = f"{a['prefix']}{key}"
            s = run_summary(run)
            info = log.get(run, {"outcome": "pending" if s is None else "unknown", "checks": []})
            stages.append({"key": key, "name": name, "desc": desc, "run": run, "summary": s, **info})
        goal = None
        last = next((st for st in reversed(stages) if st["summary"] and st["key"] in ("3-walk", "4-fast")), None)
        if last:
            cache = DATA / f"goal_{last['run']}.json"
            pm = ROOT / "runs" / last["run"] / "best" / "params.npy"
            if pm.exists() and (not cache.exists() or cache.stat().st_mtime < pm.stat().st_mtime):
                cache.write_text(json.dumps(measure(ROOT / "runs" / last["run"])))
            goal = json.loads(cache.read_text()) if cache.exists() else None
        res["attempts"].append({**a, "stages": stages, "goal": goal})
    for run in EXTRA_RUNS:
        s = run_summary(run)
        if s:
            res["extra"].append(s)
    (DATA / "curriculum.json").write_text(json.dumps(res, separators=(",", ":")))
    print(f"-> {DATA / 'curriculum.json'} ({len(res['attempts'])} attempts, {len(res['journal'])} journal entries)")


if __name__ == "__main__":
    main()
