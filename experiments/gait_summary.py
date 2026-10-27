"""Data for lesson 8 (docs/data/gait.json): per walker, score and gait on the same 100
comparison tracks (evaluate.py --seed-kind compare output), plus a replay on one track.

  python experiments/gait_summary.py v3-f20                 # one walker (before v4 exists)
  python experiments/gait_summary.py v3-f20 v4-walk         # both + paired comparison
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402

LABEL = {"v3-f20": "skipping walker (v3)", "v4-walk": "shared controller + clock (v4)"}


def summary(run):
    ev = ROOT / "results" / f"{run}_compare.json"
    if not ev.exists():
        subprocess.run([sys.executable, str(ROOT / "evaluate.py"), "--run", f"runs/{run}", "--episodes", "100",
                        "--seed-kind", "compare", "--out", str(ev)], check=True, capture_output=True)
    E = json.loads(ev.read_text())
    eps = E["episodes_detail"]
    meta = json.loads((ROOT / "runs" / run / "best" / "meta.json").read_text())
    rep = ROOT / "docs" / "data" / "episodes" / f"gait_{run}.json"
    subprocess.run([sys.executable, str(ROOT / "experiments" / "export_episode.py"), "--run", f"runs/{run}", "--seed", "30000008",
                    "--label", LABEL.get(run, run), "--out", str(rep)], check=True, capture_output=True)
    m = lambda k: round(float(np.mean([e[k] for e in eps])), 3)
    return {"run": run, "label": LABEL.get(run, run), "policy": meta.get("policy", "mlp"), "generation": meta.get("generation"),
            "held_out": meta.get("mean"), "episodes": len(eps), "mean": E["mean"], "std": E["std"], "falls": E["falls"],
            "finished": E["finished"], "crossings": m("crossings"), "lead_changes": m("lead_changes"),
            "front2_share": m("front2_share"), "zero_change_share": round(float(np.mean([e["lead_changes"] == 0 for e in eps])), 3),
            "replay": f"episodes/gait_{run}.json"}


def main():
    runs = sys.argv[1:]
    out = {"walkers": [summary(r) for r in runs]}
    if len(runs) == 2:
        c = subprocess.run([sys.executable, str(ROOT / "experiments" / "compare.py"), str(ROOT / "results" / f"{runs[1]}_compare.json"),
                            str(ROOT / "results" / f"{runs[0]}_compare.json"), "--out", str(ROOT / "results" / "gait_compare.json")],
                           check=True, capture_output=True)
        c = json.loads((ROOT / "results" / "gait_compare.json").read_text())
        out["compare"] = {k: c[k] for k in ("mean_a", "mean_b", "diff_mean", "diff_ci95", "a_better_on", "episodes", "falls_a", "falls_b")}
        out["compare"]["a"], out["compare"]["b"] = runs[1], runs[0]
    p = ROOT / "docs" / "data" / "gait.json"
    p.write_text(json.dumps(out, indent=1))
    for w in out["walkers"]:
        print(f"{w['run']}: score {w['mean']} ± {w['std']}, falls {w['falls']}, crossings {w['crossings']}, "
              f"front-foot changes {w['lead_changes']}, foot 2 in front {w['front2_share']:.0%}, never changes lead on {w['zero_change_share']:.0%}")
    print("->", p)


if __name__ == "__main__":
    main()
