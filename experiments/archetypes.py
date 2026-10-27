"""Measure the behaviours that the failed starts got stuck in, for the fitness-shaping demo
in lesson 5. Each archetype is a real policy from runs/, scored with the TRAINING rollout
(early stop on) on training-range tracks, keeping the raw parts so the page can recompute
fitness for any fall cost / stall cost:  fitness = raw + (100 - fall_cost)*fell - stall_cost*stalled.

  python experiments/archetypes.py --out docs/data/archetypes.json
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402

import common  # noqa: E402

# What each run converged to = its last CMA-mean snapshot (best/ can be an early lucky pick:
# v0 and v2 share a seed and both kept the same gen-9 candidate as champion).
ARCHETYPES = [
    ("random", "random network (gen 0)", "runs/v0-dive/milestones/gen0"),
    ("dive", "dive forward and fall (v0, end)", "runs/v0-dive"),
    ("lunge", "lunge and fall (v1, end)", "runs/v1-lunge"),
    ("early", "fall early, save effort (v2, end)", "runs/v2-torque"),
    ("stand", "creep, then stand (v3 fall cost 40, end)", "runs/v3-f40"),
    ("walk", "walk (v3 fall cost 20, champion)", None),
]


def load(path, walker):
    p = ROOT / (path or walker)
    snaps = sorted((p / "snapshots").glob("mean_*.npy")) if path and (p / "snapshots").exists() else []
    if snaps:
        sizes = json.loads((p / "run_config.json").read_text())["effective"]["sizes"]
        return common.MLP(sizes, np.load(snaps[-1])), {}, snaps[-1].name
    pol, meta = common.load_policy(p)
    return pol, meta, "best"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--walker", default="runs/v3-f20")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--out", default=str(ROOT / "docs" / "data" / "archetypes.json"))
    args = ap.parse_args()
    env = common.make_env()
    seeds = list(range(500_000, 500_000 + args.episodes))   # training range, fixed for reproducibility
    out = []
    for key, name, path in ARCHETYPES:
        pol, meta, which = load(path, args.walker)
        eps = [common.run_episode(pol, env, s, early_stop=True, action_repeat=meta.get("action_repeat", 1)) for s in seeds]
        row = {"key": key, "name": name, "run": path or args.walker, "params": which,
               "raw": round(float(np.mean([e["reward"] for e in eps])), 2),
               "fell": round(float(np.mean([e["fell"] for e in eps])), 3),
               "stalled": round(float(np.mean([e["stalled"] for e in eps])), 3),
               "x": round(float(np.mean([e["x"] for e in eps])), 2),
               "steps": round(float(np.mean([e["steps"] for e in eps])), 1)}
        out.append(row)
        print(row)
    Path(args.out).write_text(json.dumps({"episodes": args.episodes, "archetypes": out}, indent=1))
    print("->", args.out)


if __name__ == "__main__":
    main()
