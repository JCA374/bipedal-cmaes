"""Score a policy on unseen terrains: full episodes, raw reward, no early stop.

  python evaluate.py --run runs/baseline --episodes 100
  python evaluate.py --run runs/baseline/milestones/100 --episodes 100 --out runs/baseline/eval/m100.json
"""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

import common

_W = {}


def _init(run):
    _W["policy"], meta = common.load_policy(run)
    _W["ar"] = meta.get("action_repeat", 1)
    _W["env"] = common.make_env()


def _episode(seed):
    return {"seed": seed, **common.run_episode(_W["policy"], _W["env"], seed, early_stop=False,
                                               action_repeat=_W["ar"])}


def evaluate(run, seeds, workers=4):
    with mp.get_context("fork").Pool(workers, initializer=_init, initargs=(str(run),)) as pool:
        eps = pool.map(_episode, seeds, chunksize=1)
    rw = np.array([e["reward"] for e in eps])
    timeouts = [e for e in eps if not e["fell"] and e["steps"] == 1600]
    return {
        "run": str(run), "episodes": len(eps), "seeds": [seeds[0], seeds[-1]],
        "mean": round(float(rw.mean()), 2), "std": round(float(rw.std()), 2),
        "median": round(float(np.median(rw)), 2), "min": round(float(rw.min()), 2),
        "max": round(float(rw.max()), 2),
        "falls": int(sum(e["fell"] for e in eps)), "fell_frac": round(float(np.mean([e["fell"] for e in eps])), 3),
        "n_ge_300": int((rw >= 300).sum()),
        "timeouts": len(timeouts),
        "timeout_mean_x": round(float(np.mean([e["x"] for e in timeouts])), 2) if timeouts else None,
        "finished": int(sum(1 for e in eps if not e["fell"] and e["steps"] < 1600)),
        "mean_steps": round(float(np.mean([e["steps"] for e in eps])), 1),
        "episodes_detail": [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in e.items()}
                            for e in eps],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed-kind", default="evaluate", choices=list(common.SEED_RANGES))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    res = evaluate(args.run, common.seeds(args.seed_kind, args.episodes), args.workers)
    out = Path(args.out) if args.out else Path(args.run) / "eval" / "best.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "episodes_detail"}, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
