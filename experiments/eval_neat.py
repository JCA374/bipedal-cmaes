"""Score the NEAT champion with this repo's rollout (fresh Box2D world per episode), in the
same JSON format as evaluate.py, so experiments/compare.py can pair it with a CMA-ES policy.
Needs NEAT's venv:

  ../bipedal-neat/.venv/bin/python experiments/eval_neat.py --neat ../bipedal-neat/runs/baseline \
      --seed-kind compare --episodes 100 --out results/neat_compare.json
"""
import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
import numpy as np  # noqa: E402

import common  # noqa: E402
from export_episode import load_neat  # noqa: E402

_W = {}


def _init(run):
    _W["net"], _ = load_neat(run)
    _W["env"] = common.make_env()


def _ep(seed):
    return {"seed": seed, **common.run_episode(_W["net"], _W["env"], seed, early_stop=False)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neat", required=True)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed-kind", default="compare")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    seeds = common.seeds(args.seed_kind, args.episodes)
    with mp.get_context("fork").Pool(args.workers, initializer=_init, initargs=(args.neat,)) as pool:
        eps = pool.map(_ep, seeds, chunksize=1)
    rw = np.array([e["reward"] for e in eps])
    res = {"run": "NEAT " + args.neat, "episodes": len(eps), "mean": round(float(rw.mean()), 2),
           "std": round(float(rw.std()), 2), "falls": int(sum(e["fell"] for e in eps)),
           "episodes_detail": [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in e.items()} for e in eps]}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(f"NEAT: mean {res['mean']} ± {res['std']}, falls {res['falls']}/{len(eps)} -> {args.out}")


if __name__ == "__main__":
    main()
