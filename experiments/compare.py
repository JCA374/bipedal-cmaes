"""Paired comparison of two evaluations on the SAME tracks (common random numbers), with a
bootstrap 95% CI on the mean difference. Inputs are evaluate.py / eval_neat.py JSON files.

  python experiments/compare.py results/cma_compare.json results/neat_compare.json --out results/compare.json
"""
import argparse
import json
from pathlib import Path

import numpy as np


def boot_ci(d, n=10000):
    m = np.random.default_rng(0).choice(d, (n, len(d))).mean(1)
    return [round(float(np.percentile(m, 2.5)), 2), round(float(np.percentile(m, 97.5)), 2)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    A, B = (json.loads(Path(p).read_text()) for p in (args.a, args.b))
    ra = {e["seed"]: e for e in A["episodes_detail"]}
    rb = {e["seed"]: e for e in B["episodes_detail"]}
    seeds = sorted(set(ra) & set(rb))
    assert len(seeds) == len(ra) == len(rb), "the two evaluations must use the same tracks"
    d = np.array([ra[s]["reward"] - rb[s]["reward"] for s in seeds])
    res = {"a": A["run"], "b": B["run"], "episodes": len(seeds),
           "mean_a": round(float(np.mean([ra[s]["reward"] for s in seeds])), 2),
           "mean_b": round(float(np.mean([rb[s]["reward"] for s in seeds])), 2),
           "falls_a": int(sum(ra[s]["fell"] for s in seeds)), "falls_b": int(sum(rb[s]["fell"] for s in seeds)),
           "diff_mean": round(float(d.mean()), 2), "diff_ci95": boot_ci(d),
           "a_better_on": int((d > 0).sum()),
           # gait (present for evaluations made after common.run_episode tracked it)
           **({f"gait_{w}": {k: round(float(np.mean([r[s][k] for s in seeds])), 3)
                             for k in ("crossings", "lead_changes", "front2_share")}
               for w, r in (("a", ra), ("b", rb))} if "crossings" in ra[seeds[0]] and "crossings" in rb[seeds[0]] else {}),
           "pairs": [[s, round(ra[s]["reward"], 1), round(rb[s]["reward"], 1)] for s in seeds]}
    print(json.dumps({k: v for k, v in res.items() if k != "pairs"}, indent=1))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
