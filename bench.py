"""Phase 0 benchmark: env steps/s with the real MLP on N workers, plus RSS per worker.

  python bench.py --workers 1 4 6
"""
import argparse
import multiprocessing as mp
import os
import resource
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

import common
from train import mem_avail_mb


def _work(args):
    n_steps, seed, sizes = args
    rng = np.random.default_rng(seed)
    pol = common.MLP(sizes)
    env = common.make_env()
    steps, ep, t0 = 0, 0, time.time()
    while steps < n_steps:
        # Small weights so episodes last a while (random nets with σ=0.5 fall fast).
        pol.set_params(rng.normal(0, 0.1, pol.n_params))
        steps += common.run_episode(pol, env, seed * 1000 + ep, early_stop=False)["steps"]
        ep += 1
    return steps, time.time() - t0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 4, 6])
    ap.add_argument("--steps", type=int, default=20000, help="steps per worker")
    ap.add_argument("--hidden", type=int, nargs="+", default=[32])
    args = ap.parse_args()
    sizes = (24, *args.hidden, 4)
    print(f"MemAvailable before: {mem_avail_mb()} MB, sizes {sizes}")
    for w in args.workers:
        t0 = time.time()
        with mp.get_context("fork").Pool(w) as pool:
            res = pool.map(_work, [(args.steps, i + 1, sizes) for i in range(w)])
            avail = mem_avail_mb()
        wall = time.time() - t0
        total = sum(r[0] for r in res)
        print(f"workers={w}: total {total / wall:7.0f} steps/s | per worker "
              f"{sum(r[0] / r[1] for r in res) / w:6.0f} | RSS/worker {max(r[2] for r in res):4.0f} MB "
              f"| MemAvailable {avail} MB")


if __name__ == "__main__":
    main()
