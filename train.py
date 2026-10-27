"""CMA-ES on the weights of a fixed tanh MLP for BipedalWalker-v3.

  python train.py --name baseline --hours 2
  python train.py --name baseline --resume --hours 1

Training fitness = mean over K terrains (the same K for every candidate in a
generation: common random numbers) of the reward with the env's -100 fall penalty
replaced by --fall-cost (default 0), minus a stall cost when the early-stop detector
fires. Run v0 used the raw -100 and got stuck diving forward (see TRAINING_LOG.md). Every --hof-every
generations the CMA mean and the best few candidates are re-scored on held-out
terrains with full episodes; the champion is the best on those.

Settings for the next resumed chunk can be changed with runs/<name>/overrides.json,
e.g. {"k": 8, "popsize": 64, "reason": "..."}. Resume always applies the CURRENT
arguments and overrides; the checkpoint holds only the optimiser and the counters.
"""
import argparse
import csv
import gzip
import json
import multiprocessing as mp
import os
import pickle
import signal
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import cma
import numpy as np

import common

MILESTONES = [("first_positive", 0.0), ("100", 100.0), ("200", 200.0), ("300", 300.0)]
STATS_FIELDS = ["generation", "best_fitness", "mean_fitness", "median_fitness", "sigma",
                "popsize", "restarts", "fall_frac", "stall_frac", "mean_x", "hof_mean",
                "hof_best", "best_score", "env_steps_gen", "env_steps_total",
                "steps_saved_total", "gen_seconds", "wall_seconds", "mem_avail_mb"]
OVERRIDABLE = {"k": int, "popsize": int, "hof_seeds": int, "hof_top": int,
               "stall_penalty": float, "speed_bonus": float, "fall_cost": float}

# ---------------------------------------------------------------- workers
_W = {}


def _init_worker(sizes, action_repeat):
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # parent handles Ctrl-C
    _W["env"] = common.make_env()
    _W["policy"] = common.MLP(sizes)
    _W["action_repeat"] = action_repeat


def eval_theta(task):
    """task = (theta, seeds, early_stop). Returns per-episode results."""
    theta, seeds, early_stop = task
    pol = _W["policy"]
    pol.set_params(theta)
    return [common.run_episode(pol, _W["env"], s, early_stop=early_stop,
                               action_repeat=_W["action_repeat"]) for s in seeds]


def train_fitness(eps, args):
    """Training-only shaping. Reported scores are always raw rewards of full episodes."""
    f = []
    for e in eps:
        r = e["reward"]
        if e["stalled"]:
            r -= args.stall_penalty
        if e["fell"]:
            r += 100.0 - args.fall_cost  # swap the env's -100 for the training fall cost
        r += args.speed_bonus * e["x"] / 88.0
        f.append(r)
    return float(np.mean(f))


# ---------------------------------------------------------------- helpers
def mem_avail_mb():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    return -1


def gen_seeds(base_seed, generation, k):
    rng = np.random.default_rng(base_seed + generation)
    return [int(x) for x in rng.integers(*common.SEED_RANGES["train"], size=k)]


def new_es(x0, sigma, popsize, seed):
    return cma.CMAEvolutionStrategy(x0, sigma, {
        "popsize": popsize, "seed": seed + 1, "verbose": -9,
        # Noisy fitness: don't let flat-fitness / tolfun checks end the run early.
        "tolfun": 0, "tolfunhist": 0, "tolflatfitness": 1000, "tolstagnation": 10**6,
        "tolx": 1e-4, "maxiter": 10**9,
    })


def save_checkpoint(path, es, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wb", compresslevel=5) as f:
        # Deliberately NOT storing args: resume always uses the current ones.
        pickle.dump({"es": es, "state": state}, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def load_checkpoint(path):
    with gzip.open(path, "rb") as f:
        d = pickle.load(f)
    return d["es"], d["state"]


def apply_overrides(args, run_dir):
    ov_path = run_dir / "overrides.json"
    if not (args.resume and ov_path.exists()):
        return {}
    ov = json.loads(ov_path.read_text())
    for k, typ in OVERRIDABLE.items():
        if k in ov:
            setattr(args, k, typ(ov[k]))
    print(f"Overrides from {ov_path}: {ov}")
    return ov


def effective_params(args, sizes):
    return {"sizes": list(sizes), "n_params": common.n_params(sizes), "popsize": args.popsize,
            "k": args.k, "sigma0": args.sigma0, "stall_penalty": args.stall_penalty,
            "fall_cost": args.fall_cost, "speed_bonus": args.speed_bonus,
            "action_repeat": args.action_repeat, "hof_seeds": args.hof_seeds,
            "hof_top": args.hof_top, "early_stop": not args.no_early_stop}


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--hidden", type=int, nargs="+", default=[32], help="hidden layer sizes")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--generations", type=int, default=None, help="stop at this total generation")
    ap.add_argument("--hours", type=float, default=None, help="wall-clock budget for this invocation")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--popsize", type=int, default=None, help="λ (default 48; on resume: keep the current one)")
    ap.add_argument("--k", type=int, default=5, help="terrains per candidate per generation")
    ap.add_argument("--sigma0", type=float, default=0.5)
    ap.add_argument("--stall-penalty", type=float, default=0.0,
                    help="training cost when the stall detector fires (v1 used 100 and got stuck, TRAINING_LOG.md)")
    ap.add_argument("--fall-cost", type=float, default=20.0,
                    help="training cost of a fall, replacing the env's -100 (100 = raw). With 0, fitness is "
                         "progress minus torque, so walking further before a fall always scores higher")
    ap.add_argument("--speed-bonus", type=float, default=0.0, help="+bonus * x_final/88")
    ap.add_argument("--action-repeat", type=int, default=1)
    ap.add_argument("--no-early-stop", action="store_true")
    ap.add_argument("--hof-every", type=int, default=10)
    ap.add_argument("--hof-top", type=int, default=3)
    ap.add_argument("--hof-seeds", type=int, default=20)
    ap.add_argument("--snapshot-every", type=int, default=10, help="save the CMA mean (for the video)")
    ap.add_argument("--checkpoint-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stop-at", type=float, default=None, help="stop once the champion reaches this")
    ap.add_argument("--runs-dir", default=str(common.ROOT / "runs"))
    args = ap.parse_args(argv)

    run_dir = Path(args.runs_dir) / args.name
    ckpt_path = run_dir / "checkpoints" / "latest.pkl.gz"
    overrides = apply_overrides(args, run_dir)

    if args.resume:
        if not ckpt_path.exists():
            raise SystemExit(f"--resume: no checkpoint at {ckpt_path}")
        es, state = load_checkpoint(ckpt_path)
        sizes = tuple(state["sizes"])  # the architecture is fixed for a run
        args.action_repeat = state.get("action_repeat", args.action_repeat)
        # A popsize override applies once (IPOP restarts may double λ afterwards).
        want = args.popsize if args.popsize != state.get("popsize_applied") else None
        if args.popsize is not None:
            state["popsize_applied"] = args.popsize
        if want is not None and es.popsize != want:
            # pycma can't change λ in place: restart at the current mean and σ with the new λ.
            print(f"popsize {es.popsize} -> {want}: restarting CMA at the current mean")
            es = new_es(es.mean, es.sigma, want, args.seed + 1000 * (state["restarts"] + 1))
            state["restarts"] += 1
            state["restart_log"].append({"generation": state["generation"], "reason": "popsize override",
                                         "popsize": want})
        args.popsize = es.popsize
        print(f"Resumed {args.name} at generation {state['generation']} (σ={es.sigma:.3f}, λ={es.popsize})")
    else:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise SystemExit(f"{run_dir} exists; use --resume or a new --name")
        sizes = (common.N_OBS, *args.hidden, common.N_ACT)
        args.popsize = args.popsize or 48
        es = new_es(np.zeros(common.n_params(sizes)), args.sigma0, args.popsize, args.seed)
        state = {"generation": 0, "sizes": list(sizes), "action_repeat": args.action_repeat,
                 "env_steps_total": 0, "steps_saved_total": 0, "wall_seconds": 0.0,
                 "best_score": None, "milestones": {}, "hof": [], "seed": args.seed,
                 "restarts": 0, "restart_log": [], "param_history": [], "popsize_applied": args.popsize}
    # Hard requirement: at least one hidden layer.
    assert len(sizes) >= 3, "the policy must have >= 1 hidden layer"
    run_dir.mkdir(parents=True, exist_ok=True)

    params = effective_params(args, sizes)
    print("Effective params:", json.dumps(params))
    hist = state["param_history"]
    if not hist or {k: v for k, v in hist[-1].items() if k not in ("generation", "reason")} != params:
        hist.append({"generation": state["generation"], **params,
                     **({"reason": overrides["reason"]} if overrides.get("reason") else {})})
    with open(run_dir / "run_config.json", "w") as f:
        json.dump({"args": vars(args), "effective": params, "history": hist}, f, indent=2)

    stats_path = run_dir / "stats.csv"
    new_stats = not stats_path.exists()
    stats_f = open(stats_path, "a", newline="")
    stats_w = csv.DictWriter(stats_f, fieldnames=STATS_FIELDS)
    if new_stats:
        stats_w.writeheader()

    pool = mp.get_context("fork").Pool(args.workers, initializer=_init_worker,
                                       initargs=(sizes, args.action_repeat))
    t_start = time.time()
    wall0 = state["wall_seconds"]

    def wall():
        return wall0 + time.time() - t_start

    def save_best(dirpath, theta, meta):
        common.save_policy(dirpath, theta, sizes, {**meta, "action_repeat": args.action_repeat})

    def hall_of_fame(cands, generation):
        """cands = [(label, theta, train_fitness)]. Full episodes on held-out terrains."""
        seeds = common.seeds("hof", args.hof_seeds)
        flat = pool.map(eval_theta, [(th, [s], False) for _, th, _ in cands for s in seeds], chunksize=1)
        scores = []
        for i, (label, theta, tf) in enumerate(cands):
            eps = [e[0] for e in flat[i * len(seeds):(i + 1) * len(seeds)]]
            state["env_steps_total"] += sum(e["steps"] for e in eps)
            rw = [e["reward"] for e in eps]
            mean, std = float(np.mean(rw)), float(np.std(rw))
            entry = {"generation": generation, "label": label, "mean": mean, "std": std,
                     "falls": int(sum(e["fell"] for e in eps)),
                     "mean_x": float(np.mean([e["x"] for e in eps])), "train_fitness": tf}
            state["hof"].append(entry)
            scores.append(mean)
            if state["best_score"] is None or mean > state["best_score"]:
                state["best_score"] = mean
                meta = {**entry, "env_steps_total": state["env_steps_total"],
                        "wall_seconds": wall(), "held_out_seeds": seeds,
                        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}
                save_best(run_dir / "best", theta, meta)
                print(f"  HOF: new champion ({label}) mean={mean:.1f} ±{std:.1f} falls={entry['falls']}")
                for name, thr in MILESTONES:
                    if mean > thr and name not in state["milestones"]:
                        state["milestones"][name] = meta
                        save_best(run_dir / "milestones" / name, theta, meta)
                        print(f"  MILESTONE {name} reached at generation {generation}")
        state["hof"] = sorted(state["hof"], key=lambda e: e["mean"], reverse=True)[:30]
        with open(run_dir / "hof.json", "w") as f:
            json.dump(state["hof"], f, indent=2)
        return scores

    def one_generation():
        gen = state["generation"]
        t0 = time.time()
        sols = es.ask()
        seeds = gen_seeds(state["seed"], gen, args.k)
        results = pool.map(eval_theta, [(th, seeds, not args.no_early_stop) for th in sols], chunksize=1)
        fits = np.array([train_fitness(eps, args) for eps in results])
        es.tell(sols, (-fits).tolist())  # CMA minimises
        all_eps = [e for eps in results for e in eps]
        steps = sum(e["steps"] for e in all_eps)
        saved = sum(e["saved"] for e in all_eps)
        state["env_steps_total"] += steps
        state["steps_saved_total"] += saved
        if gen == 0:  # what a random network does, for the "where it started" clip
            save_best(run_dir / "milestones" / "gen0", sols[int(np.argmax(fits))],
                      {"generation": 0, "train_fitness": float(fits.max())})
        if args.snapshot_every and gen % args.snapshot_every == 0:
            (run_dir / "snapshots").mkdir(exist_ok=True)
            np.save(run_dir / "snapshots" / f"mean_{gen:05d}.npy", es.mean)
        hof = None
        if args.hof_every and (gen + 1) % args.hof_every == 0:
            top = np.argsort(-fits)[:args.hof_top]
            cands = [("mean", es.mean.copy(), None)] + \
                    [(f"cand{i}", sols[i], float(fits[i])) for i in top]
            hof = hall_of_fame(cands, gen)
        state["generation"] = gen + 1
        row = {"generation": gen, "best_fitness": round(float(fits.max()), 2),
               "mean_fitness": round(float(fits.mean()), 2),
               "median_fitness": round(float(np.median(fits)), 2),
               "sigma": round(float(es.sigma), 5), "popsize": es.popsize,
               "restarts": state["restarts"],
               "fall_frac": round(np.mean([e["fell"] for e in all_eps]), 3),
               "stall_frac": round(np.mean([e["stalled"] for e in all_eps]), 3),
               "mean_x": round(float(np.mean([e["x"] for e in all_eps])), 2),
               "hof_mean": "" if hof is None else round(hof[0], 2),
               "hof_best": "" if hof is None else round(max(hof), 2),
               "best_score": "" if state["best_score"] is None else round(state["best_score"], 2),
               "env_steps_gen": steps, "env_steps_total": state["env_steps_total"],
               "steps_saved_total": state["steps_saved_total"],
               "gen_seconds": round(time.time() - t0, 1), "wall_seconds": round(wall(), 1),
               "mem_avail_mb": mem_avail_mb()}
        stats_w.writerow(row)
        stats_f.flush()
        print(f"gen {gen:4d} best {row['best_fitness']:7.1f} mean {row['mean_fitness']:7.1f} "
              f"σ {es.sigma:.3f} falls {row['fall_frac']:.2f} stalls {row['stall_frac']:.2f} "
              f"x {row['mean_x']:5.1f} steps {steps/1e3:5.0f}k {row['gen_seconds']:5.1f}s "
              f"mem {row['mem_avail_mb']}MB" + ("" if hof is None else f" | HOF mean {hof[0]:.1f}"))

    def checkpoint():
        state["wall_seconds"] = wall()
        save_checkpoint(ckpt_path, es, state)

    try:
        while True:
            if args.generations is not None and state["generation"] >= args.generations:
                break
            if args.hours is not None and time.time() - t_start >= args.hours * 3600:
                print("Time budget reached")
                break
            if args.stop_at is not None and (state["best_score"] or -1e9) >= args.stop_at:
                print(f"Champion reached {state['best_score']:.1f} >= --stop-at {args.stop_at}")
                break
            one_generation()
            stop = es.stop()
            if stop:
                # IPOP restart: double λ, start again from the champion (or the mean).
                best = run_dir / "best" / "params.npy"
                x0 = np.load(best) if best.exists() else es.mean
                args.popsize = es.popsize * 2
                state["restarts"] += 1
                state["restart_log"].append({"generation": state["generation"], "reason": str(stop),
                                             "popsize": args.popsize})
                print(f"  RESTART #{state['restarts']} ({stop}): λ -> {args.popsize}")
                es = new_es(x0, args.sigma0 / 2, args.popsize, args.seed + 1000 * state["restarts"])
            if state["generation"] % args.checkpoint_every == 0:
                checkpoint()
    except KeyboardInterrupt:
        print("\nInterrupted, saving checkpoint")
    finally:
        checkpoint()
        pool.terminate()
        stats_f.close()
    print(f"Done at generation {state['generation']}. Champion held-out mean: {state['best_score']}")
    return es, state


if __name__ == "__main__":
    main()
