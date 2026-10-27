"""What the champion's network does while it walks (data for lesson 6 and the explainer).

  python experiments/inspect_policy.py --run runs/v3-f20 --episodes 20 --out docs/data/inside.json

1. Trace: one typical episode (median of 9 record-range tracks): inputs, the 32 hidden
   activations, actions, foot contacts and hull x for every step.
2. Gait: steps where leg 1's foot touches down split the walk into strides. Each hidden
   unit's activation is averaged by stride phase (0..1).
3. Ablation: each hidden unit in turn is frozen at its average activation; the drop in mean
   score on --episodes held-out tracks (record range) says how much walking depends on it.
4. Input importance: each input in turn is frozen at its average; same score drop.
All episodes are full length with raw reward.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402

import common  # noqa: E402


class Probe:
    """MLP wrapper that can freeze one hidden unit or one input at a fixed value."""

    def __init__(self, mlp, hidden=None, inp=None, value=0.0):
        self.mlp, self.hidden, self.inp, self.value = mlp, hidden, inp, value

    def activate(self, x):
        if self.inp is not None:
            x = x.copy()
            x[self.inp] = self.value
        (W1, b1), (W2, b2) = self.mlp.layers[0], self.mlp.layers[-1]
        h = np.tanh(W1 @ x + b1)
        if self.hidden is not None:
            h = h.copy()
            h[self.hidden] = self.value
        return np.tanh(W2 @ h + b2)


_W = {}


def _init(run):
    _W["mlp"], meta = common.load_policy(run)
    assert len(_W["mlp"].sizes) == 3, "inspect_policy handles one hidden layer"
    _W["ar"] = meta.get("action_repeat", 1)
    _W["env"] = common.make_env()


def _score(task):
    kind, idx, value, seed = task
    pol = Probe(_W["mlp"], **({"hidden": idx} if kind == "hidden" else {"inp": idx} if kind == "input" else {}), value=value)
    r = common.run_episode(pol, _W["env"], seed, early_stop=False, action_repeat=_W["ar"])
    return kind, idx, r["reward"], r["fell"], r["x"]


def trace(mlp, seed, ar=1):
    env = common.make_env()
    rec = {"obs": [], "hid": [], "act": [], "x": []}

    class Rec:
        def activate(self, x):
            a, (h,) = mlp.activate(x, return_hidden=True)
            rec["obs"].append(x)
            rec["hid"].append(h)
            rec["act"].append(a)
            return a

    res = common.run_episode(Rec(), env, seed, early_stop=False, action_repeat=ar,
                             on_step=lambda o, a, tot, e: rec["x"].append(e.unwrapped.hull.position.x))
    return {k: np.array(v) for k, v in rec.items()}, res


def gait(tr, n_bins=20):
    """Stride phase from leg-1 touchdowns (contact 0 -> 1)."""
    c = tr["obs"][:, 8] > 0.5
    down = np.where(c[1:] & ~c[:-1])[0] + 1
    if len(down) < 3:
        return None
    phase = np.full(len(c), np.nan)
    for a, b in zip(down[:-1], down[1:]):
        phase[a:b] = (np.arange(a, b) - a) / (b - a)
    ok = ~np.isnan(phase)
    bins = np.minimum((phase[ok] * n_bins).astype(int), n_bins - 1)
    hid = tr["hid"][ok]
    prof = np.array([[hid[bins == k, j].mean() for k in range(n_bins)] for j in range(hid.shape[1])])
    stride = np.diff(down)
    return {"touchdowns": down.tolist(), "stride_steps_median": float(np.median(stride)),
            "stride_steps_iqr": [float(np.percentile(stride, 25)), float(np.percentile(stride, 75))],
            "profile": np.round(prof, 3).tolist(),
            "action_profile": np.round(np.array([[tr["act"][ok][bins == k, j].mean() for k in range(n_bins)]
                                                  for j in range(4)]), 3).tolist(),
            "rhythmic": np.round(prof.max(1) - prof.min(1), 3).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "docs" / "data" / "inside.json"))
    args = ap.parse_args()

    mlp, meta = common.load_policy(args.run)
    ar = meta.get("action_repeat", 1)
    env = common.make_env()
    cands = common.seeds("record", 9)
    rs = [common.run_episode(mlp, env, s, early_stop=False, action_repeat=ar)["reward"] for s in cands]
    seed = cands[int(np.argsort(rs)[len(rs) // 2])]
    tr, res = trace(mlp, seed, ar)
    hid_mean, obs_mean = tr["hid"].mean(0), tr["obs"].mean(0)
    print(f"trace: seed {seed}, reward {res['reward']:.1f}, {res['steps']} steps")

    seeds = common.seeds("record", args.episodes, offset=100)
    tasks = [("base", None, 0.0, s) for s in seeds]
    tasks += [("hidden", j, float(hid_mean[j]), s) for j in range(mlp.sizes[1]) for s in seeds]
    tasks += [("input", i, float(obs_mean[i]), s) for i in range(24) for s in seeds]
    with mp.get_context("fork").Pool(args.workers, initializer=_init, initargs=(args.run,)) as pool:
        out = pool.map(_score, tasks, chunksize=4)
    base = np.mean([r for k, _, r, _, _ in out if k == "base"])

    def drops(kind, n):
        res_ = []
        for j in range(n):
            rr = [(r, f) for k, i, r, f, _ in out if k == kind and i == j]
            res_.append({"index": j, "mean": round(float(np.mean([r for r, _ in rr])), 1),
                         "drop": round(float(base - np.mean([r for r, _ in rr])), 1),
                         "falls": int(sum(f for _, f in rr))})
        return res_

    hid_d, inp_d = drops("hidden", mlp.sizes[1]), drops("input", 24)
    for d in inp_d:
        d["name"] = common.INPUT_NAMES[d["index"]]
    (W1, b1), (W2, b2) = mlp.layers
    with np.errstate(invalid="ignore", divide="ignore"):  # constant inputs (e.g. a lidar that never hits)
        corr_in = np.nan_to_num(np.corrcoef(np.c_[tr["hid"], tr["obs"]].T)[:mlp.sizes[1], mlp.sizes[1]:])
    every = 2
    result = {
        "run": args.run, "champion": {k: meta.get(k) for k in ("mean", "std", "generation", "label")},
        "trace": {"seed": seed, "reward": round(res["reward"], 1), "steps": res["steps"], "every": every,
                  "obs": np.round(tr["obs"][::every], 2).tolist(), "hid": np.round(tr["hid"][::every], 2).tolist(),
                  "act": np.round(tr["act"][::every], 2).tolist(), "x": np.round(tr["x"][::every], 2).tolist()},
        "gait": gait(tr),
        "ablation": {"episodes": args.episodes, "base": round(float(base), 1), "hidden": hid_d, "inputs": inp_d},
        "saturation": round(float(np.mean(np.abs(tr["hid"]) > 0.95)), 3),
        "hidden_input_corr": np.round(corr_in, 2).tolist(),
        "W1": np.round(W1, 3).tolist(), "b1": np.round(b1, 3).tolist(),
        "W2": np.round(W2, 3).tolist(), "b2": np.round(b2, 3).tolist(),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, separators=(",", ":")))
    top = sorted(hid_d, key=lambda d: -d["drop"])[:6]
    print(f"base {base:.1f} on {args.episodes} tracks; saturation {result['saturation']}")
    print("hidden units that matter most:", [(d["index"], d["drop"]) for d in top])
    print("inputs that matter most:", [(d["name"], d["drop"]) for d in sorted(inp_d, key=lambda d: -d["drop"])[:8]])
    if result["gait"]:
        print("stride (steps):", result["gait"]["stride_steps_median"], result["gait"]["stride_steps_iqr"])
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
